"""Validation-only isotonic calibration for one selected support classifier."""

import json
import math
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import sklearn
from sklearn.isotonic import IsotonicRegression

from evaluation.calibration_validation import (
    CALIBRATION_IMPLEMENTATION_VERSION,
    CalibrationValidationError,
    FittedSupportCalibrator,
    IsotonicPolicy,
    SelectedSupportModel,
    canonical_json_bytes,
    eligible_population_hash,
    load_validation_predictions,
    sha256_bytes,
    validate_isotonic_policy,
    validate_selected_support_model,
)
from evaluation.weights import compute_stay_weights
from experiments.search_governance import canonical_sha256


def fit_support_calibrator(
    selected_model: SelectedSupportModel,
    *,
    policy: IsotonicPolicy,
    mode: str = "synthetic",
) -> FittedSupportCalibrator:
    """Fit isotonic regression from the selected model's validation artifact only."""

    validate_selected_support_model(selected_model, mode=mode)
    validate_isotonic_policy(policy, mode=mode)
    records = load_validation_predictions(selected_model)
    stay_weights = compute_stay_weights(
        [record.stay_id for record in records], [True] * len(records)
    )
    if policy.sample_weighting == "stay_balanced":
        weights = stay_weights.weights
    else:
        weights = tuple(1.0 for _ in records)
    estimator = IsotonicRegression(
        y_min=policy.y_min,
        y_max=policy.y_max,
        increasing=policy.increasing,
        out_of_bounds=policy.out_of_bounds,
    )
    estimator.fit(
        np.asarray([record.raw_probability for record in records], dtype=float),
        np.asarray([record.label for record in records], dtype=float),
        sample_weight=np.asarray(weights, dtype=float),
    )
    positive_examples = sum(record.label for record in records)
    positive_stays = len({record.stay_id for record in records if record.label == 1})
    return FittedSupportCalibrator(
        artifact_version="isotonic_support_v1",
        implementation_version=CALIBRATION_IMPLEMENTATION_VERSION,
        implementation_library="sklearn.isotonic.IsotonicRegression",
        implementation_library_version=sklearn.__version__,
        policy=policy,
        x_thresholds=tuple(float(value) for value in estimator.X_thresholds_),
        y_thresholds=tuple(float(value) for value in estimator.y_thresholds_),
        selected_family=selected_model.family,
        selected_run_id=selected_model.run_id,
        selected_candidate_id=selected_model.candidate_id,
        selected_config_hash=selected_model.config_hash,
        selected_model_sha256=selected_model.artifact_sha256,
        validation_prediction_ref=selected_model.validation_prediction_ref,
        validation_prediction_sha256=selected_model.validation_prediction_sha256,
        validation_split_hash=selected_model.split_hash,
        feature_version=selected_model.feature_version,
        label_version=selected_model.label_version,
        preprocessing_hash=selected_model.preprocessing_hash,
        eligible_population_hash=eligible_population_hash(records),
        counts={
            "N_examples": len(records),
            "N_ICU_stays": stay_weights.n_icu_stays,
            "N_positive_examples": positive_examples,
            "N_positive_stays": positive_stays,
        },
        code_commit=selected_model.code_commit,
        calibration_mode=mode,
    )


def apply_support_calibrator(
    calibrator: FittedSupportCalibrator, raw_probabilities: Sequence[float]
):
    converted = []
    lower = calibrator.x_thresholds[0]
    upper = calibrator.x_thresholds[-1]
    for value in raw_probabilities:
        probability = float(value)
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise CalibrationValidationError("raw probabilities must be finite within [0,1]")
        if calibrator.policy.out_of_bounds == "raise" and not lower <= probability <= upper:
            raise CalibrationValidationError("raw probability is outside fitted isotonic range")
        converted.append(probability)
    calibrated = np.interp(
        np.asarray(converted, dtype=float),
        np.asarray(calibrator.x_thresholds, dtype=float),
        np.asarray(calibrator.y_thresholds, dtype=float),
    )
    if any(not 0.0 <= float(value) <= 1.0 for value in calibrated):
        raise CalibrationValidationError("calibrator produced an invalid probability")
    return tuple(float(value) for value in calibrated)


def calibrator_payload(calibrator: FittedSupportCalibrator) -> Mapping[str, object]:
    payload = asdict(calibrator)
    payload["x_thresholds"] = list(calibrator.x_thresholds)
    payload["y_thresholds"] = list(calibrator.y_thresholds)
    return payload


def save_calibrator_artifact(path, calibrator: FittedSupportCalibrator):
    artifact = Path(path)
    artifact.parent.mkdir(parents=True, exist_ok=True)
    content = canonical_json_bytes(calibrator_payload(calibrator))
    artifact.write_bytes(content)
    artifact_hash = sha256_bytes(content)
    metadata_path = artifact.with_name(artifact.name + ".metadata.json")
    metadata_path.write_bytes(
        canonical_json_bytes(
            {
                "artifact_ref": str(artifact),
                "artifact_sha256": artifact_hash,
                "artifact_version": calibrator.artifact_version,
                "selected_model_sha256": calibrator.selected_model_sha256,
                "validation_prediction_sha256": calibrator.validation_prediction_sha256,
                "test_accessed": False,
            }
        )
    )
    return {"artifact_ref": str(artifact), "artifact_sha256": artifact_hash}


def load_calibrator_artifact(path, *, expected_sha256: str) -> FittedSupportCalibrator:
    artifact = Path(path)
    if not artifact.is_file() or sha256_bytes(artifact.read_bytes()) != expected_sha256:
        raise CalibrationValidationError("calibrator artifact SHA-256 mismatch")
    try:
        payload = json.loads(artifact.read_text(encoding="utf-8"))
        policy = IsotonicPolicy(**payload.pop("policy"))
        payload["policy"] = policy
        payload["x_thresholds"] = tuple(payload["x_thresholds"])
        payload["y_thresholds"] = tuple(payload["y_thresholds"])
        calibrator = FittedSupportCalibrator(**payload)
    except (ValueError, TypeError, KeyError) as error:
        raise CalibrationValidationError("invalid calibrator artifact schema") from error
    if len(calibrator.x_thresholds) != len(calibrator.y_thresholds) or not calibrator.x_thresholds:
        raise CalibrationValidationError("invalid isotonic thresholds")
    if (
        calibrator.artifact_version != "isotonic_support_v1"
        or calibrator.implementation_version != CALIBRATION_IMPLEMENTATION_VERSION
        or calibrator.implementation_library != "sklearn.isotonic.IsotonicRegression"
        or calibrator.test_accessed is not False
    ):
        raise CalibrationValidationError("incompatible calibrator implementation metadata")
    if any(
        current > following
        for current, following in zip(calibrator.x_thresholds, calibrator.x_thresholds[1:])
    ) or any(not 0.0 <= value <= 1.0 for value in calibrator.y_thresholds):
        raise CalibrationValidationError("invalid isotonic threshold ordering/range")
    validate_isotonic_policy(calibrator.policy, mode=calibrator.calibration_mode)
    return calibrator


def validate_calibrator_binding(
    calibrator: FittedSupportCalibrator, selected_model: SelectedSupportModel
) -> None:
    expected = {
        "selected_family": selected_model.family,
        "selected_run_id": selected_model.run_id,
        "selected_candidate_id": selected_model.candidate_id,
        "selected_config_hash": selected_model.config_hash,
        "selected_model_sha256": selected_model.artifact_sha256,
        "validation_prediction_sha256": selected_model.validation_prediction_sha256,
        "validation_split_hash": selected_model.split_hash,
        "feature_version": selected_model.feature_version,
        "label_version": selected_model.label_version,
        "preprocessing_hash": selected_model.preprocessing_hash,
    }
    for field, value in expected.items():
        if getattr(calibrator, field) != value:
            raise CalibrationValidationError("calibrator/model compatibility mismatch: " + field)


def selected_support_from_manifest(
    manifest: Mapping[str, object],
    *,
    validation_prediction_sha256: str,
    selection_manifest_ref=None,
    selection_manifest_sha256=None,
) -> SelectedSupportModel:
    """Extract the one frozen support selection without changing its identity."""

    from evaluation.select import validate_selection_stage_manifest

    validate_selection_stage_manifest(manifest)
    support = manifest["tasks"]["organ_support"]
    prediction_ref = support.get("validation_prediction_ref")
    if not prediction_ref:
        raise CalibrationValidationError("selected support entry lacks validation predictions")
    return SelectedSupportModel(
        task="organ_support",
        family=support["family"],
        run_id=support["run_id"],
        candidate_id=support["candidate_id"],
        config_hash=support["config_hash"],
        artifact_ref=support["artifact_ref"],
        artifact_sha256=support["artifact_sha256"],
        split_hash=support["split_hash"],
        feature_version=support["feature_version"],
        label_version=support["label_version"],
        preprocessing_hash=support["preprocessing_hash"],
        code_commit=support["code_commit"],
        selection_metric=support["selection_metric"],
        probability_type="raw_uncalibrated",
        validation_prediction_ref=prediction_ref,
        validation_prediction_sha256=validation_prediction_sha256,
        selection_mode=manifest["selection_mode"],
        test_accessed=False,
        selection_manifest_ref=selection_manifest_ref,
        selection_manifest_sha256=selection_manifest_sha256,
    )


def promote_selection_manifest(
    stage_manifest: Mapping[str, object],
    selected_model: SelectedSupportModel,
    *,
    calibrator_artifact: Mapping[str, str],
    threshold_artifact: Mapping[str, str],
):
    """Attach compatible Phase-13 references without changing selected models."""

    from evaluation.select import validate_selection_stage_manifest
    from evaluation.threshold import (
        load_threshold_artifact,
        validate_threshold_binding,
    )

    validate_selection_stage_manifest(stage_manifest)
    calibrator = load_calibrator_artifact(
        calibrator_artifact["artifact_ref"],
        expected_sha256=calibrator_artifact["artifact_sha256"],
    )
    threshold = load_threshold_artifact(
        threshold_artifact["artifact_ref"],
        expected_sha256=threshold_artifact["artifact_sha256"],
    )
    validate_calibrator_binding(calibrator, selected_model)
    validate_threshold_binding(
        threshold,
        selected_model,
        calibrator_sha256=calibrator_artifact["artifact_sha256"],
        calibrator_ref=calibrator_artifact["artifact_ref"],
    )
    support_before = stage_manifest["tasks"]["organ_support"]
    identity = {
        "family": selected_model.family,
        "run_id": selected_model.run_id,
        "candidate_id": selected_model.candidate_id,
        "config_hash": selected_model.config_hash,
        "artifact_sha256": selected_model.artifact_sha256,
    }
    if any(support_before.get(field) != value for field, value in identity.items()):
        raise CalibrationValidationError("Phase-13 promotion cannot change support selection")
    promoted = deepcopy(stage_manifest)
    promoted.pop("manifest_sha256", None)
    support = promoted["tasks"]["organ_support"]
    support.update(
        {
            "calibration_status": "FROZEN_VALIDATION_ONLY",
            "calibrator": dict(calibrator_artifact),
            "threshold": {
                **dict(threshold_artifact),
                "value": threshold.threshold,
                "comparator": "greater_than_or_equal",
            },
            "validation_prediction_sha256": selected_model.validation_prediction_sha256,
        }
    )
    mode = promoted["selection_mode"]
    promoted["manifest_version"] = "selected_models_calibrated_v1"
    promoted["status"] = (
        "SYNTHETIC_CALIBRATION_FRAMEWORK_EVIDENCE_ONLY"
        if mode == "synthetic"
        else "MODEL_SELECTION_CALIBRATION_FROZEN"
    )
    promoted["serving_ready"] = False
    promoted["test_accessed"] = False
    return {**promoted, "manifest_sha256": canonical_sha256(promoted)}


def validate_calibrated_manifest(manifest: Mapping[str, object]) -> None:
    from evaluation.selection_validation import EXPLANATION_METHOD, sha256_file
    from evaluation.threshold import load_threshold_artifact, validate_threshold_binding

    if manifest.get("manifest_version") != "selected_models_calibrated_v1":
        raise CalibrationValidationError("calibrated manifest version mismatch")
    if manifest.get("serving_ready") is not False or manifest.get("test_accessed") is not False:
        raise CalibrationValidationError("Phase-13 manifest cannot access test or claim serving readiness")
    tasks = manifest.get("tasks")
    if not isinstance(tasks, Mapping) or set(tasks) != {
        "recovery",
        "icu_stay_time",
        "organ_support",
    }:
        raise CalibrationValidationError("calibrated manifest task map is incomplete")
    for task, entry in tasks.items():
        if entry.get("explanation_method") != EXPLANATION_METHOD.get(entry.get("family")):
            raise CalibrationValidationError("manifest explanation route changed")
        artifact = Path(str(entry.get("artifact_ref", "")))
        if not artifact.is_file() or sha256_file(artifact) != entry.get("artifact_sha256"):
            raise CalibrationValidationError("manifest selected model artifact/hash mismatch")
    support = tasks["organ_support"]
    if support.get("calibration_status") != "FROZEN_VALIDATION_ONLY":
        raise CalibrationValidationError("support calibration is not frozen")
    calibrator_ref = support.get("calibrator")
    threshold_ref = support.get("threshold")
    if not isinstance(calibrator_ref, Mapping) or not isinstance(threshold_ref, Mapping):
        raise CalibrationValidationError("support calibrator/threshold references are required")
    selected = SelectedSupportModel(
        task="organ_support",
        family=support["family"],
        run_id=support["run_id"],
        candidate_id=support["candidate_id"],
        config_hash=support["config_hash"],
        artifact_ref=support["artifact_ref"],
        artifact_sha256=support["artifact_sha256"],
        split_hash=support["split_hash"],
        feature_version=support["feature_version"],
        label_version=support["label_version"],
        preprocessing_hash=support["preprocessing_hash"],
        code_commit=support["code_commit"],
        selection_metric=support["selection_metric"],
        probability_type="raw_uncalibrated",
        validation_prediction_ref=support["validation_prediction_ref"],
        validation_prediction_sha256=support["validation_prediction_sha256"],
        selection_mode=manifest["selection_mode"],
        test_accessed=False,
    )
    calibrator = load_calibrator_artifact(
        calibrator_ref["artifact_ref"], expected_sha256=calibrator_ref["artifact_sha256"]
    )
    threshold = load_threshold_artifact(
        threshold_ref["artifact_ref"], expected_sha256=threshold_ref["artifact_sha256"]
    )
    validate_calibrator_binding(calibrator, selected)
    validate_threshold_binding(
        threshold,
        selected,
        calibrator_sha256=calibrator_ref["artifact_sha256"],
        calibrator_ref=calibrator_ref["artifact_ref"],
    )
    if threshold_ref.get("value") != threshold.threshold:
        raise CalibrationValidationError("manifest threshold value mismatch")
    supplied_hash = manifest.get("manifest_sha256")
    content = dict(manifest)
    content.pop("manifest_sha256", None)
    if supplied_hash != canonical_sha256(content):
        raise CalibrationValidationError("calibrated manifest SHA-256 mismatch")
