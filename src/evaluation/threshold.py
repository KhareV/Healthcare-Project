"""Stay-balanced validation-F1 threshold selection after isotonic calibration."""

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence, Tuple

from evaluation.calibrate import (
    apply_support_calibrator,
    load_calibrator_artifact,
    validate_calibrator_binding,
)
from evaluation.calibration_validation import (
    CalibrationValidationError,
    FittedSupportCalibrator,
    SelectedSupportModel,
    canonical_json_bytes,
    load_validation_predictions,
    sha256_bytes,
    validate_selected_support_model,
)
from evaluation.metrics import METRIC_IMPLEMENTATION_VERSION, weighted_confusion_metrics
from evaluation.weights import compute_stay_weights
from experiments.search_governance import canonical_sha256


THRESHOLD_IMPLEMENTATION_VERSION = "support_validation_f1_threshold_v1"
THRESHOLD_SEARCH_UNLOCKED = "UNLOCKED ENGINEERING PARAMETER — THRESHOLD SEARCH PROCEDURE"
THRESHOLD_TIE_BLOCKER = "BLOCKED — F1 THRESHOLD TIE POLICY REQUIRED"


@dataclass(frozen=True)
class ThresholdSearchPolicy:
    policy_version: str
    candidate_method: str
    tie_policy: str
    comparator: str = "greater_than_or_equal"
    approved_for_real: bool = False


@dataclass(frozen=True)
class SupportThresholdResult:
    artifact_version: str
    implementation_version: str
    metric_implementation_version: str
    policy: ThresholdSearchPolicy
    threshold: float
    validation_metrics: Mapping[str, float]
    counts: Mapping[str, int]
    selected_family: str
    selected_run_id: str
    selected_model_sha256: str
    calibrator_ref: str
    calibrator_sha256: str
    validation_prediction_ref: str
    validation_prediction_sha256: str
    calibrated_prediction_hash: str
    validation_split_hash: str
    feature_version: str
    label_version: str
    code_commit: str
    threshold_mode: str
    test_accessed: bool = False


def validate_threshold_policy(policy: ThresholdSearchPolicy, *, mode: str) -> None:
    if policy.candidate_method not in (
        "unique_calibrated_probabilities",
        "unique_calibrated_probabilities_plus_endpoints",
    ):
        raise CalibrationValidationError(THRESHOLD_SEARCH_UNLOCKED)
    if policy.tie_policy not in ("lowest_threshold", "highest_threshold", "block"):
        raise CalibrationValidationError(THRESHOLD_TIE_BLOCKER)
    if policy.comparator != "greater_than_or_equal":
        raise CalibrationValidationError("threshold comparator must match Phase-9 >= semantics")
    if not policy.policy_version:
        raise CalibrationValidationError("threshold policy version is required")
    if mode == "real" and policy.approved_for_real is not True:
        raise CalibrationValidationError(
            "BLOCKED — REAL CALIBRATION PREREQUISITES INCOMPLETE; "
            + THRESHOLD_SEARCH_UNLOCKED
            + "; "
            + THRESHOLD_TIE_BLOCKER
        )


def _candidate_thresholds(probabilities, policy):
    candidates = set(float(value) for value in probabilities)
    if policy.candidate_method == "unique_calibrated_probabilities_plus_endpoints":
        candidates.update((0.0, 1.0))
    return tuple(sorted(candidates))


def choose_support_threshold(
    selected_model: SelectedSupportModel,
    calibrator: FittedSupportCalibrator,
    *,
    calibrator_ref: str,
    calibrator_sha256: str,
    policy: ThresholdSearchPolicy,
    mode: str = "synthetic",
) -> SupportThresholdResult:
    validate_threshold_policy(policy, mode=mode)
    validate_selected_support_model(selected_model, mode=mode)
    validate_calibrator_binding(calibrator, selected_model)
    loaded_calibrator = load_calibrator_artifact(
        calibrator_ref, expected_sha256=calibrator_sha256
    )
    if loaded_calibrator != calibrator:
        raise CalibrationValidationError("threshold calibrator object/artifact mismatch")
    if calibrator.calibration_mode != mode:
        raise CalibrationValidationError("calibrator and threshold modes differ")
    records = load_validation_predictions(selected_model)
    raw = tuple(record.raw_probability for record in records)
    calibrated = apply_support_calibrator(calibrator, raw)
    labels = tuple(record.label for record in records)
    weights = compute_stay_weights(
        [record.stay_id for record in records], [True] * len(records)
    )
    scored = []
    for threshold in _candidate_thresholds(calibrated, policy):
        metrics = weighted_confusion_metrics(labels, calibrated, weights.weights, threshold)
        if math.isfinite(metrics["f1"]):
            scored.append((threshold, metrics))
    if not scored:
        raise CalibrationValidationError("threshold search produced no finite F1 candidate")
    best_f1 = max(item[1]["f1"] for item in scored)
    tied = [item for item in scored if item[1]["f1"] == best_f1]
    if len(tied) > 1 and policy.tie_policy == "block":
        raise CalibrationValidationError(THRESHOLD_TIE_BLOCKER)
    if policy.tie_policy == "highest_threshold":
        threshold, metrics = max(tied, key=lambda item: item[0])
    else:
        threshold, metrics = min(tied, key=lambda item: item[0])
    positive_examples = sum(labels)
    positive_stays = len({record.stay_id for record in records if record.label == 1})
    calibrated_hash = canonical_sha256(
        tuple(
            {"row_id": record.row_id, "calibrated_probability": probability}
            for record, probability in zip(records, calibrated)
        )
    )
    return SupportThresholdResult(
        artifact_version="threshold_support_v1",
        implementation_version=THRESHOLD_IMPLEMENTATION_VERSION,
        metric_implementation_version=METRIC_IMPLEMENTATION_VERSION,
        policy=policy,
        threshold=float(threshold),
        validation_metrics=metrics,
        counts={
            "N_examples": weights.n_examples,
            "N_ICU_stays": weights.n_icu_stays,
            "N_positive_examples": positive_examples,
            "N_positive_stays": positive_stays,
        },
        selected_family=selected_model.family,
        selected_run_id=selected_model.run_id,
        selected_model_sha256=selected_model.artifact_sha256,
        calibrator_ref=calibrator_ref,
        calibrator_sha256=calibrator_sha256,
        validation_prediction_ref=selected_model.validation_prediction_ref,
        validation_prediction_sha256=selected_model.validation_prediction_sha256,
        calibrated_prediction_hash=calibrated_hash,
        validation_split_hash=selected_model.split_hash,
        feature_version=selected_model.feature_version,
        label_version=selected_model.label_version,
        code_commit=selected_model.code_commit,
        threshold_mode=mode,
    )


def apply_support_threshold(probabilities: Sequence[float], threshold: float) -> Tuple[int, ...]:
    threshold_value = float(threshold)
    if not math.isfinite(threshold_value) or not 0.0 <= threshold_value <= 1.0:
        raise CalibrationValidationError("threshold must be within [0,1]")
    converted = []
    for probability in probabilities:
        value = float(probability)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise CalibrationValidationError("calibrated probabilities must be within [0,1]")
        converted.append(1 if value >= threshold_value else 0)
    return tuple(converted)


def threshold_payload(result: SupportThresholdResult) -> Mapping[str, object]:
    return asdict(result)


def save_threshold_artifact(path, result: SupportThresholdResult):
    artifact = Path(path)
    artifact.parent.mkdir(parents=True, exist_ok=True)
    content = canonical_json_bytes(threshold_payload(result))
    artifact.write_bytes(content)
    artifact_hash = sha256_bytes(content)
    metadata_path = artifact.with_name(artifact.name + ".metadata.json")
    metadata_path.write_bytes(
        canonical_json_bytes(
            {
                "artifact_ref": str(artifact),
                "artifact_sha256": artifact_hash,
                "selected_model_sha256": result.selected_model_sha256,
                "calibrator_sha256": result.calibrator_sha256,
                "test_accessed": False,
            }
        )
    )
    return {"artifact_ref": str(artifact), "artifact_sha256": artifact_hash}


def load_threshold_artifact(path, *, expected_sha256: str) -> SupportThresholdResult:
    artifact = Path(path)
    if not artifact.is_file() or sha256_bytes(artifact.read_bytes()) != expected_sha256:
        raise CalibrationValidationError("threshold artifact SHA-256 mismatch")
    try:
        payload = json.loads(artifact.read_text(encoding="utf-8"))
        payload["policy"] = ThresholdSearchPolicy(**payload["policy"])
        result = SupportThresholdResult(**payload)
    except (ValueError, TypeError, KeyError) as error:
        raise CalibrationValidationError("invalid threshold artifact schema") from error
    if (
        result.artifact_version != "threshold_support_v1"
        or result.implementation_version != THRESHOLD_IMPLEMENTATION_VERSION
        or result.metric_implementation_version != METRIC_IMPLEMENTATION_VERSION
        or result.test_accessed is not False
        or not 0.0 <= result.threshold <= 1.0
    ):
        raise CalibrationValidationError("incompatible threshold implementation metadata")
    calibrator_artifact = Path(result.calibrator_ref)
    if (
        not calibrator_artifact.is_file()
        or sha256_bytes(calibrator_artifact.read_bytes()) != result.calibrator_sha256
    ):
        raise CalibrationValidationError("threshold references an incompatible calibrator artifact")
    validate_threshold_policy(result.policy, mode=result.threshold_mode)
    return result


def validate_threshold_binding(
    result: SupportThresholdResult,
    selected_model: SelectedSupportModel,
    *,
    calibrator_sha256: str,
    calibrator_ref: str = None,
) -> None:
    expected = {
        "selected_family": selected_model.family,
        "selected_run_id": selected_model.run_id,
        "selected_model_sha256": selected_model.artifact_sha256,
        "calibrator_sha256": calibrator_sha256,
        "validation_prediction_sha256": selected_model.validation_prediction_sha256,
        "validation_split_hash": selected_model.split_hash,
        "feature_version": selected_model.feature_version,
        "label_version": selected_model.label_version,
    }
    if calibrator_ref is not None and result.calibrator_ref != calibrator_ref:
        raise CalibrationValidationError("threshold compatibility mismatch: calibrator_ref")
    for field, value in expected.items():
        if getattr(result, field) != value:
            raise CalibrationValidationError("threshold compatibility mismatch: " + field)
