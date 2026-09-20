import json
from dataclasses import replace
from pathlib import Path

from evaluation.calibrate import (
    fit_support_calibrator,
    save_calibrator_artifact,
    selected_support_from_manifest,
)
from evaluation.calibration_validation import (
    IsotonicPolicy,
    PREDICTION_ARTIFACT_VERSION,
    canonical_json_bytes,
    sha256_bytes,
)
from evaluation.select import build_selection_stage_manifest, select_task
from evaluation.threshold import ThresholdSearchPolicy
from experiments.search_governance import canonical_sha256
from selection_helpers import pair


DEFAULT_ROWS = [
    {"row_id": "r1", "stay_id": "s1", "organ_support_eligible": True, "raw_probability": 0.1, "label": 0},
    {"row_id": "r2", "stay_id": "s1", "organ_support_eligible": True, "raw_probability": 0.4, "label": 1},
    {"row_id": "r3", "stay_id": "s2", "organ_support_eligible": True, "raw_probability": 0.4, "label": 0},
    {"row_id": "r4", "stay_id": "s2", "organ_support_eligible": True, "raw_probability": 0.8, "label": 1},
    {"row_id": "r5", "stay_id": "s3", "organ_support_eligible": True, "raw_probability": 0.2, "label": 0},
    {"row_id": "r6", "stay_id": "s3", "organ_support_eligible": True, "raw_probability": 0.9, "label": 1},
    {"row_id": "censored", "stay_id": "s4", "organ_support_eligible": False, "raw_probability": "ignored", "label": "ignored"},
]


def isotonic_policy(**updates):
    values = {
        "policy_version": "synthetic_isotonic_policy_v1",
        "out_of_bounds": "clip",
        "sample_weighting": "stay_balanced",
        "approved_for_real": False,
    }
    values.update(updates)
    return IsotonicPolicy(**values)


def threshold_policy(**updates):
    values = {
        "policy_version": "synthetic_threshold_policy_v1",
        "candidate_method": "unique_calibrated_probabilities",
        "tie_policy": "lowest_threshold",
        "approved_for_real": False,
    }
    values.update(updates)
    return ThresholdSearchPolicy(**values)


def stage_and_selected(tmp_path, rows=None):
    Path(tmp_path).mkdir(parents=True, exist_ok=True)
    rx, rg = pair(tmp_path, "recovery", 1.0, 2.0)
    ix, ig = pair(tmp_path, "icu_stay_time", 12.0, 10.0)
    sx, sg = pair(tmp_path, "organ_support", 0.4, 0.3)
    results = {
        "recovery": select_task(task="recovery", xgboost=rx, gru=rg),
        "icu_stay_time": select_task(task="icu_stay_time", xgboost=ix, gru=ig),
        "organ_support": select_task(task="organ_support", xgboost=sx, gru=sg),
    }
    stage = build_selection_stage_manifest(results)
    support = stage["tasks"]["organ_support"]
    prediction_path = Path(tmp_path) / "selected_support_validation_predictions.json"
    payload = {
        "artifact_version": PREDICTION_ARTIFACT_VERSION,
        "task": "organ_support",
        "family": support["family"],
        "run_id": support["run_id"],
        "candidate_id": support["candidate_id"],
        "config_hash": support["config_hash"],
        "model_sha256": support["artifact_sha256"],
        "split_hash": support["split_hash"],
        "feature_version": support["feature_version"],
        "label_version": support["label_version"],
        "probability_type": "raw_uncalibrated",
        "partition": "validation",
        "test_accessed": False,
        "rows": list(DEFAULT_ROWS if rows is None else rows),
    }
    prediction_path.write_bytes(canonical_json_bytes(payload))
    prediction_hash = sha256_bytes(prediction_path.read_bytes())
    stage["tasks"]["organ_support"]["validation_prediction_ref"] = str(prediction_path)
    stage_without_hash = dict(stage)
    stage_without_hash.pop("manifest_sha256")
    stage["manifest_sha256"] = canonical_sha256(stage_without_hash)
    selected = selected_support_from_manifest(
        stage, validation_prediction_sha256=prediction_hash
    )
    return stage, selected, prediction_path


def fitted(tmp_path, rows=None, policy=None):
    stage, selected, prediction_path = stage_and_selected(tmp_path, rows=rows)
    calibrator = fit_support_calibrator(
        selected, policy=policy or isotonic_policy(), mode="synthetic"
    )
    return stage, selected, prediction_path, calibrator


def identity_calibrator(calibrator):
    return replace(calibrator, x_thresholds=(0.0, 1.0), y_thresholds=(0.0, 1.0))


def persist_calibrator(tmp_path, calibrator, name="calibrator.json"):
    return save_calibrator_artifact(Path(tmp_path) / name, calibrator)
