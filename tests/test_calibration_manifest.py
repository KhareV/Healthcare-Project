import json

import pytest

from evaluation.calibrate import (
    promote_selection_manifest,
    save_calibrator_artifact,
    validate_calibrated_manifest,
)
from evaluation.calibration_validation import CalibrationValidationError
from evaluation.threshold import choose_support_threshold, save_threshold_artifact
from experiments.search_governance import canonical_sha256
from calibration_helpers import fitted, threshold_policy


def promoted_manifest(tmp_path):
    stage, selected, _, calibrator = fitted(tmp_path)
    calibrator_ref = save_calibrator_artifact(tmp_path / "calibrator.json", calibrator)
    threshold = choose_support_threshold(
        selected,
        calibrator,
        calibrator_ref=calibrator_ref["artifact_ref"],
        calibrator_sha256=calibrator_ref["artifact_sha256"],
        policy=threshold_policy(),
    )
    threshold_ref = save_threshold_artifact(tmp_path / "threshold.json", threshold)
    return stage, promote_selection_manifest(
        stage,
        selected,
        calibrator_artifact=calibrator_ref,
        threshold_artifact=threshold_ref,
    )


def rehash(manifest):
    content = dict(manifest)
    content.pop("manifest_sha256", None)
    content["manifest_sha256"] = canonical_sha256(content)
    return content


def test_promotion_preserves_all_selected_models_and_explanation_routes(tmp_path):
    stage, promoted = promoted_manifest(tmp_path)
    for task in ("recovery", "icu_stay_time", "organ_support"):
        for field in ("family", "run_id", "candidate_id", "artifact_sha256", "explanation_method"):
            assert promoted["tasks"][task][field] == stage["tasks"][task][field]
    assert promoted["tasks"]["organ_support"]["calibration_status"] == "FROZEN_VALIDATION_ONLY"
    assert promoted["serving_ready"] is False
    validate_calibrated_manifest(promoted)


def test_manifest_model_calibrator_or_threshold_mismatch_is_rejected(tmp_path):
    _, promoted = promoted_manifest(tmp_path)
    attacked = json.loads(json.dumps(promoted))
    attacked["tasks"]["organ_support"]["threshold"]["value"] += 0.01
    with pytest.raises(CalibrationValidationError, match="threshold value mismatch"):
        validate_calibrated_manifest(rehash(attacked))


def test_phase13_cannot_change_selected_support_family(tmp_path):
    stage, selected, _, calibrator = fitted(tmp_path)
    calibrator_ref = save_calibrator_artifact(tmp_path / "calibrator.json", calibrator)
    threshold = choose_support_threshold(
        selected,
        calibrator,
        calibrator_ref=calibrator_ref["artifact_ref"],
        calibrator_sha256=calibrator_ref["artifact_sha256"],
        policy=threshold_policy(),
    )
    threshold_ref = save_threshold_artifact(tmp_path / "threshold.json", threshold)
    selected = selected.__class__(**{**selected.__dict__, "family": "gru"})
    with pytest.raises(CalibrationValidationError):
        promote_selection_manifest(
            stage,
            selected,
            calibrator_artifact=calibrator_ref,
            threshold_artifact=threshold_ref,
        )
