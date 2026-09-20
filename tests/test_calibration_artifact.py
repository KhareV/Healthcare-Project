from dataclasses import replace

import pytest

from evaluation.calibrate import (
    load_calibrator_artifact,
    save_calibrator_artifact,
    validate_calibrator_binding,
)
from evaluation.calibration_validation import CalibrationValidationError
from evaluation.threshold import (
    choose_support_threshold,
    load_threshold_artifact,
    save_threshold_artifact,
    validate_threshold_binding,
)
from calibration_helpers import fitted, persist_calibrator, threshold_policy


def test_corrupt_calibrator_hash_is_rejected(tmp_path):
    _, _, _, calibrator = fitted(tmp_path)
    saved = save_calibrator_artifact(tmp_path / "calibrator.json", calibrator)
    with open(saved["artifact_ref"], "ab") as handle:
        handle.write(b"corrupt")
    with pytest.raises(CalibrationValidationError, match="SHA-256"):
        load_calibrator_artifact(
            saved["artifact_ref"], expected_sha256=saved["artifact_sha256"]
        )


def test_model_split_and_label_bindings_fail_closed(tmp_path):
    _, selected, _, calibrator = fitted(tmp_path)
    for field, value in (
        ("artifact_sha256", "f" * 64),
        ("split_hash", "different-split"),
        ("label_version", "different-labels"),
    ):
        with pytest.raises(CalibrationValidationError, match="compatibility mismatch"):
            validate_calibrator_binding(calibrator, replace(selected, **{field: value}))


def test_threshold_is_bound_to_exact_calibrator_hash(tmp_path):
    _, selected, _, calibrator = fitted(tmp_path)
    saved = persist_calibrator(tmp_path, calibrator)
    result = choose_support_threshold(
        selected,
        calibrator,
        calibrator_ref=saved["artifact_ref"],
        calibrator_sha256=saved["artifact_sha256"],
        policy=threshold_policy(),
    )
    with pytest.raises(CalibrationValidationError, match="calibrator_sha256"):
        validate_threshold_binding(result, selected, calibrator_sha256="d" * 64)


def test_corrupt_threshold_hash_is_rejected(tmp_path):
    _, selected, _, calibrator = fitted(tmp_path)
    calibrator_saved = persist_calibrator(tmp_path, calibrator)
    result = choose_support_threshold(
        selected,
        calibrator,
        calibrator_ref=calibrator_saved["artifact_ref"],
        calibrator_sha256=calibrator_saved["artifact_sha256"],
        policy=threshold_policy(),
    )
    saved = save_threshold_artifact(tmp_path / "threshold.json", result)
    with open(saved["artifact_ref"], "ab") as handle:
        handle.write(b"corrupt")
    with pytest.raises(CalibrationValidationError, match="SHA-256"):
        load_threshold_artifact(
            saved["artifact_ref"], expected_sha256=saved["artifact_sha256"]
        )
