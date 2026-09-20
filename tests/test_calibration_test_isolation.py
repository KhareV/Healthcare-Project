import json
from dataclasses import replace

import pytest

from evaluation.calibrate import fit_support_calibrator
from evaluation.calibration_validation import (
    CalibrationValidationError,
    canonical_json_bytes,
    sha256_bytes,
)
from calibration_helpers import isotonic_policy, stage_and_selected


def test_test_artifact_sentinel_is_rejected_without_reading_content(tmp_path):
    _, selected, prediction = stage_and_selected(tmp_path)
    payload = json.loads(prediction.read_text(encoding="utf-8"))
    sentinel = tmp_path / "fake_test.json"
    sentinel.write_text("DO_NOT_READ_TEST_SENTINEL", encoding="utf-8")
    payload["test_artifact_ref"] = str(sentinel)
    prediction.write_bytes(canonical_json_bytes(payload))
    selected = replace(selected, validation_prediction_sha256=sha256_bytes(prediction.read_bytes()))
    with pytest.raises(CalibrationValidationError, match="TEST-LABELLED FIELD") as caught:
        fit_support_calibrator(selected, policy=isotonic_policy())
    assert "DO_NOT_READ_TEST_SENTINEL" not in str(caught.value)


def test_real_execution_rejects_synthetic_selected_model_and_unapproved_policy(tmp_path):
    _, selected, _ = stage_and_selected(tmp_path)
    with pytest.raises(CalibrationValidationError, match="REAL SELECTED"):
        fit_support_calibrator(selected, policy=isotonic_policy(), mode="real")

