import json
from dataclasses import replace

import pytest

from evaluation.calibrate import fit_support_calibrator
from evaluation.calibration_validation import CalibrationValidationError, canonical_json_bytes, sha256_bytes
from calibration_helpers import DEFAULT_ROWS, isotonic_policy, stage_and_selected


def test_censored_rows_cannot_change_calibrator_or_population(tmp_path):
    rows_a = list(DEFAULT_ROWS)
    rows_b = [dict(row) for row in DEFAULT_ROWS]
    rows_b[-1]["raw_probability"] = 999999
    rows_b[-1]["label"] = -999999
    _, selected_a, _, = stage_and_selected(tmp_path / "a", rows=rows_a)
    _, selected_b, _, = stage_and_selected(tmp_path / "b", rows=rows_b)
    first = fit_support_calibrator(selected_a, policy=isotonic_policy())
    second = fit_support_calibrator(selected_b, policy=isotonic_policy())
    assert first.x_thresholds == second.x_thresholds
    assert first.y_thresholds == second.y_thresholds
    assert first.eligible_population_hash == second.eligible_population_hash


def test_train_or_test_partition_artifact_is_rejected(tmp_path):
    _, selected, prediction = stage_and_selected(tmp_path)
    payload = json.loads(prediction.read_text(encoding="utf-8"))
    payload["partition"] = "train"
    prediction.write_bytes(canonical_json_bytes(payload))
    selected = replace(selected, validation_prediction_sha256=sha256_bytes(prediction.read_bytes()))
    with pytest.raises(CalibrationValidationError, match="partition"):
        fit_support_calibrator(selected, policy=isotonic_policy())


def test_non_selected_model_hash_cannot_be_calibrated(tmp_path):
    _, selected, _, = stage_and_selected(tmp_path)
    attacked = replace(selected, artifact_sha256="0" * 64)
    with pytest.raises(CalibrationValidationError, match="model artifact/hash"):
        fit_support_calibrator(attacked, policy=isotonic_policy())


def test_recovery_or_icu_outputs_cannot_enter_calibration(tmp_path):
    _, selected, _ = stage_and_selected(tmp_path)
    with pytest.raises(CalibrationValidationError, match="organ support only"):
        fit_support_calibrator(
            replace(selected, task="recovery"), policy=isotonic_policy()
        )
