from evaluation.calibrate import (
    apply_support_calibrator,
    fit_support_calibrator,
    load_calibrator_artifact,
    save_calibrator_artifact,
)
from calibration_helpers import fitted, isotonic_policy, stage_and_selected


def test_isotonic_outputs_are_probabilities_deterministic_and_ordered(tmp_path):
    _, selected, prediction_path, first = fitted(tmp_path)
    raw_artifact_before = prediction_path.read_bytes()
    second = fit_support_calibrator(selected, policy=isotonic_policy())
    raw = (0.9, 0.1, 0.4, 0.8)
    first_values = apply_support_calibrator(first, raw)
    assert first == second
    assert len(first_values) == len(raw)
    assert all(0.0 <= value <= 1.0 for value in first_values)
    assert first_values == apply_support_calibrator(second, raw)
    assert prediction_path.read_bytes() == raw_artifact_before


def test_calibrator_serialization_roundtrip(tmp_path):
    _, _, _, calibrator = fitted(tmp_path)
    saved = save_calibrator_artifact(tmp_path / "isotonic_support_v1.json", calibrator)
    loaded = load_calibrator_artifact(
        saved["artifact_ref"], expected_sha256=saved["artifact_sha256"]
    )
    raw = (0.05, 0.4, 0.95)
    assert apply_support_calibrator(loaded, raw) == apply_support_calibrator(calibrator, raw)
