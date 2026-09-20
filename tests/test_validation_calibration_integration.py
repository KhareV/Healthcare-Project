import pytest

from experiments.validation_suite import (
    ValidationSuiteError,
    require_stage_order,
    validate_postprocessing_bindings,
)


def test_calibration_and_threshold_require_stage_order():
    state = {"stages": {"preprocessing": "COMPLETE"}}
    with pytest.raises(ValidationSuiteError, match="stage order"):
        require_stage_order(state, "support_calibration")


def test_calibrator_and_threshold_hash_bindings():
    selected = {"artifact_sha256": "a" * 64}
    calibrator = {"selected_model_sha256": "a" * 64, "artifact_sha256": "b" * 64, "partition": "validation"}
    threshold = {"calibrator_sha256": "b" * 64, "criterion": "validation_f1", "partition": "validation"}
    validate_postprocessing_bindings(selected, calibrator, threshold)
    with pytest.raises(ValidationSuiteError, match="calibrator hash binding"):
        validate_postprocessing_bindings(selected, dict(calibrator, selected_model_sha256="c" * 64), threshold)
    with pytest.raises(ValidationSuiteError, match="threshold hash binding"):
        validate_postprocessing_bindings(selected, calibrator, dict(threshold, calibrator_sha256="c" * 64))
