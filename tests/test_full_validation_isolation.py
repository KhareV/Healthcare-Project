from pathlib import Path

import pytest

from experiments.validation_suite import (
    ValidationSuiteError,
    preflight_validation_suite,
    run_validation_suite,
)


ROOT = Path(__file__).resolve().parents[1]


def test_real_preflight_is_blocked_before_any_scientific_command():
    report = preflight_validation_suite(ROOT)
    assert report.overall_status == "BLOCKED"
    assert report.scientific_commands_executed == 0
    assert report.test_accessed is False


def test_hostile_test_or_training_executor_is_never_called():
    calls = []

    def hostile():
        calls.append("SCIENTIFIC_OR_TEST_DATA_OPENED")
        raise AssertionError("must not run")

    with pytest.raises(ValidationSuiteError, match="REQUIRED VALIDATION DEPENDENCY"):
        run_validation_suite(ROOT, hostile)
    assert calls == []
