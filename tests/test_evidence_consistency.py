import pytest

from evidence.audit import EvidenceAuditError, validate_result_rows


def row(**updates):
    value = {
        "split": "validation",
        "task": "recovery",
        "horizon": "24h",
        "N_examples": 10,
        "N_ICU_stays": 3,
        "selection_criterion": "validation_mae24",
    }
    value.update(updates)
    return value


def test_validation_test_and_recovery_horizon_mixups_fail():
    with pytest.raises(EvidenceAuditError, match="ambiguous split"):
        validate_result_rows((row(split="development"),))
    with pytest.raises(EvidenceAuditError, match="horizons separate"):
        validate_result_rows((row(horizon="average_24h_48h"),))


def test_missing_denominators_fail():
    with pytest.raises(EvidenceAuditError, match="denominator"):
        validate_result_rows((row(N_ICU_stays=None),))
