import pytest

from evidence.audit import EvidenceAuditError, validate_result_rows


def test_support_selection_cannot_be_rewritten_as_f1():
    bad = {
        "split": "validation",
        "task": "organ_support",
        "horizon": None,
        "N_examples": 20,
        "N_ICU_stays": 4,
        "selection_criterion": "validation_f1",
    }
    with pytest.raises(EvidenceAuditError, match="AUPRC"):
        validate_result_rows((bad,))


def test_valid_support_row_retains_counts_and_uncalibrated_selection():
    validate_result_rows(({
        "split": "validation",
        "task": "organ_support",
        "horizon": None,
        "N_examples": 20,
        "N_ICU_stays": 4,
        "N_positive_examples": 3,
        "N_positive_stays": 2,
        "selection_criterion": "validation_uncalibrated_auprc",
    },))
