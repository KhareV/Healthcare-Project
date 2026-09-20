import pytest

from evidence.audit import EvidenceAuditError, validate_bootstrap_evidence, validate_result_rows


def contract():
    return {
        "resampling_unit": "stay_id",
        "rows_sampled_independently": False,
        "ci_quantiles": [0.025, 0.975],
        "uncertainty_scope": "aggregate_performance",
    }


def test_bootstrap_evidence_rejects_row_resampling():
    validate_bootstrap_evidence(contract())
    changed = contract()
    changed["rows_sampled_independently"] = True
    with pytest.raises(EvidenceAuditError, match="bootstrap"):
        validate_bootstrap_evidence(changed)


def test_patient_interval_ci_label_is_rejected():
    with pytest.raises(EvidenceAuditError, match="mislabeled"):
        validate_result_rows(({
            "split": "test", "task": "icu_stay_time", "horizon": None,
            "N_examples": 10, "N_ICU_stays": 2,
            "ci_label": "patient prediction interval",
        },))
