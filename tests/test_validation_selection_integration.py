import pytest

from experiments.search_governance import CandidateResult, GovernanceError, choose_best_within_family
from experiments.validation_suite import ValidationSuiteError, validate_serving_manifest_roles


def test_recovery_uses_mae24_not_better_mae48():
    results = (
        CandidateResult("a", "a", "COMPLETE", {"validation_recovery24_mae": 2.0, "validation_recovery48_mae": 0.1}),
        CandidateResult("b", "b", "COMPLETE", {"validation_recovery24_mae": 1.0, "validation_recovery48_mae": 9.0}),
    )
    assert choose_best_within_family(results, "recovery").candidate_id == "b"


def test_icu_uses_weighted_median_hours_and_support_requires_raw_auprc():
    icu = (
        CandidateResult("a", "a", "COMPLETE", {"validation_median_absolute_error_hours": 3.0}),
        CandidateResult("b", "b", "COMPLETE", {"validation_median_absolute_error_hours": 2.0}),
    )
    assert choose_best_within_family(icu, "icu_stay_time").candidate_id == "b"
    support = (
        CandidateResult("a", "a", "COMPLETE", {"validation_auprc": 0.7}, "calibrated"),
    )
    with pytest.raises(GovernanceError, match="raw uncalibrated"):
        choose_best_within_family(support, "organ_support")


@pytest.mark.parametrize("family", ("naive", "lstm"))
def test_naive_and_lstm_cannot_be_selected(family):
    manifest = {
        "tasks": {
            "recovery": {"family": family, "explanation_method": "none"},
            "icu_stay_time": {"family": "xgboost", "explanation_method": "tree_shap"},
            "organ_support": {"family": "gru", "explanation_method": "integrated_gradients"},
        }
    }
    with pytest.raises(ValidationSuiteError, match="XGBoost or GRU"):
        validate_serving_manifest_roles(manifest)
