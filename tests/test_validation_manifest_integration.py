import pytest

from experiments.validation_suite import (
    ValidationSuiteError,
    assert_validation_mutation_allowed,
    validate_serving_manifest_roles,
)


def test_mixed_serving_families_and_routing_are_valid():
    validate_serving_manifest_roles(
        {
            "tasks": {
                "recovery": {"family": "gru", "explanation_method": "integrated_gradients"},
                "icu_stay_time": {"family": "xgboost", "explanation_method": "tree_shap"},
                "organ_support": {"family": "gru", "explanation_method": "integrated_gradients"},
            }
        }
    )


def test_post_selection_freeze_rejects_extra_search_or_family_change():
    with pytest.raises(ValidationSuiteError, match="scientific reset"):
        assert_validation_mutation_allowed({"status": "VALIDATION_FROZEN"})
