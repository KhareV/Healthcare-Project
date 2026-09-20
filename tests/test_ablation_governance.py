from dataclasses import replace

import pytest

from evaluation.ablations import (
    AblationValidationError,
    MATRIX_BLOCKER,
    REAL_RECOVERY_MODEL_BLOCKER,
    validate_ablation_variant,
)
from ablation_helpers import contract, variant


def test_ablation_has_parent_binding_and_zero_search_budget(tmp_path):
    candidate = variant(tmp_path)
    validate_ablation_variant(candidate, contract())
    assert candidate.parent_run_id
    assert candidate.parent_artifact_sha256
    assert candidate.hyperparameter_search_budget == 0


def test_retuning_and_nonrecovery_tasks_are_rejected(tmp_path):
    with pytest.raises(AblationValidationError, match="retuning"):
        validate_ablation_variant(
            variant(tmp_path, hyperparameter_search_budget=1), contract()
        )
    with pytest.raises(AblationValidationError, match="recovery ablations only"):
        validate_ablation_variant(variant(tmp_path, task="organ_support"), contract())


def test_real_execution_requires_real_parent_matrix_mask_and_retraining_freezes(tmp_path):
    real_variant = variant(tmp_path, mode="real")
    with pytest.raises(AblationValidationError, match="FROZEN REAL RECOVERY"):
        validate_ablation_variant(real_variant, contract())
    real_contract = replace(contract(), mode="real")
    with pytest.raises(AblationValidationError, match="MASK FEATURE GROUP"):
        validate_ablation_variant(real_variant, real_contract)


def test_ablation_cannot_expose_serving_promotion_or_replace_parent(tmp_path):
    candidate = variant(tmp_path)
    assert not hasattr(candidate, "selected_models_manifest")
    assert candidate.run_type == "ablation"

