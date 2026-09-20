from dataclasses import replace

import pytest

from evaluation.ablations import (
    AblationValidationError,
    apply_feature_ablation,
    derive_feature_manifest,
    validate_ablation_input_parity,
)
from ablation_helpers import contract, example, variant


def test_tslo_ablation_preserves_masks_imputation_targets_and_split(tmp_path):
    reference = example()
    manifest = derive_feature_manifest(variant(tmp_path), contract())
    ablated = apply_feature_ablation(reference, manifest)
    validate_ablation_input_parity(reference, ablated, manifest)
    assert "observation_mask_heart_rate" in ablated.model_inputs
    assert ablated.imputation_provenance == reference.imputation_provenance
    assert ablated.structural_padding == reference.structural_padding
    assert ablated.recovery24_target == reference.recovery24_target
    assert ablated.recovery48_target == reference.recovery48_target


def test_mask_ablation_removes_only_declared_model_information(tmp_path):
    reference = example()
    manifest = derive_feature_manifest(
        variant(tmp_path, removed=("observation_mask_features",)), contract()
    )
    ablated = apply_feature_ablation(reference, manifest)
    validate_ablation_input_parity(reference, ablated, manifest)
    assert "tslo_heart_rate" in ablated.model_inputs
    assert ablated.recovery24_eligible and ablated.recovery48_eligible


def test_target_split_or_nonablated_feature_mutation_is_detected(tmp_path):
    reference = example()
    manifest = derive_feature_manifest(variant(tmp_path), contract())
    ablated = apply_feature_ablation(reference, manifest)
    with pytest.raises(AblationValidationError, match="targets/split"):
        validate_ablation_input_parity(
            reference, replace(ablated, recovery24_target=999.0), manifest
        )
    attacked_inputs = dict(ablated.model_inputs)
    attacked_inputs["value_heart_rate"] = -999
    with pytest.raises(AblationValidationError, match="non-ablated"):
        validate_ablation_input_parity(
            reference, replace(ablated, model_inputs=attacked_inputs), manifest
        )

