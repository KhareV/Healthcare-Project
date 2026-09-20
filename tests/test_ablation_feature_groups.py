from dataclasses import replace

import pytest

from evaluation.ablations import (
    AblationValidationError,
    derive_feature_manifest,
    validate_feature_group_contract,
)
from ablation_helpers import ORDERED_FEATURES, contract, variant


def test_tslo_group_is_structurally_removed_in_parent_order(tmp_path):
    manifest = derive_feature_manifest(variant(tmp_path), contract())
    assert manifest.removed_features == ("tslo_heart_rate", "tslo_creatinine")
    assert manifest.retained_features == tuple(
        feature for feature in ORDERED_FEATURES if not feature.startswith("tslo_")
    )


def test_full_reference_manifest_retains_every_parent_feature(tmp_path):
    reference = variant(
        tmp_path,
        removed=(),
        variant_id="synthetic_full_reference",
        is_reference=True,
    )
    manifest = derive_feature_manifest(reference, contract())
    assert manifest.removed_features == ()
    assert manifest.retained_features == ORDERED_FEATURES


def test_observation_mask_group_is_distinct_from_padding_and_eligibility(tmp_path):
    manifest = derive_feature_manifest(
        variant(tmp_path, removed=("observation_mask_features",)), contract()
    )
    assert manifest.removed_features == (
        "observation_mask_heart_rate",
        "observation_mask_creatinine",
    )
    assert "padding_mask" not in manifest.removed_features
    assert not any("eligible" in feature for feature in manifest.removed_features)


def test_unknown_duplicate_and_parent_hash_fail_closed(tmp_path):
    with pytest.raises(AblationValidationError, match="unknown"):
        derive_feature_manifest(variant(tmp_path, removed=("unknown",)), contract())
    with pytest.raises(AblationValidationError, match="unique"):
        derive_feature_manifest(
            variant(tmp_path, removed=("tslo_features", "tslo_features")), contract()
        )
    bad = replace(contract(), parent_schema_hash="wrong")
    with pytest.raises(AblationValidationError, match="schema hash"):
        validate_feature_group_contract(bad)


def test_reserved_task_eligibility_group_and_remove_all_are_rejected(tmp_path):
    bad_contract = replace(
        contract(),
        feature_groups={"task_eligibility": ("value_heart_rate",)},
    )
    with pytest.raises(AblationValidationError, match="reserved"):
        validate_feature_group_contract(bad_contract)
    all_group = replace(contract(), feature_groups={"everything": ORDERED_FEATURES})
    with pytest.raises(AblationValidationError, match="remove all"):
        derive_feature_manifest(variant(tmp_path, removed=("everything",)), all_group)
