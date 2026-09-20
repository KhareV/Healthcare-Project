from pathlib import Path

from evaluation.ablations import (
    AblationVariant,
    FeatureGroupContract,
    RecoveryAblationExample,
)
from evaluation.selection_validation import sha256_file
from experiments.search_governance import canonical_sha256


ORDERED_FEATURES = (
    "value_heart_rate",
    "value_creatinine",
    "observation_mask_heart_rate",
    "observation_mask_creatinine",
    "tslo_heart_rate",
    "tslo_creatinine",
    "static_age",
)


def contract(**updates):
    structural = ("padding_mask", "prediction_time")
    eligibility = ("recovery24_eligible", "recovery48_eligible")
    schema_hash = canonical_sha256(
        {
            "parent_schema_version": "synthetic_feature_schema_v1",
            "ordered_model_features": ORDERED_FEATURES,
            "structural_fields": structural,
            "eligibility_fields": eligibility,
        }
    )
    values = {
        "contract_version": "synthetic_ablation_groups_v1",
        "parent_schema_version": "synthetic_feature_schema_v1",
        "parent_schema_hash": schema_hash,
        "ordered_model_features": ORDERED_FEATURES,
        "feature_groups": {
            "observation_mask_features": (
                "observation_mask_heart_rate",
                "observation_mask_creatinine",
            ),
            "tslo_features": ("tslo_heart_rate", "tslo_creatinine"),
        },
        "structural_fields": structural,
        "eligibility_fields": eligibility,
        "mask_group_status": "SYNTHETIC_TEST_DEFINITION_ONLY",
        "matrix_status": "SYNTHETIC_TEST_DEFINITION_ONLY",
        "mode": "synthetic",
    }
    values.update(updates)
    return FeatureGroupContract(**values)


def variant(tmp_path, removed=("tslo_features",), **updates):
    Path(tmp_path).mkdir(parents=True, exist_ok=True)
    artifact = Path(tmp_path) / "parent_recovery.model"
    artifact.write_bytes(b"synthetic frozen recovery parent")
    base = contract()
    values = {
        "variant_id": "synthetic_no_tslo",
        "task": "recovery",
        "removed_feature_groups": tuple(removed),
        "parent_run_id": "synthetic_recovery_parent",
        "parent_artifact_ref": str(artifact),
        "parent_artifact_sha256": sha256_file(artifact),
        "parent_artifact_metadata_ref": None,
        "model_family": "gru",
        "model_config_hash": "synthetic-config-hash",
        "parent_feature_schema_version": base.parent_schema_version,
        "parent_feature_schema_hash": base.parent_schema_hash,
        "split_hash": "synthetic-split-hash",
        "label_version": "synthetic-recovery-labels-v1",
        "seed": 123,
        "code_commit": "synthetic-only",
        "mode": "synthetic",
    }
    values.update(updates)
    return AblationVariant(**values)


def example():
    return RecoveryAblationExample(
        subject_id="subject-1",
        stay_id="stay-1",
        prediction_time="2020-01-02T00:00:00",
        split="validation",
        model_inputs={feature: index + 0.25 for index, feature in enumerate(ORDERED_FEATURES)},
        structural_padding=(True, True, False, False, False, False, False, False),
        imputation_provenance="synthetic_train_fit_preprocessor_v1",
        recovery24_target=-1.0,
        recovery48_target=2.0,
        recovery24_eligible=True,
        recovery48_eligible=True,
    )
