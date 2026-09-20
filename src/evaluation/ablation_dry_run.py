"""Ephemeral synthetic Phase-14 ablation and sensitivity demonstration."""

import json
import tempfile
from pathlib import Path

from evaluation.ablations import (
    AblationVariant,
    FeatureGroupContract,
    derive_feature_manifest,
    evaluate_recovery_ablation,
    write_ablation_table,
)
from evaluation.metrics import PredictionRecord
from evaluation.selection_validation import sha256_file
from evaluation.sensitivity import (
    CompleteComponentRecord,
    evaluate_complete_component_sensitivity,
)
from experiments.search_governance import canonical_sha256


def run_synthetic_dry_run():
    with tempfile.TemporaryDirectory(prefix="phase14_synthetic_") as directory:
        root = Path(directory)
        features = (
            "value_hr",
            "observation_mask_hr",
            "tslo_hr",
            "static_age",
        )
        structural = ("padding_mask", "prediction_time")
        eligibility = ("recovery24_eligible", "recovery48_eligible")
        schema_hash = canonical_sha256(
            {
                "parent_schema_version": "synthetic_features_v1",
                "ordered_model_features": features,
                "structural_fields": structural,
                "eligibility_fields": eligibility,
            }
        )
        contract = FeatureGroupContract(
            contract_version="synthetic_ablation_groups_v1",
            parent_schema_version="synthetic_features_v1",
            parent_schema_hash=schema_hash,
            ordered_model_features=features,
            feature_groups={
                "observation_mask_features": ("observation_mask_hr",),
                "tslo_features": ("tslo_hr",),
            },
            structural_fields=structural,
            eligibility_fields=eligibility,
            mask_group_status="SYNTHETIC_TEST_DEFINITION_ONLY",
            matrix_status="SYNTHETIC_TEST_DEFINITION_ONLY",
            mode="synthetic",
        )
        parent = root / "parent.model"
        parent.write_bytes(b"synthetic parent recovery model")
        variant = AblationVariant(
            variant_id="synthetic_no_tslo",
            task="recovery",
            removed_feature_groups=("tslo_features",),
            parent_run_id="synthetic-parent-run",
            parent_artifact_ref=str(parent),
            parent_artifact_sha256=sha256_file(parent),
            parent_artifact_metadata_ref=None,
            model_family="gru",
            model_config_hash="synthetic-config",
            parent_feature_schema_version=contract.parent_schema_version,
            parent_feature_schema_hash=contract.parent_schema_hash,
            split_hash="synthetic-split",
            label_version="synthetic-labels",
            seed=123,
            code_commit="synthetic-only",
        )
        manifest = derive_feature_manifest(variant, contract)
        run_artifact = root / "ablation.model"
        run_artifact.write_bytes(b"synthetic no-tslo recovery model")
        records = (
            PredictionRecord("A", "2020-01-01", 2.0, 1.0, True),
            PredictionRecord("B", "2020-01-01", -2.0, -1.0, True),
        )
        evaluation = evaluate_recovery_ablation(
            variant=variant,
            manifest=manifest,
            run_id="synthetic-ablation-run",
            run_artifact_ref=str(run_artifact),
            run_artifact_sha256=sha256_file(run_artifact),
            recovery24=records,
            recovery48=records,
            source_partition="validation",
            test_accessed=False,
        )
        table, table_hash = write_ablation_table(root / "ablation_table.csv", (evaluation,))
        flags = (True,) * 6
        sensitivity_rows = (
            CompleteComponentRecord(
                stay_id="A",
                prediction_time="2020-01-01",
                partition="validation",
                target=2.0,
                prediction=1.0,
                eligible=True,
                baseline_component_observed=flags,
                future_component_observed=flags,
            ),
        )
        sensitivity = evaluate_complete_component_sensitivity(
            sensitivity_rows, sensitivity_rows
        )
        return {
            "run_type": "synthetic_dry_run_not_scientific",
            "variant": variant.variant_id,
            "derived_schema_hash": manifest.derived_schema_hash,
            "ablation_table_sha256": table_hash,
            "ablation_table_exists_during_run": table.is_file(),
            "complete_24_examples": sensitivity.recovery24.complete_N_examples,
            "complete_48_examples": sensitivity.recovery48.complete_N_examples,
            "test_accessed": False,
            "repository_artifacts_written": False,
        }


def main():
    print(json.dumps(run_synthetic_dry_run(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
