"""Recovery-only feature-ablation contracts and synthetic-safe utilities."""

import csv
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping, Optional, Sequence, Tuple

from evaluation.metrics import (
    METRIC_IMPLEMENTATION_VERSION,
    PredictionRecord,
    evaluate_recovery_horizon,
)
from evaluation.selection_validation import reject_test_fields, sha256_file
from experiments.search_governance import canonical_sha256


ABLATION_IMPLEMENTATION_VERSION = "recovery_feature_ablation_v1"
REAL_RECOVERY_MODEL_BLOCKER = (
    "BLOCKED — FROZEN REAL RECOVERY MODEL REQUIRED FOR SCIENTIFIC ABLATION RESULTS"
)
MATRIX_BLOCKER = "BLOCKED — EXACT MASK/TSLO ABLATION MATRIX REQUIRED"
MASK_GROUP_BLOCKER = "BLOCKED — MASK FEATURE GROUP DEFINITION REQUIRED"
RETRAINING_UNLOCKED = (
    "UNLOCKED SCIENTIFIC/ENGINEERING PARAMETER — ABLATION RETRAINING POLICY"
)
ABLATION_SPECIFICATION_BLOCKER = "BLOCKED — ABLATION SPECIFICATION REQUIRED"
FINAL_TEST_DEFERRED = (
    "REAL FINAL-TEST SOFA COMPLETENESS SENSITIVITY — DEFERRED UNTIL AUTHORIZED TEST PHASE"
)
RESERVED_NON_ABLATABLE_GROUPS = (
    "task_eligibility",
    "target_fields",
    "padding_semantics",
    "identifiers",
)


class AblationValidationError(ValueError):
    """Raised when an ablation would change more than declared model information."""


@dataclass(frozen=True)
class FeatureGroupContract:
    contract_version: str
    parent_schema_version: str
    parent_schema_hash: str
    ordered_model_features: Tuple[str, ...]
    feature_groups: Mapping[str, Tuple[str, ...]]
    structural_fields: Tuple[str, ...]
    eligibility_fields: Tuple[str, ...]
    mask_group_status: str
    matrix_status: str
    mode: str


@dataclass(frozen=True)
class AblationVariant:
    variant_id: str
    task: str
    removed_feature_groups: Tuple[str, ...]
    parent_run_id: str
    parent_artifact_ref: str
    parent_artifact_sha256: str
    parent_artifact_metadata_ref: Optional[str]
    model_family: str
    model_config_hash: str
    parent_feature_schema_version: str
    parent_feature_schema_hash: str
    split_hash: str
    label_version: str
    seed: int
    code_commit: str
    run_type: str = "ablation"
    hyperparameter_search_budget: int = 0
    retraining_policy_version: Optional[str] = None
    matrix_version: Optional[str] = None
    mode: str = "synthetic"
    test_accessed: bool = False
    is_reference: bool = False


@dataclass(frozen=True)
class DerivedFeatureManifest:
    manifest_version: str
    feature_group_contract_version: str
    parent_schema_version: str
    parent_schema_hash: str
    ablation_variant_id: str
    removed_feature_groups: Tuple[str, ...]
    removed_features: Tuple[str, ...]
    retained_features: Tuple[str, ...]
    retained_feature_count: int
    derived_schema_hash: str
    code_commit: str
    mode: str


@dataclass(frozen=True)
class RecoveryAblationExample:
    subject_id: str
    stay_id: str
    prediction_time: str
    split: str
    model_inputs: Mapping[str, object]
    structural_padding: Tuple[bool, ...]
    imputation_provenance: str
    recovery24_target: Optional[float]
    recovery48_target: Optional[float]
    recovery24_eligible: bool
    recovery48_eligible: bool


@dataclass(frozen=True)
class AblationEvaluation:
    variant_id: str
    run_id: str
    model_family: str
    parent_run_id: str
    parent_artifact_sha256: str
    model_config_hash: str
    parent_feature_schema_version: str
    parent_feature_schema_hash: str
    derived_schema_hash: str
    removed_feature_groups: Tuple[str, ...]
    split_hash: str
    label_version: str
    seed: int
    run_artifact_sha256: str
    recovery24: object
    recovery48: object
    status: str
    metric_implementation_version: str
    test_accessed: bool = False


def validate_feature_group_contract(contract: FeatureGroupContract) -> None:
    if contract.mode not in ("synthetic", "real"):
        raise AblationValidationError("feature-group contract mode is invalid")
    if not contract.ordered_model_features or len(set(contract.ordered_model_features)) != len(
        contract.ordered_model_features
    ):
        raise AblationValidationError("parent model features must be unique and nonempty")
    assigned = []
    for group, features in contract.feature_groups.items():
        if group in RESERVED_NON_ABLATABLE_GROUPS:
            raise AblationValidationError("reserved scientific fields cannot be feature groups")
        if not features or len(set(features)) != len(features):
            raise AblationValidationError("feature groups must be explicit and duplicate-free")
        if any(feature not in contract.ordered_model_features for feature in features):
            raise AblationValidationError("feature group contains an unknown model feature")
        assigned.extend(features)
    if len(set(assigned)) != len(assigned):
        raise AblationValidationError("one model feature cannot belong to multiple ablation groups")
    if any(field in contract.ordered_model_features for field in contract.eligibility_fields):
        raise AblationValidationError("task eligibility cannot be exposed as ablatable input")
    expected_hash = canonical_sha256(
        {
            "parent_schema_version": contract.parent_schema_version,
            "ordered_model_features": contract.ordered_model_features,
            "structural_fields": contract.structural_fields,
            "eligibility_fields": contract.eligibility_fields,
        }
    )
    if expected_hash != contract.parent_schema_hash:
        raise AblationValidationError("parent feature schema hash mismatch")


def reject_ablation_test_fields(payload) -> None:
    try:
        reject_test_fields(payload, path="ablation_input")
    except ValueError as error:
        raise AblationValidationError(str(error)) from error


def validate_ablation_variant(
    variant: AblationVariant, contract: FeatureGroupContract
) -> None:
    validate_feature_group_contract(contract)
    if variant.task != "recovery" or variant.run_type != "ablation":
        raise AblationValidationError("Phase 14 supports recovery ablations only")
    if variant.model_family not in ("xgboost", "gru"):
        raise AblationValidationError("parent recovery family must be XGBoost or GRU")
    if variant.test_accessed is not False:
        raise AblationValidationError("HARD FAIL — PHASE 14 TEST ACCESS FORBIDDEN")
    if variant.hyperparameter_search_budget != 0:
        raise AblationValidationError("ablation hyperparameter search/retuning is forbidden")
    if variant.is_reference:
        if variant.removed_feature_groups:
            raise AblationValidationError("reference variant cannot remove feature groups")
    elif not variant.removed_feature_groups:
        raise AblationValidationError("ablation must name a removed feature group")
    if len(set(variant.removed_feature_groups)) != len(variant.removed_feature_groups):
        raise AblationValidationError("removed feature groups must be explicit and unique")
    unknown = set(variant.removed_feature_groups) - set(contract.feature_groups)
    if unknown:
        raise AblationValidationError("unknown ablation feature group: " + sorted(unknown)[0])
    if variant.parent_feature_schema_version != contract.parent_schema_version:
        raise AblationValidationError("parent feature schema version mismatch")
    if variant.parent_feature_schema_hash != contract.parent_schema_hash:
        raise AblationValidationError("parent feature schema hash mismatch")
    parent = Path(variant.parent_artifact_ref)
    if not parent.is_file() or sha256_file(parent) != variant.parent_artifact_sha256:
        raise AblationValidationError("parent recovery artifact/hash mismatch")
    if variant.mode == "real":
        if contract.mode != "real":
            raise AblationValidationError(REAL_RECOVERY_MODEL_BLOCKER)
        if contract.mask_group_status != "FROZEN":
            raise AblationValidationError(MASK_GROUP_BLOCKER)
        if contract.matrix_status != "FROZEN" or not variant.matrix_version:
            raise AblationValidationError(MATRIX_BLOCKER)
        if not variant.retraining_policy_version:
            raise AblationValidationError(RETRAINING_UNLOCKED)
        metadata_path = Path(str(variant.parent_artifact_metadata_ref or ""))
        if not metadata_path.is_file():
            raise AblationValidationError("real parent recovery metadata is required")
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise AblationValidationError("real parent recovery metadata is invalid") from error
        expected = {
            "task": "recovery",
            "model_family": variant.model_family,
            "run_id": variant.parent_run_id,
            "config_hash": variant.model_config_hash,
            "split_hash": variant.split_hash,
            "feature_schema_version": variant.parent_feature_schema_version,
            "label_version": variant.label_version,
        }
        for field, value in expected.items():
            if metadata.get(field) != value:
                raise AblationValidationError("real parent metadata mismatch: " + field)
    elif variant.mode != "synthetic":
        raise AblationValidationError("ablation mode must be synthetic or real")


def derive_feature_manifest(
    variant: AblationVariant, contract: FeatureGroupContract
) -> DerivedFeatureManifest:
    validate_ablation_variant(variant, contract)
    removed_set = {
        feature
        for group in variant.removed_feature_groups
        for feature in contract.feature_groups[group]
    }
    retained = tuple(
        feature for feature in contract.ordered_model_features if feature not in removed_set
    )
    if not retained:
        raise AblationValidationError("ablation cannot remove all model features")
    removed = tuple(
        feature for feature in contract.ordered_model_features if feature in removed_set
    )
    content = {
        "manifest_version": "derived_ablation_feature_manifest_v1",
        "feature_group_contract_version": contract.contract_version,
        "parent_schema_version": contract.parent_schema_version,
        "parent_schema_hash": contract.parent_schema_hash,
        "ablation_variant_id": variant.variant_id,
        "removed_feature_groups": variant.removed_feature_groups,
        "removed_features": removed,
        "retained_features": retained,
        "mode": variant.mode,
        "code_commit": variant.code_commit,
    }
    return DerivedFeatureManifest(
        **content,
        retained_feature_count=len(retained),
        derived_schema_hash=canonical_sha256(content),
    )


def apply_feature_ablation(
    example: RecoveryAblationExample, manifest: DerivedFeatureManifest
) -> RecoveryAblationExample:
    if set(example.model_inputs) != set(manifest.retained_features) | set(
        manifest.removed_features
    ):
        raise AblationValidationError("example inputs do not match parent feature manifest")
    retained = {
        feature: example.model_inputs[feature] for feature in manifest.retained_features
    }
    return replace(example, model_inputs=retained)


def validate_ablation_input_parity(
    reference: RecoveryAblationExample,
    ablated: RecoveryAblationExample,
    manifest: DerivedFeatureManifest,
) -> None:
    immutable = (
        "subject_id",
        "stay_id",
        "prediction_time",
        "split",
        "structural_padding",
        "imputation_provenance",
        "recovery24_target",
        "recovery48_target",
        "recovery24_eligible",
        "recovery48_eligible",
    )
    if any(getattr(reference, field) != getattr(ablated, field) for field in immutable):
        raise AblationValidationError("ablation changed identifiers/targets/split/eligibility")
    if tuple(ablated.model_inputs) != manifest.retained_features:
        raise AblationValidationError("ablated feature ordering differs from derived manifest")
    for feature in manifest.retained_features:
        if reference.model_inputs[feature] != ablated.model_inputs[feature]:
            raise AblationValidationError("ablation changed a non-ablated feature")


def evaluate_recovery_ablation(
    *,
    variant: AblationVariant,
    manifest: DerivedFeatureManifest,
    run_id: str,
    run_artifact_ref: str,
    run_artifact_sha256: str,
    recovery24: Sequence[PredictionRecord],
    recovery48: Sequence[PredictionRecord],
    source_partition: str,
    test_accessed: bool,
) -> AblationEvaluation:
    if source_partition != "validation" or test_accessed is not False:
        raise AblationValidationError("HARD FAIL — PHASE 14 TEST ACCESS FORBIDDEN")
    artifact = Path(run_artifact_ref)
    if not artifact.is_file() or sha256_file(artifact) != run_artifact_sha256:
        raise AblationValidationError("ablation run artifact/hash mismatch")
    result24 = evaluate_recovery_horizon(recovery24, horizon="24h")
    result48 = evaluate_recovery_horizon(recovery48, horizon="48h")
    if any(
        result.metadata["metric_implementation_version"] != METRIC_IMPLEMENTATION_VERSION
        for result in (result24, result48)
    ):
        raise AblationValidationError("ablation did not reuse Phase-9 metrics")
    return AblationEvaluation(
        variant_id=variant.variant_id,
        run_id=run_id,
        model_family=variant.model_family,
        parent_run_id=variant.parent_run_id,
        parent_artifact_sha256=variant.parent_artifact_sha256,
        model_config_hash=variant.model_config_hash,
        parent_feature_schema_version=variant.parent_feature_schema_version,
        parent_feature_schema_hash=variant.parent_feature_schema_hash,
        derived_schema_hash=manifest.derived_schema_hash,
        removed_feature_groups=variant.removed_feature_groups,
        split_hash=variant.split_hash,
        label_version=variant.label_version,
        seed=variant.seed,
        run_artifact_sha256=run_artifact_sha256,
        recovery24=result24,
        recovery48=result48,
        status="SYNTHETIC_EVIDENCE_ONLY" if variant.mode == "synthetic" else "COMPLETE",
        metric_implementation_version=METRIC_IMPLEMENTATION_VERSION,
    )


ABLATION_TABLE_FIELDS = (
    "variant_id",
    "run_id",
    "reference_or_ablation",
    "model_family",
    "parent_run_id",
    "parent_artifact_sha256",
    "model_config_hash",
    "parent_feature_schema_version",
    "parent_feature_schema_hash",
    "derived_schema_hash",
    "removed_feature_groups",
    "split_hash",
    "label_version",
    "seed",
    "N_examples_24",
    "N_stays_24",
    "MAE24",
    "RMSE24",
    "directional24",
    "N_examples_48",
    "N_stays_48",
    "MAE48",
    "RMSE48",
    "directional48",
    "run_artifact_sha256",
    "metric_implementation_version",
    "status",
)


def ablation_table_row(result: AblationEvaluation) -> Mapping[str, object]:
    return {
        "variant_id": result.variant_id,
        "run_id": result.run_id,
        "reference_or_ablation": "reference" if not result.removed_feature_groups else "ablation",
        "model_family": result.model_family,
        "parent_run_id": result.parent_run_id,
        "parent_artifact_sha256": result.parent_artifact_sha256,
        "model_config_hash": result.model_config_hash,
        "parent_feature_schema_version": result.parent_feature_schema_version,
        "parent_feature_schema_hash": result.parent_feature_schema_hash,
        "derived_schema_hash": result.derived_schema_hash,
        "removed_feature_groups": "|".join(result.removed_feature_groups),
        "split_hash": result.split_hash,
        "label_version": result.label_version,
        "seed": result.seed,
        "N_examples_24": result.recovery24.counts["N_examples"],
        "N_stays_24": result.recovery24.counts["N_ICU_stays"],
        "MAE24": result.recovery24.metrics["mae"],
        "RMSE24": result.recovery24.metrics["rmse"],
        "directional24": result.recovery24.metrics["directional_agreement"],
        "N_examples_48": result.recovery48.counts["N_examples"],
        "N_stays_48": result.recovery48.counts["N_ICU_stays"],
        "MAE48": result.recovery48.metrics["mae"],
        "RMSE48": result.recovery48.metrics["rmse"],
        "directional48": result.recovery48.metrics["directional_agreement"],
        "run_artifact_sha256": result.run_artifact_sha256,
        "metric_implementation_version": result.metric_implementation_version,
        "status": result.status,
    }


def write_ablation_table(path, results: Sequence[AblationEvaluation]):
    if not results:
        raise AblationValidationError("ablation table requires generated evaluation results")
    table = Path(path)
    table.parent.mkdir(parents=True, exist_ok=True)
    rows = [ablation_table_row(result) for result in results]
    with table.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ABLATION_TABLE_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return table, sha256_file(table)
