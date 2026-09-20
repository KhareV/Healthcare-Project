"""Load and validate the Phase-2 synthetic-generator specification.

The validator checks governance, schema, leakage, and integration contracts. It
does not simulate patients, construct labels, assign splits, or import modeling
code. Phase 3 owns executable generation.
"""

import csv
import hashlib
import json
from pathlib import Path
from typing import Dict, Mapping, Set, Tuple

import yaml


GENERATOR_SPEC_PATH = Path("configs/synthetic/synthetic_generator_v1.yaml")
RAW_SCHEMA_PATH = Path("configs/synthetic/synthetic_schema_v1.json")
MANIFEST_TEMPLATE_PATH = Path(
    "configs/synthetic/synthetic_dataset_manifest_template_v1.json"
)
FIELD_ROLE_MATRIX_PATH = Path(
    "docs/sanskruti/SYNTHETIC_FIELD_ROLE_MATRIX_V1.csv"
)
SCOPE_PATH = Path("configs/governance/project_scope_v2.json")

EXPECTED_GENERATOR_VERSION = "synthetic_generator_v1"
EXPECTED_CONFIG_VERSION = "synthetic_generator_config_v1"
EXPECTED_SCHEMA_VERSION = "synthetic_raw_schema_v1"
EXPECTED_SCOPE_VERSION = "PROJECT_SCOPE_V2"
EXPECTED_STATUS = "DRAFT_SPECIFICATION_REVIEW_REQUIRED"

REQUIRED_TABLES = {
    "subjects",
    "episodes",
    "raw_events",
    "support_intervals",
    "generator_internal_audit",
}
REQUIRED_SOFA_DOMAINS = {
    "RESPIRATORY",
    "COAGULATION",
    "LIVER",
    "CARDIOVASCULAR",
    "CNS",
    "RENAL",
}
REQUIRED_SOFA_CONCEPTS = {
    "pao2",
    "fio2",
    "platelet_count",
    "bilirubin_total",
    "mean_arterial_pressure",
    "glasgow_coma_scale",
    "creatinine",
    "urine_output_volume",
}
ALLOWED_ROLES = {
    "MODEL_ELIGIBLE_RAW",
    "STATIC_MODEL_ELIGIBLE",
    "STRUCTURAL_LABEL_ONLY",
    "PROVENANCE_ONLY",
    "GENERATOR_INTERNAL_AUDIT_ONLY",
}
PROHIBITED_MODEL_FIELD_NAMES = {
    "target",
    "label",
    "trajectory",
    "trajectory_class",
    "outcome_class",
    "recovery_class",
    "deterioration_class",
    "future_sofa",
    "future_state",
    "future_support",
    "remaining_stay",
    "remaining_stay_hours",
    "delta_sofa_24",
    "delta_sofa_48",
    "support_label",
    "support_will_start",
    "organ_support_label",
    "final_outcome",
    "split",
    "split_membership",
    "train_val_test",
    "y",
}
REQUIRED_MANIFEST_FIELDS = {
    "project_scope_version",
    "project_scope_sha256",
    "generator_version",
    "generator_code_identity",
    "generator_config_version",
    "generator_config_sha256",
    "raw_schema_version",
    "raw_schema_sha256",
    "seed_policy_version",
    "primary_seed_identity",
    "artifacts",
}


class SyntheticSpecError(ValueError):
    """Raised when the Phase-2 specification violates a frozen contract."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path) -> Mapping[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SyntheticSpecError("cannot load JSON specification: " + str(path)) from error
    if not isinstance(value, dict):
        raise SyntheticSpecError("JSON specification must be an object: " + str(path))
    return value


def _load_yaml(path: Path) -> Mapping[str, object]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as error:
        raise SyntheticSpecError("cannot load YAML specification: " + str(path)) from error
    if not isinstance(value, dict):
        raise SyntheticSpecError("YAML specification must be a mapping: " + str(path))
    return value


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise SyntheticSpecError(name + " must be a mapping")
    return value


def _sequence(value: object, name: str) -> Tuple[object, ...]:
    if not isinstance(value, list):
        raise SyntheticSpecError(name + " must be a list")
    return tuple(value)


def _validate_scope(root: Path, config: Mapping[str, object], schema: Mapping[str, object]) -> None:
    scope = _load_json(root / SCOPE_PATH)
    if scope.get("scope_version") != EXPECTED_SCOPE_VERSION:
        raise SyntheticSpecError("Project Scope v2 is required")
    if scope.get("status") != "DRAFT_COMPLETE_TEAM_FREEZE_REQUIRED":
        raise SyntheticSpecError("unexpected Project Scope v2 status")
    tasks = _mapping(scope.get("tasks"), "scope.tasks")
    if set(tasks) != {"recovery", "remaining_episode_time", "organ_support"}:
        raise SyntheticSpecError("scientific task conflict with Project Scope v2")
    time = _mapping(scope.get("time"), "scope.time")
    history = _mapping(scope.get("history"), "scope.history")
    if (
        time.get("candidate_formula") != "t_k = intime + 24h + 6h*k"
        or time.get("retention_rule") != "t_k <= outtime - 6h"
        or history.get("interval") != "(t-48h, t]"
        or history.get("number_of_bins") != 8
    ):
        raise SyntheticSpecError("clock/history conflict with Project Scope v2")
    scope_hash = _sha256(root / SCOPE_PATH)
    config_scope = _mapping(config.get("scope"), "config.scope")
    if config_scope.get("version") != EXPECTED_SCOPE_VERSION or config_scope.get("sha256") != scope_hash:
        raise SyntheticSpecError("generator scope reference/hash mismatch")
    if schema.get("scope_version") != EXPECTED_SCOPE_VERSION or schema.get("scope_sha256") != scope_hash:
        raise SyntheticSpecError("raw schema scope reference/hash mismatch")


def _schema_fields(schema: Mapping[str, object]) -> Dict[Tuple[str, str], Mapping[str, object]]:
    tables = _mapping(schema.get("tables"), "schema.tables")
    if set(tables) != REQUIRED_TABLES:
        raise SyntheticSpecError("raw schema table set is incomplete or unexpected")
    result: Dict[Tuple[str, str], Mapping[str, object]] = {}
    for table_name, raw_table in tables.items():
        table = _mapping(raw_table, "schema table " + table_name)
        fields = _sequence(table.get("fields"), "fields for " + table_name)
        for raw_field in fields:
            field = _mapping(raw_field, "field in " + table_name)
            name = field.get("name")
            if not isinstance(name, str) or not name:
                raise SyntheticSpecError("schema field name must be non-empty")
            key = (table_name, name)
            if key in result:
                raise SyntheticSpecError("duplicate schema field: " + repr(key))
            role = field.get("role")
            if role not in ALLOWED_ROLES:
                raise SyntheticSpecError("invalid field role: " + str(role))
            if field.get("model_eligible") is not (role in {"MODEL_ELIGIBLE_RAW", "STATIC_MODEL_ELIGIBLE"}):
                raise SyntheticSpecError("field role/model eligibility mismatch: " + repr(key))
            if field.get("model_eligible") and name.lower() in PROHIBITED_MODEL_FIELD_NAMES:
                raise SyntheticSpecError("prohibited model-visible field: " + name)
            result[key] = field
    return result


def _require_fields(fields: Mapping[Tuple[str, str], object]) -> None:
    required = {
        ("subjects", "subject_id"),
        ("episodes", "subject_id"),
        ("episodes", "stay_id"),
        ("episodes", "intime"),
        ("episodes", "outtime"),
        ("raw_events", "event_id"),
        ("raw_events", "stay_id"),
        ("raw_events", "event_time"),
        ("support_intervals", "support_event_id"),
        ("support_intervals", "stay_id"),
        ("support_intervals", "interval_start"),
        ("support_intervals", "interval_end"),
    }
    missing = required - set(fields)
    if missing:
        raise SyntheticSpecError("required raw fields are missing: " + repr(sorted(missing)))
    if fields[("episodes", "outtime")].get("model_eligible") is not False:
        raise SyntheticSpecError("outtime must never be model eligible")


def _validate_table_relationships(schema: Mapping[str, object]) -> None:
    tables = _mapping(schema.get("tables"), "schema.tables")
    expected_primary_keys = {
        "subjects": ["subject_id"],
        "episodes": ["stay_id"],
        "raw_events": ["event_id"],
        "support_intervals": ["support_event_id"],
        "generator_internal_audit": ["internal_state_id"],
    }
    expected_references = {
        "episodes": {"subjects(subject_id)"},
        "raw_events": {"subjects(subject_id)", "episodes(stay_id)"},
        "support_intervals": {"subjects(subject_id)", "episodes(stay_id)"},
        "generator_internal_audit": {"episodes(stay_id)"},
    }
    for table_name, primary_key in expected_primary_keys.items():
        table = _mapping(tables.get(table_name), "schema table " + table_name)
        if table.get("primary_key") != primary_key:
            raise SyntheticSpecError("unexpected primary key for " + table_name)
        references = {
            str(_mapping(value, "foreign key").get("references"))
            for value in _sequence(table.get("foreign_keys"), "foreign keys")
        }
        if references != expected_references.get(table_name, set()):
            raise SyntheticSpecError("foreign-key contract mismatch for " + table_name)
    internal = _mapping(tables.get("generator_internal_audit"), "internal audit table")
    if internal.get("exported") is not False:
        raise SyntheticSpecError("generator internal audit table must not be exported")


def _validate_field_matrix(root: Path, fields: Mapping[Tuple[str, str], Mapping[str, object]]) -> None:
    try:
        with (root / FIELD_ROLE_MATRIX_PATH).open(encoding="utf-8", newline="") as handle:
            rows = tuple(csv.DictReader(handle))
    except OSError as error:
        raise SyntheticSpecError("cannot load field-role matrix") from error
    matrix_keys = {(row.get("table", ""), row.get("field", "")) for row in rows}
    if matrix_keys != set(fields):
        raise SyntheticSpecError("field-role matrix and raw schema fields diverge")
    for row in rows:
        key = (row["table"], row["field"])
        expected = "true" if fields[key]["model_eligible"] else "false"
        if row.get("model_eligible") != expected:
            raise SyntheticSpecError("matrix model eligibility mismatch: " + repr(key))


def _validate_science_coverage(config: Mapping[str, object]) -> None:
    inventory = _mapping(config.get("raw_variable_inventory"), "raw_variable_inventory")
    if not REQUIRED_SOFA_CONCEPTS.issubset(inventory):
        raise SyntheticSpecError("six-component SOFA raw concepts are incomplete")
    domains: Set[str] = set()
    for raw_value in inventory.values():
        value = _mapping(raw_value, "raw variable")
        domain = value.get("domain")
        if isinstance(domain, str):
            domains.add(domain)
        if not value.get("unit") or not value.get("event_semantics"):
            raise SyntheticSpecError("raw variables require units and event semantics")
    if not REQUIRED_SOFA_DOMAINS.issubset(domains):
        raise SyntheticSpecError("six-component SOFA domain coverage is incomplete")
    support = _mapping(config.get("support_process"), "support_process")
    if support.get("interval_convention") != "CLOSED_OPEN":
        raise SyntheticSpecError("support intervals must use CLOSED_OPEN semantics")
    ventilation = _mapping(support.get("ventilation"), "support_process.ventilation")
    categories = set(_sequence(ventilation.get("internal_categories"), "ventilation categories"))
    if not {"INVASIVE", "NON_INVASIVE", "HFNC", "ORDINARY_OXYGEN"}.issubset(categories):
        raise SyntheticSpecError("ventilation categories cannot support Pulkit contract")
    if ventilation.get("endpoint_qualifying_categories") != ["INVASIVE"]:
        raise SyntheticSpecError("only INVASIVE ventilation may qualify")


def _validate_manifest(root: Path, schema: Mapping[str, object]) -> None:
    template = _load_json(root / MANIFEST_TEMPLATE_PATH)
    if template.get("manifest_status") != "SPECIFICATION_ONLY_NO_DATA_GENERATED":
        raise SyntheticSpecError("manifest template must remain specification-only")
    if not REQUIRED_MANIFEST_FIELDS.issubset(template):
        raise SyntheticSpecError("manifest provenance contract is incomplete")
    manifest_contract = _mapping(schema.get("manifest_contract"), "schema.manifest_contract")
    declared = set(_sequence(manifest_contract.get("required_fields"), "manifest required fields"))
    if not REQUIRED_MANIFEST_FIELDS.issubset(declared):
        raise SyntheticSpecError("schema manifest contract is incomplete")
    if template.get("raw_schema_sha256") != _sha256(root / RAW_SCHEMA_PATH):
        raise SyntheticSpecError("manifest template raw-schema hash mismatch")


def _validate_governance(config: Mapping[str, object], schema: Mapping[str, object]) -> None:
    if config.get("generator_spec_version") != EXPECTED_GENERATOR_VERSION:
        raise SyntheticSpecError("unexpected generator specification version")
    if config.get("config_version") != EXPECTED_CONFIG_VERSION:
        raise SyntheticSpecError("unexpected generator config version")
    if config.get("status") != EXPECTED_STATUS or schema.get("status") != EXPECTED_STATUS:
        raise SyntheticSpecError("Phase-2 specifications must remain review-required")
    if config.get("artifact_status") != "SPECIFICATION_ONLY_NO_DATA_GENERATED":
        raise SyntheticSpecError("generator config falsely claims generated data")
    if schema.get("artifact_status") != "SPECIFICATION_ONLY_NO_DATA_GENERATED":
        raise SyntheticSpecError("raw schema falsely claims generated data")
    review = _mapping(config.get("review"), "config.review")
    if review.get("approved_by") != [] or review.get("team_freeze_required") is not True:
        raise SyntheticSpecError("approval cannot be fabricated")
    rng = _mapping(config.get("rng_policy"), "rng_policy")
    if rng.get("primary_seed") is not None or rng.get("global_rng_mutation_allowed") is not False:
        raise SyntheticSpecError("final seed must remain unlocked and global RNG is prohibited")
    population = _mapping(config.get("population"), "population")
    if population.get("n_subjects") is not None:
        raise SyntheticSpecError("final cohort size must remain unlocked")
    output = _mapping(config.get("output_policy"), "output_policy")
    if output.get("storage_format") is not None:
        raise SyntheticSpecError("primary storage format must remain unlocked")
    architecture = _mapping(config.get("architecture"), "architecture")
    prohibited_outputs = set(_sequence(architecture.get("generator_must_not_emit"), "generator_must_not_emit"))
    for required in ("prediction_cutoffs", "canonical_examples", "recovery_labels", "split_membership"):
        if required not in prohibited_outputs:
            raise SyntheticSpecError("missing downstream-separation prohibition: " + required)
    top_level_prohibited = {"model", "training", "calibration", "selection", "split"}
    if top_level_prohibited.intersection(config):
        raise SyntheticSpecError("model/training/split configuration embedded in generator spec")


def validate_phase2_specification(root: Path) -> Mapping[str, object]:
    """Validate all Phase-2 artifacts without producing scientific data."""

    root = root.resolve()
    config = _load_yaml(root / GENERATOR_SPEC_PATH)
    schema = _load_json(root / RAW_SCHEMA_PATH)
    _validate_governance(config, schema)
    _validate_scope(root, config, schema)
    fields = _schema_fields(schema)
    _require_fields(fields)
    _validate_table_relationships(schema)
    _validate_field_matrix(root, fields)
    _validate_science_coverage(config)
    _validate_manifest(root, schema)
    return {
        "status": "PASS",
        "generator_spec_version": config["generator_spec_version"],
        "raw_schema_version": schema["schema_version"],
        "table_count": len(_mapping(schema["tables"], "schema.tables")),
        "field_count": len(fields),
        "raw_variable_count": len(_mapping(config["raw_variable_inventory"], "raw_variable_inventory")),
        "final_dataset_generated": False,
    }
