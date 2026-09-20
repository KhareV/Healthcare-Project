import csv
import hashlib
import json
import shutil
from pathlib import Path

import pytest
import yaml

from data.synthetic.spec import (
    FIELD_ROLE_MATRIX_PATH,
    GENERATOR_SPEC_PATH,
    MANIFEST_TEMPLATE_PATH,
    RAW_SCHEMA_PATH,
    SCOPE_PATH,
    SyntheticSpecError,
    validate_phase2_specification,
)


ROOT = Path(__file__).resolve().parents[1]
HUMAN_SPEC = ROOT / "docs/sanskruti/SYNTHETIC_GENERATOR_SPEC_V1.md"
THREAT_MODEL = ROOT / "docs/sanskruti/SYNTHETIC_GENERATOR_LEAKAGE_THREAT_MODEL_V1.md"
REVIEW = ROOT / "docs/sanskruti/PHASE2_GENERATOR_SPEC_REVIEW.md"
HASH_MANIFEST = ROOT / "docs/sanskruti/phase2_generator_spec_hashes.json"


def _yaml():
    return yaml.safe_load((ROOT / GENERATOR_SPEC_PATH).read_text(encoding="utf-8"))


def _schema():
    return json.loads((ROOT / RAW_SCHEMA_PATH).read_text(encoding="utf-8"))


def _isolated_spec(tmp_path):
    for relative in (
        GENERATOR_SPEC_PATH,
        RAW_SCHEMA_PATH,
        MANIFEST_TEMPLATE_PATH,
        FIELD_ROLE_MATRIX_PATH,
        SCOPE_PATH,
    ):
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
    return tmp_path


def test_phase2_required_artifacts_exist():
    for path in (
        ROOT / GENERATOR_SPEC_PATH,
        ROOT / RAW_SCHEMA_PATH,
        ROOT / MANIFEST_TEMPLATE_PATH,
        ROOT / FIELD_ROLE_MATRIX_PATH,
        HUMAN_SPEC,
        THREAT_MODEL,
        REVIEW,
        HASH_MANIFEST,
    ):
        assert path.is_file(), path


def test_complete_specification_validator_passes():
    result = validate_phase2_specification(ROOT)
    assert result == {
        "status": "PASS",
        "generator_spec_version": "synthetic_generator_v1",
        "raw_schema_version": "synthetic_raw_schema_v1",
        "table_count": 5,
        "field_count": 37,
        "raw_variable_count": 15,
        "final_dataset_generated": False,
    }


def test_phase2_hash_manifest_matches_specification_artifacts():
    manifest = json.loads(HASH_MANIFEST.read_text(encoding="utf-8"))
    assert manifest["algorithm"] == "SHA-256"
    assert manifest["status"] == "DRAFT_SPECIFICATION_REVIEW_REQUIRED"
    for relative_path, expected in manifest["files"].items():
        actual = hashlib.sha256((ROOT / relative_path).read_bytes()).hexdigest()
        assert actual == expected, relative_path


def test_spec_is_review_required_and_does_not_fabricate_approval():
    spec = _yaml()
    assert spec["status"] == "DRAFT_SPECIFICATION_REVIEW_REQUIRED"
    assert spec["artifact_status"] == "SPECIFICATION_ONLY_NO_DATA_GENERATED"
    assert spec["review"]["approved_by"] == []
    assert spec["review"]["team_freeze_required"] is True


def test_scope_reference_and_versions_are_explicit():
    spec = _yaml()
    schema = _schema()
    assert spec["scope"]["version"] == "PROJECT_SCOPE_V2"
    assert spec["generator_spec_version"] == "synthetic_generator_v1"
    assert spec["config_version"] == "synthetic_generator_config_v1"
    assert schema["schema_version"] == "synthetic_raw_schema_v1"


def test_subject_episode_and_explicit_boundary_contract():
    schema = _schema()
    assert schema["tables"]["subjects"]["primary_key"] == ["subject_id"]
    assert schema["tables"]["episodes"]["primary_key"] == ["stay_id"]
    episode_fields = {field["name"]: field for field in schema["tables"]["episodes"]["fields"]}
    assert {"subject_id", "stay_id", "intime", "outtime"}.issubset(episode_fields)
    assert episode_fields["outtime"]["model_eligible"] is False
    assert _yaml()["episode_process"]["episodes_per_subject"] == 1


def test_raw_events_are_long_form_and_cutoff_safe_by_contract():
    schema = _schema()
    fields = {field["name"] for field in schema["tables"]["raw_events"]["fields"]}
    assert {"event_id", "subject_id", "stay_id", "event_time", "concept_code", "value_numeric", "unit"}.issubset(fields)
    assert schema["timestamp_contract"]["event_bounds"] == "intime <= event_time <= outtime"
    assert _yaml()["raw_schema"]["representation"] == "LONG_FORM_EVENT_TABLES"


def test_generator_does_not_own_cutoffs_bins_labels_splits_or_preprocessing():
    spec = _yaml()
    prohibited = set(spec["architecture"]["generator_must_not_emit"])
    assert {
        "prediction_cutoffs", "canonical_examples", "six_hour_bins",
        "imputed_values", "recovery_labels", "remaining_stay_labels",
        "organ_support_labels", "task_eligibility", "split_membership",
    }.issubset(prohibited)
    assert spec["downstream_separation"]["subject_split"]["generator_emits"] is False


def test_no_model_visible_target_trajectory_or_split_field():
    schema = _schema()
    prohibited = set(_yaml()["forbidden_exports"]["exact_names"])
    fields = [field for table in schema["tables"].values() for field in table["fields"]]
    visible_names = {field["name"].lower() for field in fields if field["model_eligible"]}
    assert visible_names.isdisjoint(prohibited)
    assert "trajectory" not in visible_names
    assert "split" not in visible_names


def test_six_component_sofa_prerequisite_concepts_and_domains_exist():
    inventory = _yaml()["raw_variable_inventory"]
    required = {
        "pao2", "fio2", "platelet_count", "bilirubin_total",
        "mean_arterial_pressure", "glasgow_coma_scale", "creatinine",
        "urine_output_volume",
    }
    assert required.issubset(inventory)
    assert {inventory[name]["domain"] for name in required} == {
        "RESPIRATORY", "COAGULATION", "LIVER", "CARDIOVASCULAR", "CNS", "RENAL"
    }
    assert all(inventory[name]["unit"] for name in required)


def test_support_contract_can_feed_pulkit_state_engines():
    support = _yaml()["support_process"]
    assert support["interval_convention"] == "CLOSED_OPEN"
    assert support["vasopressor"]["pulkit_adapter_target"] == "NormalizedActiveInterval"
    assert support["ventilation"]["pulkit_adapter_target"] == "NormalizedVentilationInterval"
    assert support["ventilation"]["endpoint_qualifying_categories"] == ["INVASIVE"]
    assert set(support["ventilation"]["explicitly_nonqualifying_categories"]) == {
        "NON_INVASIVE", "HFNC", "ORDINARY_OXYGEN"
    }


def test_rng_is_local_deterministic_and_final_seed_remains_unlocked():
    rng = _yaml()["rng_policy"]
    assert rng["engine"] == "numpy.random.Generator"
    assert rng["global_rng_mutation_allowed"] is False
    assert rng["primary_seed"] is None
    assert "SeedSequence" in rng["deterministic_substreams"]["derivation"]
    assert len(set(rng["deterministic_substreams"]["component_codes"].values())) == 7


def test_final_size_storage_and_subtypes_remain_unlocked_or_review_required():
    spec = _yaml()
    assert spec["population"]["n_subjects"] is None
    assert spec["output_policy"]["storage_format"] is None
    assert spec["population"]["cardiac_condition_group"]["vocabulary"] is None
    assert spec["population"]["cardiac_condition_group"]["heart_failure_only"] is False


def test_manifest_is_only_a_template_with_required_provenance():
    manifest = json.loads((ROOT / MANIFEST_TEMPLATE_PATH).read_text(encoding="utf-8"))
    assert manifest["manifest_status"] == "SPECIFICATION_ONLY_NO_DATA_GENERATED"
    assert manifest["generator_config_sha256"] is None
    assert manifest["primary_seed_identity"] is None
    assert manifest["artifacts"] == []
    for key in (
        "project_scope_sha256", "generator_version", "generator_code_identity",
        "raw_schema_version", "raw_schema_sha256", "seed_policy_version",
        "record_counts", "artifacts",
    ):
        assert key in manifest


def test_field_role_matrix_exactly_matches_schema():
    schema_fields = {
        (table, field["name"]): field
        for table, definition in _schema()["tables"].items()
        for field in definition["fields"]
    }
    with (ROOT / FIELD_ROLE_MATRIX_PATH).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert {(row["table"], row["field"]) for row in rows} == set(schema_fields)
    for row in rows:
        field = schema_fields[(row["table"], row["field"])]
        assert row["model_eligible"] == str(field["model_eligible"]).lower()


def test_human_spec_contains_nonclinical_and_phase_boundaries():
    text = HUMAN_SPEC.read_text(encoding="utf-8")
    for phrase in (
        "not a MIMIC replica", "not clinical validation", "NOT GENERATED",
        "NOT IMPLEMENTED — PHASE 3", "does not calculate", "No training-",
        "one shared canonical builder",
    ):
        assert phrase in text


def test_nonclinical_limitations_are_machine_readable():
    limitations = _yaml()["limitations"]
    assert limitations["not_a_mimic_replica"] is True
    assert limitations["not_epidemiology"] is True
    assert limitations["not_a_digital_twin"] is True
    assert limitations["not_a_causal_simulator"] is True
    assert limitations["not_clinically_validated"] is True
    assert limitations["no_clinical_deployment_claim"] is True


def test_no_mimic_dependency_or_model_training_configuration_is_embedded():
    spec = _yaml()
    assert not {"model", "training", "calibration", "selection", "split"}.intersection(spec)
    text = (ROOT / GENERATOR_SPEC_PATH).read_text(encoding="utf-8").lower()
    for dependency in ("physionet_credential", "mimic_table", "mimic_itemid", "mimic_code_commit"):
        assert dependency not in text
    assert spec["limitations"]["synthetic_data_reason"] == "MIMIC_AUTHORIZATION_WAS_REQUESTED_BUT_NOT_OBTAINED"


def test_threat_model_covers_required_leakage_routes_and_later_owners():
    text = THREAT_MODEL.read_text(encoding="utf-8")
    for phrase in (
        "Latent trajectory exposed", "Future physiology exported at cutoff",
        "`outtime` feature", "Future support flag", "Split-specific generation",
        "Subject clones cross split", "Three deterministic templates",
        "Support threshold copies visible feature", "Serving/training divergence",
        "Phase 3", "Phase 7", "Phase 9", "Phase 10",
    ):
        assert phrase in text


def test_spec_module_has_no_training_or_model_imports():
    text = (ROOT / "src/data/synthetic/spec.py").read_text(encoding="utf-8")
    assert "import torch" not in text
    assert "import xgboost" not in text
    assert "from training" not in text
    assert "from models" not in text
    assert "from evaluation" not in text


def test_phase3_generator_does_not_turn_phase2_template_into_final_data():
    assert (ROOT / "src/data/synthetic/generator.py").is_file()
    assert not (ROOT / "data/generated").exists()
    manifests = list((ROOT / "artifacts/data/synthetic").glob("**/synthetic_dataset_manifest_v1.json"))
    for path in manifests:
        assert json.loads(path.read_text())["manifest_status"] in {"FIXTURE", "SMOKE"}


def test_validator_rejects_model_visible_target_alias(tmp_path):
    root = _isolated_spec(tmp_path)
    path = root / RAW_SCHEMA_PATH
    schema = json.loads(path.read_text())
    schema["tables"]["subjects"]["fields"].append({
        "name": "future_support", "type": "boolean", "nullable": False,
        "role": "STATIC_MODEL_ELIGIBLE", "model_eligible": True,
        "unit": None, "allowed_domain": [False, True], "description": "attack",
    })
    path.write_text(json.dumps(schema), encoding="utf-8")
    with pytest.raises(SyntheticSpecError, match="prohibited model-visible field"):
        validate_phase2_specification(root)


def test_validator_rejects_outtime_as_model_feature(tmp_path):
    root = _isolated_spec(tmp_path)
    path = root / RAW_SCHEMA_PATH
    schema = json.loads(path.read_text())
    field = next(f for f in schema["tables"]["episodes"]["fields"] if f["name"] == "outtime")
    field["role"] = "MODEL_ELIGIBLE_RAW"
    field["model_eligible"] = True
    path.write_text(json.dumps(schema), encoding="utf-8")
    with pytest.raises(SyntheticSpecError, match="outtime must never"):
        validate_phase2_specification(root)


def test_validator_rejects_split_configuration(tmp_path):
    root = _isolated_spec(tmp_path)
    path = root / GENERATOR_SPEC_PATH
    spec = yaml.safe_load(path.read_text())
    spec["split"] = {"train": 0.7, "validation": 0.15, "test": 0.15}
    path.write_text(yaml.safe_dump(spec), encoding="utf-8")
    with pytest.raises(SyntheticSpecError, match="model/training/split"):
        validate_phase2_specification(root)


def test_validator_rejects_missing_sofa_domain(tmp_path):
    root = _isolated_spec(tmp_path)
    path = root / GENERATOR_SPEC_PATH
    spec = yaml.safe_load(path.read_text())
    del spec["raw_variable_inventory"]["bilirubin_total"]
    path.write_text(yaml.safe_dump(spec), encoding="utf-8")
    with pytest.raises(SyntheticSpecError, match="SOFA raw concepts"):
        validate_phase2_specification(root)


def test_validator_rejects_incomplete_ventilation_categories(tmp_path):
    root = _isolated_spec(tmp_path)
    path = root / GENERATOR_SPEC_PATH
    spec = yaml.safe_load(path.read_text())
    spec["support_process"]["ventilation"]["internal_categories"].remove("HFNC")
    path.write_text(yaml.safe_dump(spec), encoding="utf-8")
    with pytest.raises(SyntheticSpecError, match="ventilation categories"):
        validate_phase2_specification(root)


def test_validator_rejects_scope_hash_drift(tmp_path):
    root = _isolated_spec(tmp_path)
    path = root / GENERATOR_SPEC_PATH
    spec = yaml.safe_load(path.read_text())
    spec["scope"]["sha256"] = "0" * 64
    path.write_text(yaml.safe_dump(spec), encoding="utf-8")
    with pytest.raises(SyntheticSpecError, match="scope reference/hash"):
        validate_phase2_specification(root)


def test_validator_rejects_internal_latent_export(tmp_path):
    root = _isolated_spec(tmp_path)
    path = root / RAW_SCHEMA_PATH
    schema = json.loads(path.read_text())
    schema["tables"]["generator_internal_audit"]["exported"] = True
    path.write_text(json.dumps(schema), encoding="utf-8")
    with pytest.raises(SyntheticSpecError, match="must not be exported"):
        validate_phase2_specification(root)
