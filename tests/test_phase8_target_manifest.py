import json
from pathlib import Path
import pytest
from labels.synthetic_manifest import validate_target_manifest

ROOT=Path(__file__).resolve().parents[1]

@pytest.mark.parametrize("kind,name",[("fixtures","phase8_fixture_v1"),("smoke","phase8_smoke_v1")])
def test_partial_artifacts_validate_and_are_truthfully_blocked(kind,name):
    path=ROOT/f"artifacts/data/synthetic/targets/{kind}/{name}/synthetic_phase8_target_manifest_v1.json"
    manifest=validate_target_manifest(path,ROOT)
    assert manifest["combined_phase8_status"]=="PARTIAL"
    assert manifest["engineering_icu_time_artifact_status"]=="GENERATED"
    assert manifest["final_recovery_artifact_status"].startswith("BLOCKED_")
    assert manifest["organ_support"]=="NOT_CREATED_PHASE9" and manifest["split"]=="NOT_ASSIGNED_PHASE10"

def test_target_schema_has_exact_identity_order_and_no_support_fields():
    schema=json.loads((ROOT/"configs/labels/target_schema_v1.json").read_text())
    assert schema["identity"]==["subject_id","stay_id","prediction_time","grid_index"]
    assert schema["recovery_horizon_order"]==["delta_sofa_24h","delta_sofa_48h"]
    assert schema["organ_support_fields"]=="NOT_OWNED_PHASE9"

def test_manifest_bound_spec_hash_attack_fails(tmp_path):
    source=ROOT/"artifacts/data/synthetic/targets/fixtures/phase8_fixture_v1/synthetic_phase8_target_manifest_v1.json"
    payload=json.loads(source.read_text()); payload["recovery_spec_sha256"]="0"*64
    hostile=tmp_path/"manifest.json"; hostile.write_text(json.dumps(payload))
    with pytest.raises(ValueError,match="recovery_spec_sha256"):
        validate_target_manifest(hostile,ROOT)

def test_label_modules_do_not_import_models_or_randomness():
    text="\n".join((ROOT/path).read_text() for path in ("src/labels/recovery.py","src/labels/icu_time.py","src/labels/synthetic_manifest.py"))
    for forbidden in ("import torch","import random","models.","xgboost","shap","calibrat"):
        assert forbidden not in text.lower()
