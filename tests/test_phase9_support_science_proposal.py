import json
from pathlib import Path
import yaml

from data.synthetic.provenance import sha256_file
from data.synthetic.support_proposal import validate_support_proposal

ROOT=Path(__file__).resolve().parents[1]
PROCESS=ROOT/"configs/synthetic/support_process_v1.proposed.yaml"
DICTIONARY=ROOT/"configs/event_dict_v2.proposed.yaml"
INVENTORY=ROOT/"configs/synthetic/support_unresolved_inventory_v1.json"

def test_complete_non_authorizing_proposals_validate():
    process,dictionary,inventory=validate_support_proposal(PROCESS,DICTIONARY,INVENTORY)
    assert process["status"]==dictionary["status"]=="PROPOSED_NOT_AUTHORIZED"
    assert len(inventory["parameters"])==20

def test_no_hidden_nulls_or_todo_markers_in_proposed_parameter_sets():
    for path in (PROCESS,DICTIONARY):
        text=path.read_text().upper()
        assert " NULL" not in text and ": NULL" not in text
        assert all(marker not in text for marker in ("TODO","TBD","PLACEHOLDER","UNRESOLVED","DRAFT"))

def test_vocabulary_roles_units_and_policies_are_explicit():
    process=yaml.safe_load(PROCESS.read_text()); dictionary=yaml.safe_load(DICTIONARY.read_text())
    vaso={x["canonical_name"]:x for x in process["vasoactive_vocabulary"]["concepts"]}
    assert vaso["dobutamine"]["endpoint_qualifying"] is False and vaso["dobutamine"]["sofa_cardiovascular_relevant"] is True
    assert all(x["sofa_cardiovascular_relevant"] for x in vaso.values())
    assert process["vasoactive_vocabulary"]["canonical_rate_unit"]=="ug/kg/min"
    assert dictionary["endpoint"]["onset_equal_outtime"]=="INVALID_AMBIGUOUS_RECORD_FAIL_CLOSED"
    assert dictionary["endpoint"]["unknown_state"]=="FAIL_CLOSED_NOT_OFF"
    assert dictionary["intervals"]["duplicate_event_id"]=="FAIL_CLOSED"

def test_existing_authorities_remain_byte_identical():
    assert sha256_file(ROOT/"configs/synthetic/synthetic_generator_v1.yaml")=="5f7d3f73572a34282c214860f1c0c656b18727a16fa0c4687caf071f690a7484"
    assert sha256_file(ROOT/"configs/event_dict_v1.yaml")=="717f6183ad8014602b3cc32eeaa3a810eab8a26a19937bda8e214200bb2f9620"

def test_historical_proposal_remediation_paths_remain_separate_from_final_outputs():
    assert not (ROOT/"artifacts/data/synthetic/support").exists()
    assert not (ROOT/"artifacts/labels/synthetic_phase9").exists()
    final = ROOT/"artifacts/data/synthetic/final/phase9_final_v1/synthetic_dataset_manifest_v1.json"
    if final.exists():
        payload=json.loads(final.read_text())
        assert payload["manifest_status"]=="AUTHORIZED_FINAL_SYNTHETIC_DATA"
        assert payload["scientific_support_generation_status"]=="GENERATED_FROZEN_PHASE9"
    for p in ROOT.glob("artifacts/data/synthetic/*/phase3_*_v1/support_intervals.jsonl"):
        assert p.read_text()==""

def test_final_generation_plan_is_full_replay_and_pre_split():
    process=yaml.safe_load(PROCESS.read_text())
    plan=process["final_generation_plan"]
    assert plan["strategy"]=="FULL_PHASE3_DETERMINISTIC_REPLAY_REQUIRED"
    assert plan["primary_seed"] not in (42,31001,31002)
    assert plan["n_subjects"]!=100
    assert "split" not in process and "split_assignment" not in process

def test_every_numerical_parameter_group_has_design_governance():
    process=yaml.safe_load(PROCESS.read_text())
    groups=process["numerical_parameter_governance"]
    assert len(groups)>=10
    allowed={"CLINICAL_DEFINITION","PROJECT_SYNTHETIC_DESIGN","ENGINEERING_ONLY"}
    for item in groups.values():
        assert {"scale","proposed","plausible_range","rationale","sensitivity_concern","source_type"}<=set(item)
        assert item["source_type"] in allowed
