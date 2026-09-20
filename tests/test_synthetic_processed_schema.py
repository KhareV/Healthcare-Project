import csv,json
from pathlib import Path
from synthetic_timeline_helpers import MAP,ROOT,SCHEMA,normalized

def test_processed_schema_is_v1_because_no_prior_production_schema_exists():
    schema=json.loads(SCHEMA.read_text()); assert schema["schema_version"]=="synthetic_processed_timeline_v1" and schema["stage"]=="CANONICAL_TIMELINE"
def test_no_phase7_or_label_fields_in_schema():
    names={f["name"] for f in json.loads(SCHEMA.read_text())["fields"]}; forbidden={"bin","mask","tslo","padding","sofa","target","label","split","imputed_value"}; assert not names&forbidden
def test_event_time_and_identity_roles_are_explicit():
    fields={f["name"]:f for f in json.loads(SCHEMA.read_text())["fields"]}; assert fields["event_time"]["role"]=="EVENT_AVAILABILITY" and fields["subject_id"]["model_eligible"] is False and fields["stay_id"]["model_eligible"] is False
def test_every_present_concept_has_machine_provenance():
    with (ROOT/"docs/sanskruti/feature_provenance_v1.csv").open(newline="") as h: records={r["canonical_concept"]:r for r in csv.DictReader(h)}
    assert {r["canonical_concept"] for r in normalized()}<=set(records)
    assert all(records[r["canonical_concept"]]["provenance_id"]==r["provenance_id"] for r in normalized())
def test_sofa_six_domain_dependencies_are_structurally_complete():
    mapping=json.loads(MAP.read_text())["mappings"]; assert {v["sofa_dependency"] for v in mapping.values() if v["sofa_dependency"]!="NONE"}=={"RESPIRATORY","COAGULATION","LIVER","CARDIOVASCULAR","CNS","RENAL"}
def test_support_dependencies_are_explicitly_deferred():
    mapping=json.loads(MAP.read_text())["mappings"]; assert mapping["pao2"]["support_dependency"].endswith("PHASE9") and mapping["mean_arterial_pressure"]["support_dependency"].endswith("PHASE9")
def test_outtime_is_documented_structural_not_model_feature():
    text=(ROOT/"docs/sanskruti/FEATURE_DICTIONARY.md").read_text(); assert "`outtime`" in text and "not model inputs" in text
def test_feature_order_and_derivations_remain_unfrozen():
    with (ROOT/"docs/sanskruti/feature_provenance_v1.csv").open(newline="") as h: rows=list(csv.DictReader(h)); assert all(r["planned_phase7_derivations"]=="UNFROZEN" for r in rows)
