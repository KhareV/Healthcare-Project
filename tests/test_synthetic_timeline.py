import copy,json,math
from pathlib import Path
import pytest
from data.synthetic.normalization import SyntheticTimelineError, normalize_events, validate_timeline
from synthetic_timeline_helpers import MAP,ROOT,SCHEMA,normalized,upstream

def data():
    cohort,phase3,ct,rt=upstream(); return cohort,phase3,ct,copy.deepcopy(rt),json.loads(MAP.read_text()),json.loads(SCHEMA.read_text())
def run(raw=None,mapping=None):
    cohort,phase3,ct,rt,m,s=data(); return normalize_events((raw or rt)["raw_events"],ct["retained_cohort"],mapping or m,phase3["raw_schema_version"])

def test_retained_stay_events_are_canonicalized_and_ordered():
    rows,filtered=run(); assert rows and filtered==0
    keys=[(r["subject_id"],r["stay_id"],r["event_time"],r["canonical_concept"],r["event_id"]) for r in rows]; assert keys==sorted(keys)
def test_identity_and_provenance_preserved():
    cohort,phase3,ct,rt,m,s=data(); rows,_=run(); raw={r["event_id"]:r for r in rt["raw_events"]}
    assert all(r["subject_id"]==raw[r["event_id"]]["subject_id"] and r["stay_id"]==raw[r["event_id"]]["stay_id"] and r["source_concept"]==raw[r["event_id"]]["concept_code"] for r in rows)
def test_all_concepts_map_deterministically():
    rows,_=run(); m=json.loads(MAP.read_text())["mappings"]; assert {r["canonical_concept"] for r in rows}=={x["canonical_concept"] for x in m.values()}
def test_units_are_explicit_identity_conversions():
    rows,_=run(); m=json.loads(MAP.read_text())["mappings"]; assert all(r["unit"]==m[r["source_concept"]]["canonical_unit"] and m[r["source_concept"]]["conversion"]=="IDENTITY" for r in rows)
def test_quantity_interval_is_preserved():
    rows,_=run(); urine=[r for r in rows if r["canonical_concept"]=="urine_output_volume"]; assert urine and all(r["event_kind"]=="quantity_interval" and r["interval_end"]==r["event_time"] for r in urine)
def test_same_timestamp_events_are_not_collapsed():
    c,p,ct,rt,m,s=data(); duplicate=copy.deepcopy(rt["raw_events"][0]); duplicate["event_id"]+="-LEGIT"; rt["raw_events"].append(duplicate); rows,_=run(rt); assert len(rows)==len(rt["raw_events"])
def test_nonretained_stay_events_are_filtered():
    c,p,ct,rt,m,s=data(); ct["retained_cohort"]=ct["retained_cohort"][:-1]; rows,filtered=normalize_events(rt["raw_events"],ct["retained_cohort"],m,p["raw_schema_version"]); assert filtered>0 and all(r["stay_id"]!=rt["episodes"][-1]["stay_id"] for r in rows)
def test_cross_stay_subject_mismatch_rejected():
    c,p,ct,rt,m,s=data(); rt["raw_events"][0]["subject_id"]="WRONG"
    with pytest.raises(SyntheticTimelineError,match="subject/stay"): run(rt)
def test_duplicate_event_id_rejected():
    c,p,ct,rt,m,s=data(); rt["raw_events"][1]["event_id"]=rt["raw_events"][0]["event_id"]
    with pytest.raises(SyntheticTimelineError,match="duplicate event_id"): run(rt)
def test_unknown_concept_rejected():
    c,p,ct,rt,m,s=data(); rt["raw_events"][0]["concept_code"]="unknown"
    with pytest.raises(SyntheticTimelineError,match="unknown raw concept"): run(rt)
def test_unsupported_unit_rejected():
    c,p,ct,rt,m,s=data(); rt["raw_events"][0]["unit"]="mystery"
    with pytest.raises(SyntheticTimelineError,match="unsupported unit"): run(rt)
def test_infinite_value_rejected():
    c,p,ct,rt,m,s=data(); rt["raw_events"][0]["value_numeric"]=math.inf
    with pytest.raises(SyntheticTimelineError,match="invalid numeric"): run(rt)
@pytest.mark.parametrize("time",["2000-01-01T00:00:00.000000Z","2200-01-01T00:00:00.000000Z"])
def test_event_outside_bounds_rejected(time):
    c,p,ct,rt,m,s=data(); rt["raw_events"][0]["event_time"]=time
    with pytest.raises(SyntheticTimelineError,match="outside retained episode"): run(rt)
def test_exact_existing_boundaries_validate():
    rows,_=run(); assert validate_timeline(rows,json.loads(SCHEMA.read_text()),json.loads(MAP.read_text()))
@pytest.mark.parametrize("field",["trajectory","outcome_class","remaining_stay","split","latent_state"])
def test_forbidden_or_unknown_source_field_rejected(field):
    c,p,ct,rt,m,s=data(); rt["raw_events"][0][field]="attack"
    with pytest.raises(SyntheticTimelineError,match="unknown, missing, or prohibited"): run(rt)
def test_double_normalization_rejected_by_stage_and_shape():
    rows=normalized(); c,p,ct,rt,m,s=data()
    with pytest.raises(SyntheticTimelineError): normalize_events(rows,ct["retained_cohort"],m,p["raw_schema_version"])
