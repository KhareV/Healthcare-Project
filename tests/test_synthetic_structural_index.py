from datetime import timedelta
import json
from pathlib import Path
from data.synthetic.cohort import INDEX_SCHEMA_VERSION
from synthetic_cohort_helpers import BASE, build, episode

ROOT=Path(__file__).resolve().parents[1]

def test_exact_end_boundary_is_legal():
    result=build(episodes=[episode(hours=30)],events=[])
    prediction=__import__("datetime").datetime.fromisoformat(result.structural_index[0]["prediction_time"].replace("Z","+00:00"))
    outtime=__import__("datetime").datetime.fromisoformat(result.retained_cohort[0]["outtime"].replace("Z","+00:00"))
    assert len(result.structural_index)==1 and result.structural_index[0]["grid_index"]==0
    assert prediction==BASE+timedelta(hours=24)==outtime-timedelta(hours=6)

def test_first_cutoff_and_six_hour_cadence():
    rows=build(episodes=[episode(hours=80)],events=[]).structural_index
    times=[__import__("datetime").datetime.fromisoformat(r["prediction_time"].replace("Z","+00:00")) for r in rows]
    assert times[0]==BASE+timedelta(hours=24)
    assert all(b-a==timedelta(hours=6) for a,b in zip(times,times[1:]))

def test_long_stay_is_capped_at_twelve_without_reindexing():
    rows=build(episodes=[episode(hours=300)],events=[]).structural_index
    assert len(rows)==12 and [r["grid_index"] for r in rows]==list(range(12))

def test_structural_identity_is_unique_and_deterministically_ordered():
    result=build(); rows=result.structural_index
    assert len({(r["stay_id"],r["prediction_time"]) for r in rows})==len(rows)
    assert list(rows)==sorted(rows,key=lambda r:(r["subject_id"],r["stay_id"],r["prediction_time"],r["grid_index"]))

def test_index_is_outcome_free_and_split_neutral():
    rows=build().structural_index; prohibited={"target","label","sofa","remaining_stay","support_state","split","fold","anchor_year_group","outtime"}
    assert all(not set(row)&prohibited for row in rows)

def test_structural_flags_are_preserved_from_vedant_contract():
    row=build().structural_index[0]
    assert {"recovery24_followup_available","recovery48_followup_available","support24_full_followup_available","icu_time_temporally_eligible"}<=set(row)

def test_schema_marks_every_field_non_model_eligible():
    schema=json.loads((ROOT/"configs/synthetic/structural_index_schema_v2.json").read_text())
    assert schema["schema_version"]==INDEX_SCHEMA_VERSION and all(f["model_eligible"] is False for f in schema["fields"])
    assert next(f for f in schema["fields"] if f["name"]=="prediction_time")["role"]=="INDEX_ROUTING"

def test_outtime_is_structural_and_not_model_eligible():
    schema=json.loads((ROOT/"configs/synthetic/retained_cohort_schema_v2.json").read_text()); field=next(f for f in schema["fields"] if f["name"]=="outtime")
    assert field["role"]=="STRUCTURAL_LABEL_ONLY" and field["model_eligible"] is False

def test_no_prediction_formula_is_duplicated_in_phase4_source():
    text=(ROOT/"src/data/synthetic/cohort.py").read_text()
    assert "generate_prediction_timestamps" in text and "24 + 6" not in text and "timedelta(hours=24" not in text
