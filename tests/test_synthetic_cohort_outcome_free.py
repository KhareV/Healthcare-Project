import json
from pathlib import Path
from data.synthetic.cohort_manifest import _load_source
from data.synthetic.cohort import build_cohort

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/"artifacts/data/synthetic/smoke/phase3_smoke_v1/synthetic_dataset_manifest_v1.json"

def smoke_build():
    source,runtime,tables=_load_source(SOURCE,ROOT)
    return build_cohort(tables["subjects"],tables["episodes"],tables["raw_events"],minimum_age=18,allowed_cardiac_groups=runtime.values["population"]["condition_groups"])

def test_smoke_cohort_excludes_only_no_cutoff_episode():
    result=smoke_build(); assert result.generated_subject_count==24 and len(result.retained_cohort)==23
    assert result.exclusion_counts=={"NO_LEGAL_PREDICTION_CUTOFF":1}

def test_smoke_structural_counts_match_timestamp_contract():
    result=smoke_build(); counts={row["stay_id"]:0 for row in result.retained_cohort}
    for row in result.structural_index: counts[row["stay_id"]]+=1
    assert len(result.structural_index)==136 and min(counts.values())==1 and max(counts.values())==12

def test_structural_index_contains_only_schema_fields():
    result=smoke_build(); schema=json.loads((ROOT/"configs/synthetic/structural_index_schema_v2.json").read_text()); allowed={f["name"] for f in schema["fields"]}
    assert all(set(row)==allowed for row in result.structural_index)

def test_phase4_source_has_no_model_metric_or_preprocessing_imports():
    text="\n".join((ROOT/"src/data/synthetic"/name).read_text() for name in ("cohort.py","cohort_manifest.py"))
    for token in ("import torch","import xgboost","import sklearn","from models","from evaluation","StandardScaler","SimpleImputer","AUROC","MAE"):
        assert token not in text

def test_no_later_phase_artifact_semantics_are_emitted():
    keys=set().union(*(set(row) for row in smoke_build().structural_index)); joined=" ".join(keys).lower()
    for token in ("feature","sofa","target","label","split","support_state","remaining"):
        assert token not in joined
