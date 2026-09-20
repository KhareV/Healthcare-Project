import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from data.timestamps import PredictionTimestamp
from features.synthetic import FeatureBuildContext, SyntheticCanonicalFeatureBuilder

ROOT=Path(__file__).resolve().parents[1]
SCHEMA=ROOT/"configs/synthetic/feature_schema_v1.json"
T=datetime(2100,1,4,12,tzinfo=timezone.utc)
STATIC={"subject_id":"S","stay_id":"A","age_years":64,"sex_category":"FEMALE","cardiac_condition_group":"SYNTHETIC_ISCHEMIC","intime":(T-timedelta(hours=72)).isoformat(),"outtime":(T+timedelta(hours=72)).isoformat(),"provenance_id":"PROV-STATIC-SUBJECT-EPISODE-V1"}

def builder(statics=None):
    return SyntheticCanonicalFeatureBuilder(schema_path=SCHEMA,root=ROOT,statics_by_stay={"A":statics or STATIC})

def event(concept,value,when,ref=None,stay="A",subject="S"):
    mapping=json.loads((ROOT/"configs/synthetic/concept_map_v1.json").read_text())["mappings"][concept]
    return {"event_id":ref or concept+when.isoformat(),"subject_id":subject,"stay_id":stay,"event_time":when.isoformat(),"interval_start":None if mapping["event_kind"]=="point" else (when-timedelta(hours=1)).isoformat(),"interval_end":None if mapping["event_kind"]=="point" else when.isoformat(),"canonical_concept":concept,"value_numeric":value,"unit":mapping["canonical_unit"],"event_kind":mapping["event_kind"],"provenance_id":mapping["provenance_id"],"source_concept":concept,"source_schema_version":"synthetic_raw_schema_v1","normalization_version":"synthetic_concept_map_v1"}

def context(events=(),t=T,intime=None,text=None):
    intime=intime or T-timedelta(hours=72)
    elapsed=int((t-intime).total_seconds()/3600)
    grid=(elapsed-24)//6
    return FeatureBuildContext(subject_id="S",stay_id="A",intime=intime,outtime=T+timedelta(hours=72),events=tuple(events),prediction_time=t,prediction_time_text=text or t.isoformat(),grid_index=grid,icu_elapsed_hours=elapsed)
