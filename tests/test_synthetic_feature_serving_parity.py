import json
from pathlib import Path
from data.schema import temporal_information_views,unflatten_temporal
from data.synthetic.history_provider import CanonicalTimelineHistoryProvider
from data.synthetic.validation import load_jsonl
from features.synthetic import FeatureBuildContext,SyntheticCanonicalFeatureBuilder,_dt
from serving.history import SyntheticPointEventHistoryTruncator
from serving.preprocessing import CanonicalHistoryInputProvider
from synthetic_feature_helpers import ROOT,SCHEMA

PM=ROOT/"artifacts/data/synthetic/timelines/fixtures/phase5_fixture_v1/synthetic_processed_manifest_v1.json"
STRUCT=ROOT/"artifacts/data/synthetic/cohorts/fixtures/phase4_fixture_v2/structural_index.jsonl"

def setup():
    pm=json.loads(PM.read_text()); arts={a["logical_name"]:ROOT/a["repository_relative_path"] for a in pm["artifacts"]}; statics=load_jsonl(arts["canonical_statics"]); timeline=load_jsonl(arts["canonical_timeline"]); structural=load_jsonl(STRUCT); by={s["stay_id"]:s for s in statics}; b=SyntheticCanonicalFeatureBuilder(schema_path=SCHEMA,root=ROOT,statics_by_stay=by); return b,by,timeline,structural

def test_direct_full_history_equals_pulkit_truncation_and_same_builder():
    b,statics,timeline,structural=setup(); row=structural[0]; stay=statics[row["stay_id"]]; cutoff=row["prediction_time"].replace("Z","+00:00")
    direct=b.build(FeatureBuildContext(row["subject_id"],row["stay_id"],_dt(stay["intime"]),_dt(stay["outtime"]),tuple(timeline),_dt(cutoff),cutoff,row["grid_index"],row["icu_elapsed_hours"]))
    provider=CanonicalHistoryInputProvider(history_provider=CanonicalTimelineHistoryProvider(PM,ROOT),history_truncator=SyntheticPointEventHistoryTruncator(),build_features=b,feature_schema=b.feature_schema)
    serving=provider.get_canonical_input(stay_id=row["stay_id"],prediction_time=cutoff,task="recovery",family="gru",feature_version=b.feature_schema.version)
    assert serving==temporal_information_views(direct)["gru"]
    quality=provider.data_quality(stay_id=row["stay_id"],prediction_time=cutoff); assert "confidence" not in quality

def test_xgb_is_only_flattened_gru_information_for_label_free_input():
    b,statics,timeline,structural=setup(); row=structural[-1]; stay=statics[row["stay_id"]]; context=FeatureBuildContext(row["subject_id"],row["stay_id"],_dt(stay["intime"]),_dt(stay["outtime"]),tuple(timeline),_dt(row["prediction_time"]),row["prediction_time"],row["grid_index"],row["icu_elapsed_hours"]); item=b.build(context); views=temporal_information_views(item); f=b.feature_schema.feature_dim
    assert unflatten_temporal(views["xgboost"]["history_values"],f)==views["gru"]["history_values"]
    assert unflatten_temporal(views["xgboost"]["observation_mask"],f)==views["gru"]["observation_mask"]
    assert unflatten_temporal(views["xgboost"]["tslo_hours"],f)==views["gru"]["tslo_hours"]
    assert views["xgboost"]["padding_mask"]==views["gru"]["padding_mask"] and views["xgboost"]["static_features"]==views["gru"]["static_features"]
