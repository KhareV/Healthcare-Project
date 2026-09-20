from datetime import datetime,timedelta
from pathlib import Path
import tempfile,shutil
import pytest
from data.synthetic.history_provider import CanonicalTimelineHistoryProvider
from data.synthetic.processed_manifest import build_processed_bundle
from serving.history import StoredStayTimeline,SyntheticPointEventHistoryTruncator,UnknownStayError
from synthetic_timeline_helpers import COHORT,MAP,ROOT,SCHEMA

@pytest.fixture
def provider_bundle():
    parent=Path(tempfile.mkdtemp(prefix="phase5-history-",dir=ROOT)); out=parent/"bundle"; manifest=build_processed_bundle(COHORT,MAP,SCHEMA,out,ROOT,mode="engineering",generation_command="pytest",generation_timestamp="2026-09-20T17:20:00Z")
    yield CanonicalTimelineHistoryProvider(manifest,ROOT)
    shutil.rmtree(parent)

def test_provider_fetches_full_stay_without_features(provider_bundle):
    stay=provider_bundle.get_stay("SYN-E-00000001"); assert stay.events and all(r["stay_id"]==stay.stay_id for r in stay.events); assert not any(k in stay.events[0] for k in ("bin","mask","tslo","feature_vector"))
def test_provider_unknown_stay_has_no_fallback(provider_bundle):
    with pytest.raises(UnknownStayError): provider_bundle.get_stay("UNKNOWN")
def test_existing_pulkit_truncator_includes_exact_t_and_excludes_future(provider_bundle):
    timeline=provider_bundle.get_stay("SYN-E-00000001"); cutoff=timeline.intime+timedelta(hours=24); base=dict(timeline.events[0]); exact=dict(base,event_id="EXACT",event_time=cutoff.isoformat()); future=dict(base,event_id="FUTURE",event_time=(cutoff+timedelta(microseconds=1)).isoformat()); changed=StoredStayTimeline.create(subject_id=timeline.subject_id,stay_id=timeline.stay_id,intime=timeline.intime,outtime=timeline.outtime,events=timeline.events+(exact,future),contract=timeline.contract,source_version=timeline.source_version,source_sha256=timeline.source_sha256); result=SyntheticPointEventHistoryTruncator().truncate(changed,cutoff.isoformat()); ids={r["event_id"] for r in result.events}; assert "EXACT" in ids and "FUTURE" not in ids
def test_future_perturbation_does_not_change_truncated_history(provider_bundle):
    timeline=provider_bundle.get_stay("SYN-E-00000001"); cutoff=timeline.intime+timedelta(hours=24); trunc=SyntheticPointEventHistoryTruncator(); before=trunc.truncate(timeline,cutoff.isoformat()).events; events=[]
    for row in timeline.events:
        changed=dict(row)
        if datetime.fromisoformat(changed["event_time"].replace("Z","+00:00"))>cutoff: changed["value_numeric"]+=1
        events.append(changed)
    altered=StoredStayTimeline.create(subject_id=timeline.subject_id,stay_id=timeline.stay_id,intime=timeline.intime,outtime=timeline.outtime,events=events,contract=timeline.contract,source_version=timeline.source_version,source_sha256=timeline.source_sha256); assert before==trunc.truncate(altered,cutoff.isoformat()).events
def test_raw_history_keeps_events_older_than_48h(provider_bundle):
    timeline=provider_bundle.get_stay("SYN-E-00000003"); cutoff=timeline.intime+timedelta(hours=78); result=SyntheticPointEventHistoryTruncator().truncate(timeline,cutoff.isoformat()); assert any(datetime.fromisoformat(r["event_time"].replace("Z","+00:00"))<cutoff-timedelta(hours=48) for r in result.events)
def test_stay_isolation(provider_bundle):
    a=provider_bundle.get_stay("SYN-E-00000001"); b=provider_bundle.get_stay("SYN-E-00000002"); assert {r["stay_id"] for r in a.events}=={a.stay_id} and {r["event_id"] for r in a.events}.isdisjoint({r["event_id"] for r in b.events})
def test_phase5_source_contains_no_preprocessing_or_feature_builder():
    text="\n".join((ROOT/"src/data/synthetic"/n).read_text() for n in ("normalization.py","processed_manifest.py","history_provider.py"))
    for token in ("fillna","ffill","bfill","interpolate","StandardScaler","MinMaxScaler","SimpleImputer","def build_features","torch","xgboost"): assert token not in text
