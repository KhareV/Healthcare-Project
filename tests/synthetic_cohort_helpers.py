from datetime import datetime, timedelta, timezone

from data.synthetic.cohort import build_cohort

BASE=datetime(2100,1,1,tzinfo=timezone.utc)
def iso(value): return value.isoformat(timespec="microseconds").replace("+00:00","Z")
def subject(subject_id="S1",age=50,group="CARDIAC_A"):
    return {"subject_id":subject_id,"age_years":age,"sex_category":"X","cardiac_condition_group":group}
def episode(subject_id="S1",stay_id="E1",hours=60,intime=BASE):
    return {"subject_id":subject_id,"stay_id":stay_id,"intime":iso(intime),"outtime":iso(intime+timedelta(hours=hours)),"episode_sequence":1,"generator_subject_stream_id":"stream"}
def event(subject_id="S1",stay_id="E1",hours=1,value=80):
    return {"event_id":"O1","subject_id":subject_id,"stay_id":stay_id,"event_time":iso(BASE+timedelta(hours=hours)),"interval_start":None,"interval_end":None,"concept_code":"heart_rate","value_numeric":value,"value_text":None,"unit":"beats/min","generator_component":"test"}
def build(subjects=None,episodes=None,events=None):
    return build_cohort(subjects if subjects is not None else [subject()],episodes if episodes is not None else [episode()],events if events is not None else [event()],minimum_age=18,allowed_cardiac_groups=["CARDIAC_A"])
