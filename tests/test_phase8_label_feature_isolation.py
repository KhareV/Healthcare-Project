from dataclasses import replace
from datetime import timedelta
from features.synthetic import FeatureBuildContext
from labels.icu_time import remaining_episode_time_label
from synthetic_feature_helpers import T,builder,context,event

def test_future_event_does_not_change_phase7_features():
    b=builder(); base=b.build(context([event("heart_rate",70,T-timedelta(hours=1),"past")]))
    future=event("heart_rate",250,T+timedelta(hours=12),"future")
    assert b.build(context([event("heart_rate",70,T-timedelta(hours=1),"past"),future]))==base

def test_outtime_changes_label_but_not_fixed_legal_phase7_input():
    b=builder(); c=context([event("heart_rate",70,T-timedelta(hours=1),"past")])
    first=b.build(c); second=b.build(replace(c,outtime=c.outtime+timedelta(hours=24)))
    assert first==second
    a=remaining_episode_time_label(prediction_time=T,outtime=c.outtime,icu_time_temporally_eligible=True)
    z=remaining_episode_time_label(prediction_time=T,outtime=c.outtime+timedelta(hours=24),icu_time_temporally_eligible=True)
    assert z.remaining_hours-a.remaining_hours==24 and z.log1p_remaining_hours!=a.log1p_remaining_hours

def test_model_facing_fields_contain_no_phase8_targets_or_outtime():
    item=builder().build(context())
    text=repr(item)
    for forbidden in ("remaining_hours","delta_sofa_24","future_sofa","outtime"):
        assert forbidden not in text
