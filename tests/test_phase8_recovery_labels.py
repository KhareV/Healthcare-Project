from dataclasses import replace
from datetime import datetime, timedelta, timezone
import pytest

from data.synthetic.sofa import COMPONENTS, ComponentResult, SOFAResult
from labels.recovery import RecoveryLabelError, build_recovery_label, validate_recovery_label

T=datetime(2100,1,2,12,tzinfo=timezone.utc)

def score(cutoff,total,stay="A",spec_hash="a"*64):
    parts=[]; remaining=total
    for i,name in enumerate(COMPONENTS):
        value=min(4,remaining); remaining-=value
        parts.append(ComponentResult(name,value,True,(name,), (cutoff.isoformat(),), "sofa-v1"))
    return SOFAResult("S",stay,cutoff.isoformat(),total,tuple(parts),"sofa-v1",spec_hash,"impl","processed","b"*64,"concept","c"*64,"d"*64,"timeline","e"*64,"support","f"*64,"0"*64)

def provider(values):
    calls=[]
    def fn(cutoff): calls.append(cutoff); return score(cutoff,values[cutoff])
    return fn,calls

def test_same_baseline_independent_horizons_and_signs():
    fn,calls=provider({T:10,T+timedelta(hours=24):7,T+timedelta(hours=48):12})
    label=build_recovery_label(stay_id="A",prediction_time=T,outtime=T+timedelta(hours=48),recovery24_followup_available=True,recovery48_followup_available=True,score_at=fn)
    assert (label.delta_sofa_24,label.delta_sofa_48)==(-3.0,2.0)
    assert calls==[T,T+timedelta(hours=24),T+timedelta(hours=48)]
    assert label.future_sofa_48-label.baseline_sofa==2 and label.future_sofa_48-label.future_sofa_24==5
    validate_recovery_label(label)

def test_independent_masks_equality_boundaries_and_neither_case():
    fn,_=provider({T:8,T+timedelta(hours=24):8})
    label=build_recovery_label(stay_id="A",prediction_time=T,outtime=T+timedelta(hours=24),recovery24_followup_available=True,recovery48_followup_available=False,score_at=fn)
    assert label.delta_sofa_24==0 and label.delta_sofa_48 is None
    empty=build_recovery_label(stay_id="A",prediction_time=T,outtime=T+timedelta(hours=23),recovery24_followup_available=False,recovery48_followup_available=False,score_at=lambda _:pytest.fail("SOFA must not be called"))
    assert empty.delta_sofa_24 is empty.delta_sofa_48 is None

def test_flag_attacks_sofa_identity_and_nullness_fail_closed():
    with pytest.raises(RecoveryLabelError,match="flags"):
        build_recovery_label(stay_id="A",prediction_time=T,outtime=T+timedelta(hours=48),recovery24_followup_available=False,recovery48_followup_available=True,score_at=lambda x:score(x,1))
    with pytest.raises(RecoveryLabelError,match="specification"):
        values={T:score(T,1,spec_hash="a"*64),T+timedelta(hours=24):score(T+timedelta(hours=24),2,spec_hash="b"*64)}
        build_recovery_label(stay_id="A",prediction_time=T,outtime=T+timedelta(hours=24),recovery24_followup_available=True,recovery48_followup_available=False,score_at=values.__getitem__)
    fn,_=provider({T:10,T+timedelta(hours=24):7})
    label=build_recovery_label(stay_id="A",prediction_time=T,outtime=T+timedelta(hours=24),recovery24_followup_available=True,recovery48_followup_available=False,score_at=fn)
    with pytest.raises(RecoveryLabelError): validate_recovery_label(replace(label,delta_sofa_24=None))

def test_raw_unclipped_delta_and_component_completeness_do_not_change_eligibility():
    fn,_=provider({T:24,T+timedelta(hours=24):0})
    label=build_recovery_label(stay_id="A",prediction_time=T,outtime=T+timedelta(hours=24),recovery24_followup_available=True,recovery48_followup_available=False,score_at=fn)
    assert label.delta_sofa_24==-24
    assert label.recovery24_eligible and len(label.component_observed_baseline)==6
