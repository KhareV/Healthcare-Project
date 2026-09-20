import copy
import pytest
from data.synthetic.cohort import SyntheticCohortError
from synthetic_cohort_helpers import BASE, build, episode, event, iso, subject

def test_valid_adult_cardiac_episode_retained():
    result=build(); assert len(result.retained_cohort)==1 and len(result.structural_index)>0

def test_underage_subject_excluded():
    result=build(subjects=[subject(age=17)]); assert not result.retained_cohort and result.exclusion_counts=={"NOT_ADULT":1}

def test_outside_cardiac_domain_excluded():
    result=build(subjects=[subject(group="NON_CARDIAC")]); assert result.exclusion_counts=={"OUTSIDE_CARDIAC_SCOPE":1}

def test_missing_subject_id_rejected_with_reason():
    row=subject(); row["subject_id"]=None; result=build(subjects=[row],episodes=[],events=[]); assert result.exclusion_counts=={"INVALID_SUBJECT_ID":1}

def test_missing_stay_id_rejected_with_reason():
    row=episode(); row["stay_id"]=None; result=build(episodes=[row],events=[]); assert result.exclusion_counts=={"INVALID_STAY_ID":1}

@pytest.mark.parametrize("field,reason",[("intime","INVALID_INTIME"),("outtime","INVALID_OUTTIME")])
def test_missing_episode_boundary_rejected(field,reason):
    row=episode(); row[field]=None; result=build(episodes=[row],events=[]); assert result.exclusion_counts=={reason:1}

@pytest.mark.parametrize("hours",[0,-1])
def test_nonpositive_episode_duration_rejected(hours):
    result=build(episodes=[episode(hours=hours)],events=[]); assert result.exclusion_counts=={"NONPOSITIVE_EPISODE_DURATION":1}

def test_no_legal_cutoff_excluded_using_vedant_output():
    result=build(episodes=[episode(hours=29)],events=[]); assert result.exclusion_counts=={"NO_LEGAL_PREDICTION_CUTOFF":1}

def test_duplicate_stay_rejected():
    with pytest.raises(SyntheticCohortError,match="duplicate stay_id"): build(episodes=[episode(),episode()])

def test_cross_subject_stay_collision_rejected():
    with pytest.raises(SyntheticCohortError,match="cross-subject"): build(subjects=[subject("S1"),subject("S2")],episodes=[episode("S1","E1"),episode("S2","E1")],events=[])

def test_multiple_episode_contract_violation_rejected():
    with pytest.raises(SyntheticCohortError,match="multiple episodes"): build(episodes=[episode(stay_id="E1"),episode(stay_id="E2")],events=[])

def test_event_after_episode_is_upstream_defect():
    with pytest.raises(SyntheticCohortError,match="UPSTREAM PHASE-3 DEFECT"): build(events=[event(hours=61)])

def test_event_before_episode_is_upstream_defect():
    with pytest.raises(SyntheticCohortError,match="UPSTREAM PHASE-3 DEFECT"): build(events=[event(hours=-1)])

def test_event_density_does_not_change_cohort_or_index():
    dense=build(events=[event()]); sparse=build(events=[])
    assert dense.retained_cohort==sparse.retained_cohort and dense.structural_index==sparse.structural_index

def test_future_clinical_value_does_not_change_cohort_or_index():
    first=event(hours=40,value=50); second=copy.deepcopy(first); second["value_numeric"]=200
    assert build(events=[first]).structural_index==build(events=[second]).structural_index
