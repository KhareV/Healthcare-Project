from copy import deepcopy
from datetime import timedelta

from data.synthetic.sofa import sofa_at
from data.synthetic.sofa_provider import VentilationInterval
from synthetic_sofa_helpers import BINDINGS, CUTOFF, SPEC, STAY, event, normal_history, provider


def run(rows, support=None, cutoff=CUTOFF, stay=STAY):
    return sofa_at(rows, stay, cutoff, spec=SPEC, bindings=BINDINGS, support_provider=support or provider())


def test_window_is_left_open_and_right_closed():
    left = CUTOFF - timedelta(hours=24)
    rows = [
        event("platelet_count", 1, when=left, event_id="left"),
        event("platelet_count", 30, when=left + timedelta(microseconds=1), event_id="inside"),
        event("platelet_count", 120, when=CUTOFF, event_id="cutoff"),
    ]
    result = run(rows)
    coag = result.components[1]
    assert (coag.score, coag.evidence_refs) == (3, ("inside",))


def test_future_high_severity_attack_is_ignored():
    baseline = run(normal_history())
    attacked = normal_history() + [event("platelet_count", 1, when=CUTOFF + timedelta(microseconds=1), event_id="future")]
    assert run(attacked) == baseline


def test_old_high_severity_attack_is_ignored():
    baseline = run(normal_history())
    attacked = normal_history() + [event("bilirubin_total", 50, when=CUTOFF - timedelta(hours=24, microseconds=1), event_id="old")]
    assert run(attacked) == baseline


def test_other_stay_and_subject_attack_is_ignored():
    baseline = run(normal_history())
    attacked = normal_history() + [event("creatinine", 20, stay_id="stay-B", subject_id="subject-B", event_id="other")]
    assert run(attacked) == baseline


def test_same_function_accepts_t_plus_24_and_t_plus_48_without_horizon_branch():
    rows = normal_history(CUTOFF) + normal_history(CUTOFF + timedelta(hours=24)) + normal_history(CUTOFF + timedelta(hours=48))
    first = run(rows, cutoff=CUTOFF)
    second = run(rows, cutoff=CUTOFF + timedelta(hours=24))
    third = run(rows, cutoff=CUTOFF + timedelta(hours=48))
    assert (first.total_score, second.total_score, third.total_score) == (0, 0, 0)


def test_future_support_interval_attack_is_ignored():
    baseline = run(normal_history())
    future = VentilationInterval("stay-A", CUTOFF + timedelta(hours=1), CUTOFF + timedelta(hours=2), "future-vent")
    assert run(normal_history(), provider(extra_vent=(future,))) == baseline


def test_same_timestamp_ties_have_deterministic_sorted_evidence():
    rows = [event("creatinine", 5.5, event_id="z"), event("creatinine", 6, event_id="a")]
    assert run(rows).components[5].evidence_refs == ("a", "z")
