from datetime import timedelta
import pytest

from data.synthetic.sofa import COMPONENTS, sofa_at
from synthetic_sofa_helpers import BINDINGS, CUTOFF, SPEC, STAY, event, normal_history, provider, urine, vaso


def score(rows, support=None, cutoff=CUTOFF):
    return sofa_at(rows, STAY, cutoff, spec=SPEC, bindings=BINDINGS, support_provider=support or provider())


def component(result, name):
    return next(item for item in result.components if item.component_name == name)


def test_spec_and_all_observed_normal_golden_case():
    result = score(normal_history())
    assert tuple(item.component_name for item in result.components) == COMPONENTS
    assert result.component_scores == (0, 0, 0, 0, 0, 0)
    assert result.component_observed == (True,) * 6
    assert result.total_score == 0


def test_all_six_component_abnormal_golden_case():
    rows = [
        event("pao2", 75, event_id="p"), event("fio2", 1.0, event_id="f"),
        event("platelet_count", 15), event("bilirubin_total", 13),
        event("mean_arterial_pressure", 60), event("glasgow_coma_scale", 5),
        event("creatinine", 5.1),
    ]
    result = score(rows, provider(ventilated=True, vaso=(vaso("norepinephrine", .2),)))
    assert result.component_scores == (4, 4, 4, 4, 4, 4)
    assert result.total_score == 24


@pytest.mark.parametrize("value,expected", [(400,0),(399.999,1),(300,1),(299.999,2),(200,2),(199.999,2),(100,2),(99.999,2)])
def test_respiratory_boundaries_without_invasive_support(value, expected):
    result = score([event("pao2", value, event_id="p"), event("fio2", 1, event_id="f")])
    assert component(result, "respiratory").score == expected


@pytest.mark.parametrize("value,expected", [(200,2),(199.999,3),(100,3),(99.999,4)])
def test_respiratory_boundaries_with_invasive_support(value, expected):
    result = score([event("pao2", value, event_id="p"), event("fio2", 1, event_id="f")], provider(ventilated=True))
    assert component(result, "respiratory").score == expected


@pytest.mark.parametrize("value,expected", [(150,0),(149.999,1),(100,1),(99.999,2),(50,2),(49.999,3),(20,3),(19.999,4)])
def test_coagulation_boundaries(value, expected):
    assert component(score([event("platelet_count", value)]), "coagulation").score == expected


@pytest.mark.parametrize("value,expected", [(1.199,0),(1.2,1),(1.999,1),(2,2),(5.999,2),(6,3),(12,3),(12.001,4)])
def test_liver_boundaries(value, expected):
    assert component(score([event("bilirubin_total", value)]), "liver").score == expected


@pytest.mark.parametrize("value,expected", [(15,0),(14,1),(13,1),(12,2),(10,2),(9,3),(6,3),(5,4),(3,4)])
def test_cns_boundaries(value, expected):
    assert component(score([event("glasgow_coma_scale", value)]), "central_nervous_system").score == expected


@pytest.mark.parametrize("value,expected", [(1.199,0),(1.2,1),(2,2),(3.5,3),(5,3),(5.001,4)])
def test_creatinine_boundaries(value, expected):
    assert component(score([event("creatinine", value)]), "renal").score == expected


@pytest.mark.parametrize("agent,rate,expected", [("dobutamine",.01,2),("dopamine",5,2),("dopamine",5.001,3),("dopamine",15,3),("dopamine",15.001,4),("epinephrine",.1,3),("epinephrine",.101,4),("norepinephrine",.1,3),("norepinephrine",.101,4)])
def test_vasoactive_boundaries(agent, rate, expected):
    result = score([event("mean_arterial_pressure", 80)], provider(vaso=(vaso(agent, rate),)))
    assert component(result, "cardiovascular").score == expected


def test_map_boundary_and_worst_value_selection():
    rows = [event("mean_arterial_pressure", 70, event_id="normal"), event("mean_arterial_pressure", 69.9, event_id="low")]
    result = score(rows)
    assert component(result, "cardiovascular").score == 1
    assert component(result, "cardiovascular").evidence_refs == ("low",)


def test_complete_urine_output_and_creatinine_use_worse_score():
    start = CUTOFF - timedelta(hours=24)
    rows = [event("creatinine", 1.2)]
    for index in range(4):
        left = start + timedelta(hours=6 * index)
        rows.append(urine(40, left, left + timedelta(hours=6), "u" + str(index)))
    renal = component(score(rows), "renal")
    assert renal.score == 4
    assert renal.observed is True


def test_incomplete_urine_is_not_treated_as_24_hour_total():
    start = CUTOFF - timedelta(hours=23)
    renal = component(score([urine(100, start, CUTOFF, "partial")]), "renal")
    assert (renal.score, renal.observed) == (0, False)
