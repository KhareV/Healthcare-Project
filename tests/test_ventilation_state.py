import inspect
import json
from dataclasses import fields, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from data.collate import CanonicalBatch
from labels.event_dict import EventDictionaryError, load_event_dictionary
from labels.support_state import ExecutionMode
from labels.ventilation_state import (
    InvasiveVentilationState,
    NormalizedVentilationInterval,
    RespiratoryCategory,
    VentilationStateError,
    VentilationStateResult,
    query_invasive_ventilation_state,
)
import labels.ventilation_state as ventilation_module


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/event_dict_v1.yaml"
FIXTURE = ROOT / "tests/fixtures/ventilation_state/state_transitions_v1.json"
CUTOFF = datetime(2026, 1, 2, 12, tzinfo=timezone.utc)
STAY = "SYNTHETIC_STAY_VENT_A"
CONCEPT_VERSION = "SYNTHETIC_VENT_CONCEPT_V1_NOT_MIMIC"
ADAPTER_VERSION = "SYNTHETIC_VENT_ADAPTER_V1_NOT_MIMIC"


@pytest.fixture(scope="module")
def event_dictionary():
    return load_event_dictionary(CONFIG)


def interval(
    ref,
    *,
    category=RespiratoryCategory.INVASIVE,
    start=None,
    end=None,
    stay_id=STAY,
    concept_version=CONCEPT_VERSION,
    adapter_version=ADAPTER_VERSION,
):
    return NormalizedVentilationInterval(
        stay_id=stay_id,
        category=category,
        interval_start=start or CUTOFF - timedelta(hours=1),
        interval_end=end or CUTOFF + timedelta(hours=1),
        source_state_ref=ref,
        concept_version=concept_version,
        adapter_version=adapter_version,
        normalization_provenance_ref="SYNTHETIC_NORMALIZER_V1_NOT_MIMIC_CODE",
    )


def query(events, event_dictionary, *, cutoff=CUTOFF, **overrides):
    arguments = {
        "stay_id": STAY,
        "cutoff": cutoff,
        "event_dictionary": event_dictionary,
        "execution_mode": ExecutionMode.SYNTHETIC,
    }
    arguments.update(overrides)
    return query_invasive_ventilation_state(events, **arguments)


def test_no_respiratory_support_is_invasive_off_and_at_risk(event_dictionary):
    result = query([], event_dictionary)
    assert result.invasive_state is InvasiveVentilationState.OFF
    assert result.invasive_on is False
    assert result.vent_at_risk is True
    assert result.active_categories == ()


def test_invasive_active_is_on_and_not_at_risk(event_dictionary):
    result = query([interval("invasive")], event_dictionary)
    assert result.invasive_state is InvasiveVentilationState.ON
    assert result.invasive_on is True
    assert result.vent_at_risk is False
    assert result.supporting_invasive_interval_refs == ("invasive",)
    assert result.event_dictionary_sha256 == event_dictionary.config_sha256


@pytest.mark.parametrize(
    "category",
    [
        RespiratoryCategory.NON_INVASIVE,
        RespiratoryCategory.HFNC,
        RespiratoryCategory.ORDINARY_OXYGEN,
    ],
)
def test_non_invasive_categories_remain_invasive_off(category, event_dictionary):
    result = query([interval("non-invasive", category=category)], event_dictionary)
    assert result.invasive_state is InvasiveVentilationState.OFF
    assert result.vent_at_risk is True
    assert result.active_categories == (category,)


def test_past_and_future_invasive_intervals_are_off(event_dictionary):
    past = interval(
        "past",
        start=CUTOFF - timedelta(hours=3),
        end=CUTOFF - timedelta(hours=2),
    )
    future = interval(
        "future",
        start=CUTOFF + timedelta(minutes=1),
        end=CUTOFF + timedelta(hours=2),
    )
    assert query([past], event_dictionary).invasive_on is False
    assert query([future], event_dictionary).invasive_on is False


def test_already_on_continuation_stays_baseline_on(event_dictionary):
    continuing = interval(
        "continuing",
        start=CUTOFF - timedelta(hours=8),
        end=CUTOFF + timedelta(hours=8),
    )
    result = query([continuing], event_dictionary)
    assert result.invasive_on
    assert not result.vent_at_risk


def test_future_event_perturbation_does_not_change_baseline(event_dictionary):
    niv = interval("niv", category=RespiratoryCategory.NON_INVASIVE)
    future = interval(
        "future-invasive",
        start=CUTOFF + timedelta(hours=2),
        end=CUTOFF + timedelta(hours=5),
    )
    assert query([niv], event_dictionary) == query([niv, future], event_dictionary)


def test_other_stay_isolation(event_dictionary):
    other = interval("other-invasive", stay_id="SYNTHETIC_STAY_VENT_B")
    assert query([other], event_dictionary).invasive_on is False


def test_event_order_invariance_and_deterministic_evidence(event_dictionary):
    events = [
        interval("z-invasive"),
        interval("a-invasive", start=CUTOFF - timedelta(hours=2)),
        interval("hfnc", category=RespiratoryCategory.HFNC),
    ]
    forward = query(events, event_dictionary)
    reverse = query(list(reversed(events)), event_dictionary)
    assert forward == reverse
    assert forward.supporting_invasive_interval_refs == (
        "a-invasive",
        "z-invasive",
    )
    assert forward.active_categories == (
        RespiratoryCategory.HFNC,
        RespiratoryCategory.INVASIVE,
    )


def test_unknown_category_fails_instead_of_becoming_off():
    raw = {
        "stay_id": STAY,
        "category": "UNCLASSIFIABLE_TEST_STATE",
        "interval_start": "2026-01-02T11:00:00Z",
        "interval_end": "2026-01-02T13:00:00Z",
        "source_state_ref": "unknown",
        "concept_version": CONCEPT_VERSION,
        "adapter_version": ADAPTER_VERSION,
        "normalization_provenance_ref": "SYNTHETIC_NORMALIZER_V1_NOT_MIMIC_CODE",
    }
    with pytest.raises(VentilationStateError, match="refusing to map it to OFF"):
        NormalizedVentilationInterval.from_mapping(raw)


def test_direct_unknown_category_attack_fails(event_dictionary):
    attacked = replace(interval("bad-category"), category="UNKNOWN")
    with pytest.raises(VentilationStateError, match="refusing to map it to OFF"):
        query([attacked], event_dictionary)


def test_concept_and_adapter_version_mismatch_fail(event_dictionary):
    with pytest.raises(VentilationStateError, match="concept version mismatch"):
        query(
            [interval("wrong-concept", concept_version="DIFFERENT_CONCEPT")],
            event_dictionary,
        )
    with pytest.raises(VentilationStateError, match="adapter version mismatch"):
        query(
            [interval("wrong-adapter", adapter_version="DIFFERENT_ADAPTER")],
            event_dictionary,
        )


def test_checked_in_draft_blocks_scientific_execution(event_dictionary):
    with pytest.raises(EventDictionaryError, match="PINNED|VERIFIED"):
        query(
            [interval("invasive")],
            event_dictionary,
            execution_mode=ExecutionMode.SCIENTIFIC,
        )


def test_synthetic_closed_open_boundaries_are_explicit(event_dictionary):
    at_start = interval("start", start=CUTOFF, end=CUTOFF + timedelta(hours=1))
    at_end = interval("end", start=CUTOFF - timedelta(hours=1), end=CUTOFF)
    assert query([at_start], event_dictionary).invasive_on is True
    result = query([at_end], event_dictionary)
    assert result.invasive_on is False
    assert result.interval_boundary_policy == "CLOSED_OPEN"
    assert result.execution_mode is ExecutionMode.SYNTHETIC


def test_adjacent_transition_fixture_before_during_after(event_dictionary):
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert payload["artifact_kind"].endswith("NOT_MIMIC_CODE")
    events = [NormalizedVentilationInterval.from_mapping(row) for row in payload["intervals"]]
    before = query(
        events,
        event_dictionary,
        cutoff=datetime(2026, 1, 2, 9, tzinfo=timezone.utc),
    )
    during = query(
        events,
        event_dictionary,
        cutoff=datetime(2026, 1, 2, 12, tzinfo=timezone.utc),
    )
    after = query(
        events,
        event_dictionary,
        cutoff=datetime(2026, 1, 2, 15, tzinfo=timezone.utc),
    )
    assert before.active_categories == (RespiratoryCategory.NON_INVASIVE,)
    assert before.invasive_state is InvasiveVentilationState.OFF
    assert during.invasive_state is InvasiveVentilationState.ON
    assert after.active_categories == (RespiratoryCategory.NON_INVASIVE,)
    assert after.invasive_state is InvasiveVentilationState.OFF


@pytest.mark.parametrize("hours", [0, -1])
def test_zero_duration_or_reversed_interval_fails(event_dictionary, hours):
    invalid = interval("invalid", start=CUTOFF, end=CUTOFF + timedelta(hours=hours))
    with pytest.raises(VentilationStateError, match="end must be after start"):
        query([invalid], event_dictionary)


def test_duplicate_and_conflicting_normalized_records_fail(event_dictionary):
    first = interval("same-ref")
    conflict = replace(first, interval_end=CUTOFF + timedelta(hours=2))
    with pytest.raises(VentilationStateError, match="source_state_ref"):
        query([first, conflict], event_dictionary)

    exact_new_ref = replace(first, source_state_ref="different-ref")
    with pytest.raises(VentilationStateError, match="exact duplicate"):
        query([first, exact_new_ref], event_dictionary)


def test_missing_or_malformed_timestamp_fails():
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))["intervals"][0]
    malformed = dict(payload)
    malformed["interval_start"] = "not-a-time"
    with pytest.raises(VentilationStateError, match="malformed"):
        NormalizedVentilationInterval.from_mapping(malformed)
    missing = dict(payload)
    missing.pop("interval_end")
    with pytest.raises(VentilationStateError, match="ISO-8601"):
        NormalizedVentilationInterval.from_mapping(missing)


def test_phase2_result_has_no_phase3_label_or_censor_fields():
    names = {field.name for field in fields(VentilationStateResult)}
    assert not names & {
        "Y_support",
        "organ_support_eligible",
        "censor_reason",
        "future_invasive_onset",
        "support_label",
    }


def test_ventilation_audit_evidence_is_not_added_to_model_batch():
    names = {field.name for field in fields(CanonicalBatch)}
    assert not names & {
        "vent_at_risk",
        "future_invasive_onset",
        "supporting_invasive_interval_refs",
        "censor_reason",
    }


def test_phase2_module_contains_no_prevalence_path():
    source = inspect.getsource(ventilation_module).lower()
    assert "prevalence" not in source
    assert "positive_rate" not in source
    assert "class_balance" not in source

