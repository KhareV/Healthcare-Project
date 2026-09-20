import json
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from labels.event_dict import EventDictionaryError, load_event_dictionary
from labels.support_state import (
    ExecutionMode,
    NormalizedActiveInterval,
    SupportState,
    SupportStateError,
    query_vasopressor_state,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/event_dict_v1.yaml"
FIXTURE = ROOT / "tests/fixtures/support_state/state_cases_v1.json"
CUTOFF = datetime(2026, 1, 2, 12, tzinfo=timezone.utc)
STAY = "SYNTHETIC_STAY_A"


@pytest.fixture(scope="module")
def event_dictionary():
    return load_event_dictionary(CONFIG)


def interval(
    ref,
    *,
    agent="QUALIFYING_AGENT_A",
    start=None,
    end=None,
    stay_id=STAY,
):
    return NormalizedActiveInterval(
        stay_id=stay_id,
        agent_key=agent,
        interval_start=start or CUTOFF - timedelta(hours=1),
        interval_end=end or CUTOFF + timedelta(hours=1),
        source_event_ref=ref,
        normalization_provenance_version="SYNTHETIC_NORMALIZER_V1_NOT_MIMIC",
    )


def query(events, event_dictionary, **overrides):
    arguments = {
        "stay_id": STAY,
        "cutoff": CUTOFF,
        "event_dictionary": event_dictionary,
        "execution_mode": ExecutionMode.SYNTHETIC,
    }
    arguments.update(overrides)
    return query_vasopressor_state(events, **arguments)


def test_no_qualifying_interval_is_off_and_at_risk(event_dictionary):
    result = query([], event_dictionary)
    assert result.support_state is SupportState.OFF
    assert result.vaso_at_risk is True
    assert result.qualifying_interval_refs == ()


def test_one_qualifying_active_interval_is_on_and_not_at_risk(event_dictionary):
    result = query([interval("active")], event_dictionary)
    assert result.support_state is SupportState.ON
    assert result.vaso_at_risk is False
    assert result.qualifying_interval_refs == ("active",)
    assert result.event_dictionary_sha256 == event_dictionary.config_sha256
    assert result.mapping_provenance_version.endswith("NOT_MIMIC")


def test_past_and_future_qualifying_intervals_are_off(event_dictionary):
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
    assert query([past], event_dictionary).support_state is SupportState.OFF
    assert query([future], event_dictionary).support_state is SupportState.OFF


def test_nonqualifying_active_interval_is_ignored(event_dictionary):
    events = [interval("nonqual", agent="NONQUALIFYING_AGENT_X")]
    result = query(events, event_dictionary)
    assert result.support_state is SupportState.OFF
    assert result.vaso_at_risk


def test_multiple_and_overlapping_qualifying_intervals_collapse_to_on(event_dictionary):
    events = [
        interval("z", agent="QUALIFYING_AGENT_A"),
        interval(
            "a",
            agent="QUALIFYING_AGENT_B",
            start=CUTOFF - timedelta(hours=2),
            end=CUTOFF + timedelta(hours=3),
        ),
    ]
    result = query(events, event_dictionary)
    assert result.support_state is SupportState.ON
    assert not result.vaso_at_risk
    assert result.qualifying_interval_refs == ("a", "z")


def test_future_event_perturbation_cannot_change_baseline_state(event_dictionary):
    baseline = [interval("active")]
    future = interval(
        "future",
        start=CUTOFF + timedelta(hours=2),
        end=CUTOFF + timedelta(hours=4),
    )
    assert query(baseline, event_dictionary) == query(
        baseline + [future], event_dictionary
    )


def test_event_order_invariance(event_dictionary):
    events = [
        interval("b", agent="QUALIFYING_AGENT_B"),
        interval("a", agent="QUALIFYING_AGENT_A"),
        interval("x", agent="NONQUALIFYING_AGENT_X"),
    ]
    assert query(events, event_dictionary) == query(list(reversed(events)), event_dictionary)


def test_other_stay_isolation(event_dictionary):
    other = interval("other", stay_id="SYNTHETIC_STAY_B")
    assert query([other], event_dictionary).support_state is SupportState.OFF


def test_many_nonqualifying_intervals_cannot_attack_state(event_dictionary):
    events = [
        interval(
            "nonqual-{}".format(index),
            agent="NONQUALIFYING_AGENT_X_{}".format(index),
        )
        for index in range(50)
    ]
    assert query(events, event_dictionary).support_state is SupportState.OFF


def test_checked_in_draft_refuses_scientific_query(event_dictionary):
    with pytest.raises(EventDictionaryError, match="MIMIC SCHEMA / MAPPING"):
        query(
            [interval("active")],
            event_dictionary,
            execution_mode=ExecutionMode.SCIENTIFIC,
        )


def test_synthetic_closed_open_boundary_is_explicit_not_scientific(event_dictionary):
    at_start = interval("at-start", start=CUTOFF, end=CUTOFF + timedelta(hours=1))
    at_end = interval("at-end", start=CUTOFF - timedelta(hours=1), end=CUTOFF)
    between = interval(
        "between", start=CUTOFF - timedelta(hours=1), end=CUTOFF + timedelta(hours=1)
    )
    assert query([at_start], event_dictionary).support_state is SupportState.ON
    assert query([at_end], event_dictionary).support_state is SupportState.OFF
    result = query([between], event_dictionary)
    assert result.support_state is SupportState.ON
    assert result.interval_boundary_policy == "CLOSED_OPEN"
    assert result.execution_mode is ExecutionMode.SYNTHETIC


def test_fixture_contract_loads_and_queries_deterministically(event_dictionary):
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert payload["artifact_kind"].endswith("NOT_MIMIC")
    events = [NormalizedActiveInterval.from_mapping(item) for item in payload["intervals"]]
    result = query(events, event_dictionary)
    assert result.support_state is SupportState.ON
    assert result.qualifying_interval_refs == ("SYNTHETIC_EVENT_ACTIVE_A",)


@pytest.mark.parametrize(
    "field",
    ["interval_start", "interval_end"],
)
def test_missing_or_malformed_timestamp_fails(field):
    raw = {
        "stay_id": STAY,
        "agent_key": "QUALIFYING_AGENT_A",
        "interval_start": "2026-01-02T11:00:00Z",
        "interval_end": "2026-01-02T13:00:00Z",
        "source_event_ref": "event",
        "normalization_provenance_version": "SYNTHETIC_NORMALIZER_V1_NOT_MIMIC",
    }
    raw[field] = "not-a-timestamp"
    with pytest.raises(SupportStateError, match="malformed"):
        NormalizedActiveInterval.from_mapping(raw)
    raw.pop(field)
    with pytest.raises(SupportStateError, match="ISO-8601"):
        NormalizedActiveInterval.from_mapping(raw)


@pytest.mark.parametrize("hours", [0, -1])
def test_zero_length_or_end_before_start_fails(event_dictionary, hours):
    invalid = interval("invalid", start=CUTOFF, end=CUTOFF + timedelta(hours=hours))
    with pytest.raises(SupportStateError, match="end must be after start"):
        query([invalid], event_dictionary)


def test_invalid_or_inconsistent_stay_id_fails(event_dictionary):
    with pytest.raises(SupportStateError, match="stay_id"):
        query([interval("invalid", stay_id=True)], event_dictionary)
    with pytest.raises(SupportStateError, match="query stay_id"):
        query([], event_dictionary, stay_id="")


def test_duplicate_source_reference_and_exact_duplicate_fail(event_dictionary):
    first = interval("same")
    conflict = replace(first, interval_end=CUTOFF + timedelta(hours=2))
    with pytest.raises(SupportStateError, match="source_event_ref"):
        query([first, conflict], event_dictionary)

    same_interval_new_ref = replace(first, source_event_ref="different")
    with pytest.raises(SupportStateError, match="exact duplicate"):
        query([first, same_interval_new_ref], event_dictionary)


def test_timezone_awareness_mismatch_fails(event_dictionary):
    naive = interval(
        "naive",
        start=datetime(2026, 1, 2, 11),
        end=datetime(2026, 1, 2, 13),
    )
    with pytest.raises(SupportStateError, match="timezone awareness"):
        query([naive], event_dictionary)


def test_execution_mode_must_be_explicit_enum(event_dictionary):
    with pytest.raises(SupportStateError, match="execution_mode"):
        query([interval("active")], event_dictionary, execution_mode="SYNTHETIC")
