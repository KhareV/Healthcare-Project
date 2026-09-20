from dataclasses import replace

import pytest

from serving.history import (
    HistoryContractError,
    IllegalCutoffError,
    InMemorySyntheticHistoryProvider,
    SYNTHETIC_TIMELINE_SCOPE,
    StoredStayTimeline,
    SyntheticPointEventHistoryTruncator,
    TimelineContract,
    UnknownStayError,
)
from phase7_helpers import (
    LATER_CUTOFF,
    MAIN_CUTOFF,
    MAIN_STAY,
    load_timelines,
    parse_time,
    truncate,
)


def test_exact_cutoff_retained_and_t_plus_epsilon_excluded():
    result = truncate()
    times = [row["synthetic_available_time"] for row in result.events]
    assert MAIN_CUTOFF in times
    assert "2026-01-04T12:00:01+00:00" not in times
    assert result.trace.max_retained_event_time == MAIN_CUTOFF


def test_illegal_cutoff_reuses_frozen_timestamp_grid():
    with pytest.raises(IllegalCutoffError, match="legal prediction grid"):
        truncate(cutoff="2026-01-04T12:00:01+00:00")


def test_later_cutoff_independently_includes_newly_available_rows():
    earlier = truncate()
    later = truncate(cutoff=LATER_CUTOFF)
    assert all(parse_time(row["synthetic_available_time"]) <= parse_time(MAIN_CUTOFF) for row in earlier.events)
    assert any(MAIN_CUTOFF < row["synthetic_available_time"] <= LATER_CUTOFF for row in later.events)


def test_unknown_stay_is_structured_internal_failure():
    provider = InMemorySyntheticHistoryProvider(load_timelines())
    with pytest.raises(UnknownStayError, match="unknown stay"):
        provider.get_stay("ABSENT_SYNTHETIC_STAY")


def test_stay_isolation_with_overlapping_timestamps():
    provider = InMemorySyntheticHistoryProvider(load_timelines())
    timeline = provider.get_stay(MAIN_STAY)
    assert all(row["synthetic_stay_key"] == MAIN_STAY for row in timeline.events)
    assert all(row["synthetic_value"] != -999999.0 for row in timeline.events)


def test_provider_rejects_mixed_stay_rows():
    timeline = load_timelines()[0]
    bad = dict(timeline.events[0])
    bad["synthetic_stay_key"] = "OTHER"
    changed = StoredStayTimeline.create(
        subject_id=timeline.subject_id,
        stay_id=timeline.stay_id,
        intime=timeline.intime,
        outtime=timeline.outtime,
        events=(bad,) + timeline.events[1:],
        contract=timeline.contract,
        source_version=timeline.source_version,
        source_sha256=timeline.source_sha256,
    )
    with pytest.raises(HistoryContractError, match="another stay"):
        InMemorySyntheticHistoryProvider((changed,))


def test_truncation_does_not_mutate_source_timeline():
    timeline = load_timelines()[0]
    before = tuple(dict(row) for row in timeline.events)
    truncate(timeline)
    assert tuple(dict(row) for row in timeline.events) == before
    with pytest.raises(TypeError):
        timeline.events[0]["synthetic_value"] = 0


def test_real_and_stateful_contracts_fail_closed():
    timeline = load_timelines()[0]
    real = replace(timeline.contract, scope="real")
    with pytest.raises(HistoryContractError, match="production"):
        SyntheticPointEventHistoryTruncator().truncate(replace(timeline, contract=real), MAIN_CUTOFF)
    stateful = replace(timeline.contract, stateful_intervals_present=True)
    with pytest.raises(HistoryContractError, match="stateful"):
        SyntheticPointEventHistoryTruncator().truncate(replace(timeline, contract=stateful), MAIN_CUTOFF)


def test_event_time_field_is_explicit_not_inferred():
    timeline = load_timelines()[0]
    wrong = replace(timeline.contract, event_time_field="guessed_timestamp")
    with pytest.raises(HistoryContractError, match="guessed_timestamp"):
        SyntheticPointEventHistoryTruncator().truncate(replace(timeline, contract=wrong), MAIN_CUTOFF)


def test_history_module_contains_no_feature_or_prediction_implementation():
    from pathlib import Path

    source = (Path(__file__).resolve().parents[1] / "src/serving/history.py").read_text()
    assert "def build_features" not in source
    assert "def predict" not in source
    assert "drop_duplicates" not in source
