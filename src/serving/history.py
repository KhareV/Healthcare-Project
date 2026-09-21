"""Cutoff-safe raw-history access for serving.

This module deliberately contains no feature aggregation, binning, imputation,
mask, TSLO, static encoding, or tensor construction logic.  The only concrete
truncator currently provided is restricted to an explicitly declared synthetic
point-event contract.  A production implementation remains blocked until the
upstream processed-timeline and stateful-event semantics are frozen.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Mapping, Optional, Protocol, Sequence, Tuple

from data.timestamps import (
    PredictionTimestamp,
    RetainedICUStay,
    TimestampValidationError,
    generate_prediction_rows_for_stay,
)


SYNTHETIC_TIMELINE_SCOPE = "SYNTHETIC_ENGINEERING_FIXTURE_NOT_A_MIMIC_SCHEMA_CLAIM"
SYNTHETIC_POINT_EVENT_SEMANTICS = "SYNTHETIC_POINT_EVENT_AVAILABLE_AT_EVENT_TIME"


class HistoryError(RuntimeError):
    """Base class for structured internal history failures."""


class UnknownStayError(HistoryError):
    """The requested stay is absent from the configured timeline store."""


class IllegalCutoffError(HistoryError):
    """The requested cutoff is not emitted by the frozen timestamp contract."""


class HistoryContractError(HistoryError):
    """The timeline cannot be safely interpreted by this truncator."""


@dataclass(frozen=True)
class TimelineContract:
    """Explicit field mapping; names are never inferred by serving."""

    version: str
    scope: str
    stay_id_field: str
    event_time_field: str
    event_time_semantics: str
    stateful_intervals_present: bool


@dataclass(frozen=True)
class StoredStayTimeline:
    """One retained ICU stay and its immutable retrospective source rows."""

    subject_id: object
    stay_id: object
    intime: datetime
    outtime: datetime
    events: Tuple[Mapping[str, object], ...]
    support_intervals: Tuple[Mapping[str, object], ...]
    contract: TimelineContract
    source_version: str
    source_sha256: str

    @classmethod
    def create(
        cls,
        *,
        subject_id: object,
        stay_id: object,
        intime: datetime,
        outtime: datetime,
        events: Sequence[Mapping[str, object]],
        support_intervals: Sequence[Mapping[str, object]] = (),
        contract: TimelineContract,
        source_version: str,
        source_sha256: str,
    ) -> "StoredStayTimeline":
        # Copy and freeze each row so truncation cannot mutate the backing store.
        frozen = tuple(MappingProxyType(dict(row)) for row in events)
        frozen_support = tuple(MappingProxyType(dict(row)) for row in support_intervals)
        return cls(
            subject_id=subject_id,
            stay_id=stay_id,
            intime=intime,
            outtime=outtime,
            events=frozen, support_intervals=frozen_support,
            contract=contract,
            source_version=source_version,
            source_sha256=source_sha256,
        )


@dataclass(frozen=True)
class HistoryTrace:
    requested_cutoff: str
    source_timeline_version: str
    source_timeline_sha256: str
    source_event_count: int
    retained_event_count: int
    max_retained_event_time: Optional[str]


@dataclass(frozen=True)
class TruncatedStayHistory:
    """Raw rows through a legal cutoff; still not a feature representation."""

    subject_id: object
    stay_id: object
    intime: datetime
    outtime: datetime
    events: Tuple[Mapping[str, object], ...]
    support_intervals: Tuple[Mapping[str, object], ...]
    contract: TimelineContract
    prediction_row: PredictionTimestamp
    trace: HistoryTrace


class HistoryProvider(Protocol):
    def get_stay(self, stay_id: object) -> StoredStayTimeline:
        """Return exactly one retained-stay timeline or raise UnknownStayError."""


class HistoryTruncator(Protocol):
    def truncate(
        self, timeline: StoredStayTimeline, prediction_time: str
    ) -> TruncatedStayHistory:
        """Restrict raw information before canonical feature construction."""


class InMemorySyntheticHistoryProvider:
    """Synthetic fixture adapter; not a production storage-format decision."""

    def __init__(self, timelines: Sequence[StoredStayTimeline]) -> None:
        indexed = {}
        for timeline in timelines:
            if timeline.contract.scope != SYNTHETIC_TIMELINE_SCOPE:
                raise HistoryContractError("synthetic provider rejects non-synthetic timeline")
            key = (type(timeline.stay_id), timeline.stay_id)
            if key in indexed:
                raise HistoryContractError("duplicate stay in timeline provider")
            field = timeline.contract.stay_id_field
            if any(row.get(field) != timeline.stay_id for row in timeline.events):
                raise HistoryContractError("timeline contains an event from another stay")
            indexed[key] = timeline
        self._timelines = MappingProxyType(indexed)

    def get_stay(self, stay_id: object) -> StoredStayTimeline:
        timeline = self._timelines.get((type(stay_id), stay_id))
        if timeline is None:
            raise UnknownStayError("unknown stay")
        return timeline


def _instant(value: datetime) -> datetime:
    return value.astimezone(timezone.utc) if value.utcoffset() is not None else value


def _parse_datetime(value: object, field: str) -> datetime:
    if not isinstance(value, str):
        raise HistoryContractError(field + " must be an ISO-8601 string")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise HistoryContractError(field + " must be an ISO-8601 string") from error
    return parsed


class SyntheticPointEventHistoryTruncator:
    """Filter verified synthetic point events at ``event_time <= t``.

    This implementation intentionally refuses production and interval-state
    timelines.  Those require the still-missing upstream contract.
    """

    def truncate(
        self, timeline: StoredStayTimeline, prediction_time: str
    ) -> TruncatedStayHistory:
        contract = timeline.contract
        if contract.scope != SYNTHETIC_TIMELINE_SCOPE:
            raise HistoryContractError(
                "production raw-history truncation contract is unresolved"
            )
        if (
            contract.event_time_semantics != SYNTHETIC_POINT_EVENT_SEMANTICS
            or contract.stateful_intervals_present
        ):
            raise HistoryContractError(
                "stateful or unknown event-time semantics require upstream verification"
            )
        requested = _parse_datetime(prediction_time, "prediction_time")
        stay = RetainedICUStay(
            subject_id=timeline.subject_id,
            stay_id=timeline.stay_id,
            intime=timeline.intime,
            outtime=timeline.outtime,
        )
        try:
            legal_rows = generate_prediction_rows_for_stay(stay)
        except TimestampValidationError as error:
            raise HistoryContractError("invalid retained-stay timestamp contract") from error
        prediction_row = next(
            (
                row
                for row in legal_rows
                if _instant(row.prediction_time) == _instant(requested)
                and (row.prediction_time.utcoffset() is None)
                == (requested.utcoffset() is None)
            ),
            None,
        )
        if prediction_row is None:
            raise IllegalCutoffError("cutoff is not on the legal prediction grid")

        retained = []
        retained_times = []
        for row in timeline.events:
            event_time = _parse_datetime(
                row.get(contract.event_time_field), contract.event_time_field
            )
            if (event_time.utcoffset() is None) != (requested.utcoffset() is None):
                raise HistoryContractError("event and cutoff timezone awareness differ")
            if _instant(event_time) <= _instant(requested):
                retained.append(row)
                retained_times.append(event_time)
        # Support intervals are a separate frozen semantic source.  Only
        # intervals whose onset is known by the cutoff enter the builder.
        # Their cessation bound is used internally solely for state-as-of
        # queries and is never emitted as a predictor.
        support_intervals = []
        for row in timeline.support_intervals:
            interval_start = _parse_datetime(row.get("interval_start"), "interval_start")
            if (interval_start.utcoffset() is None) != (requested.utcoffset() is None):
                raise HistoryContractError("support interval and cutoff timezone awareness differ")
            if _instant(interval_start) <= _instant(requested):
                support_intervals.append(row)
        max_time = max(retained_times, key=_instant) if retained_times else None
        trace = HistoryTrace(
            requested_cutoff=prediction_row.prediction_time.isoformat(),
            source_timeline_version=timeline.source_version,
            source_timeline_sha256=timeline.source_sha256,
            source_event_count=len(timeline.events),
            retained_event_count=len(retained),
            max_retained_event_time=None if max_time is None else max_time.isoformat(),
        )
        return TruncatedStayHistory(
            subject_id=timeline.subject_id,
            stay_id=timeline.stay_id,
            intime=timeline.intime,
            outtime=timeline.outtime,
            events=tuple(retained), support_intervals=tuple(support_intervals),
            contract=contract,
            prediction_row=prediction_row,
            trace=trace,
        )
