"""Deterministic vasopressor ON/OFF lookup over normalized active intervals.

This module does not normalize raw MIMIC records, infer valid rates, or decide
which real agents qualify.  Its input intervals assert that upstream logic has
already validated activity over their complete span.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Iterable, Mapping, Tuple, Union

from labels.event_dict import EventDictionary, EventDictionaryError


Identifier = Union[int, str]


class SupportStateError(ValueError):
    """Raised for invalid normalized events or unauthorized execution."""


class SupportState(str, Enum):
    OFF = "OFF"
    ON = "ON"


class ExecutionMode(str, Enum):
    SYNTHETIC = "SYNTHETIC"
    SCIENTIFIC = "SCIENTIFIC"


@dataclass(frozen=True)
class NormalizedActiveInterval:
    """Internal contract; field names are not claims about MIMIC columns."""

    stay_id: Identifier
    agent_key: str
    interval_start: datetime
    interval_end: datetime
    source_event_ref: str
    normalization_provenance_version: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "NormalizedActiveInterval":
        """Parse the normalized fixture/adapter representation fail-closed."""

        def timestamp(field: str) -> datetime:
            raw = value.get(field)
            if not isinstance(raw, str):
                raise SupportStateError(field + " must be an ISO-8601 string")
            try:
                parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            except ValueError as error:
                raise SupportStateError(field + " is malformed") from error
            return parsed

        return cls(
            stay_id=value.get("stay_id"),  # type: ignore[arg-type]
            agent_key=value.get("agent_key"),  # type: ignore[arg-type]
            interval_start=timestamp("interval_start"),
            interval_end=timestamp("interval_end"),
            source_event_ref=value.get("source_event_ref"),  # type: ignore[arg-type]
            normalization_provenance_version=value.get(
                "normalization_provenance_version"
            ),  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class VasopressorStateResult:
    stay_id: Identifier
    cutoff: datetime
    support_state: SupportState
    vaso_at_risk: bool
    qualifying_interval_refs: Tuple[str, ...]
    event_dictionary_version: str
    event_dictionary_sha256: str
    state_rule_version: str
    mapping_provenance_version: str
    interval_boundary_policy: str
    execution_mode: ExecutionMode


def _valid_identifier(value: object) -> bool:
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        or isinstance(value, str)
        and bool(value.strip())
    )


def _instant(value: datetime) -> datetime:
    if value.utcoffset() is None:
        return value
    return value.astimezone(timezone.utc)


def _validate_interval(interval: NormalizedActiveInterval) -> None:
    if not isinstance(interval, NormalizedActiveInterval):
        raise SupportStateError("events must be NormalizedActiveInterval instances")
    if not _valid_identifier(interval.stay_id):
        raise SupportStateError("stay_id must be a non-empty int or string")
    for field, value in (
        ("agent_key", interval.agent_key),
        ("source_event_ref", interval.source_event_ref),
        ("normalization_provenance_version", interval.normalization_provenance_version),
    ):
        if not isinstance(value, str) or not value.strip():
            raise SupportStateError(field + " must be a non-empty string")
    if not isinstance(interval.interval_start, datetime):
        raise SupportStateError("interval_start must be a datetime")
    if not isinstance(interval.interval_end, datetime):
        raise SupportStateError("interval_end must be a datetime")
    if (interval.interval_start.utcoffset() is None) != (
        interval.interval_end.utcoffset() is None
    ):
        raise SupportStateError("interval timestamps must use consistent timezone awareness")
    if _instant(interval.interval_end) <= _instant(interval.interval_start):
        raise SupportStateError("normalized active interval end must be after start")


def _is_active(interval: NormalizedActiveInterval, cutoff: datetime, policy: str) -> bool:
    start = _instant(interval.interval_start)
    end = _instant(interval.interval_end)
    point = _instant(cutoff)
    if policy == "CLOSED_OPEN":
        return start <= point < end
    if policy == "CLOSED_CLOSED":
        return start <= point <= end
    raise SupportStateError("unsupported interval-boundary policy")


def query_vasopressor_state(
    intervals: Iterable[NormalizedActiveInterval],
    *,
    stay_id: Identifier,
    cutoff: datetime,
    event_dictionary: EventDictionary,
    execution_mode: ExecutionMode,
) -> VasopressorStateResult:
    """Return baseline vasopressor state at exactly ``cutoff``.

    Synthetic mode uses only the dictionary's explicitly synthetic profile.
    Scientific mode requires a genuinely frozen, reviewed dictionary.
    """

    if not _valid_identifier(stay_id):
        raise SupportStateError("query stay_id must be a non-empty int or string")
    if not isinstance(cutoff, datetime):
        raise SupportStateError("cutoff must be a datetime")
    if not isinstance(execution_mode, ExecutionMode):
        raise SupportStateError("execution_mode must be explicit")

    if execution_mode is ExecutionMode.SCIENTIFIC:
        if not event_dictionary.is_frozen_for_scientific_use:
            raise EventDictionaryError(
                "BLOCKED — MIMIC SCHEMA / MAPPING VERIFICATION REQUIRED"
            )
        qualifying_keys = event_dictionary.scientific_qualifying_agent_keys
        boundary_policy = event_dictionary.scientific_boundary_policy
        mapping_provenance = event_dictionary.mapping_provenance_version
        if boundary_policy is None or mapping_provenance is None:
            raise EventDictionaryError("scientific mapping/boundary provenance is incomplete")
    else:
        qualifying_keys = event_dictionary.synthetic_qualifying_agent_keys
        boundary_policy = event_dictionary.synthetic_boundary_policy
        mapping_provenance = event_dictionary.synthetic_mapping_provenance_version

    materialized = tuple(intervals)
    seen_refs = set()
    seen_exact = set()
    active_refs = []
    cutoff_aware = cutoff.utcoffset() is not None
    for interval in materialized:
        _validate_interval(interval)
        if (interval.interval_start.utcoffset() is not None) != cutoff_aware:
            raise SupportStateError("cutoff and intervals must use consistent timezone awareness")
        exact_key = (
            type(interval.stay_id),
            interval.stay_id,
            interval.agent_key,
            interval.interval_start,
            interval.interval_end,
            interval.normalization_provenance_version,
        )
        if interval.source_event_ref in seen_refs:
            raise SupportStateError("duplicate/conflicting source_event_ref")
        if exact_key in seen_exact:
            raise SupportStateError("exact duplicate normalized interval")
        seen_refs.add(interval.source_event_ref)
        seen_exact.add(exact_key)

        if type(interval.stay_id) is not type(stay_id) or interval.stay_id != stay_id:
            continue
        if interval.agent_key not in qualifying_keys:
            continue
        if _is_active(interval, cutoff, boundary_policy):
            active_refs.append(interval.source_event_ref)

    refs = tuple(sorted(active_refs))
    state = SupportState.ON if refs else SupportState.OFF
    return VasopressorStateResult(
        stay_id=stay_id,
        cutoff=cutoff,
        support_state=state,
        vaso_at_risk=state is SupportState.OFF,
        qualifying_interval_refs=refs,
        event_dictionary_version=event_dictionary.event_dictionary_version,
        event_dictionary_sha256=event_dictionary.config_sha256,
        state_rule_version=event_dictionary.state_rule_version,
        mapping_provenance_version=mapping_provenance,
        interval_boundary_policy=boundary_policy,
        execution_mode=execution_mode,
    )

