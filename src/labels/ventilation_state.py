"""Invasive-ventilation ON/OFF lookup over normalized concept intervals.

This is an INTERNAL ENGINEERING CONTRACT.  It does not extract MIMIC data or
recreate MIMIC-Code logic.  Real execution remains blocked until the shared
event dictionary contains a pinned, reviewed ventilation concept.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Iterable, Mapping, Tuple, Union

from labels.event_dict import EventDictionary, EventDictionaryError
from labels.support_state import ExecutionMode


Identifier = Union[int, str]


class VentilationStateError(ValueError):
    """Raised for invalid or incompatible normalized ventilation states."""


class RespiratoryCategory(str, Enum):
    """Synthetic/internal categories, not raw MIMIC-Code values."""

    INVASIVE = "INVASIVE"
    NON_INVASIVE = "NON_INVASIVE"
    HFNC = "HFNC"
    ORDINARY_OXYGEN = "ORDINARY_OXYGEN"


class InvasiveVentilationState(str, Enum):
    OFF = "OFF"
    ON = "ON"


@dataclass(frozen=True)
class NormalizedVentilationInterval:
    """Already-classified interval from a versioned concept adapter."""

    stay_id: Identifier
    category: RespiratoryCategory
    interval_start: datetime
    interval_end: datetime
    source_state_ref: str
    concept_version: str
    adapter_version: str
    normalization_provenance_ref: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "NormalizedVentilationInterval":
        """Parse normalized fixture/adapter output and reject unknown categories."""

        def timestamp(field: str) -> datetime:
            raw = value.get(field)
            if not isinstance(raw, str):
                raise VentilationStateError(field + " must be an ISO-8601 string")
            try:
                return datetime.fromisoformat(raw.replace("Z", "+00:00"))
            except ValueError as error:
                raise VentilationStateError(field + " is malformed") from error

        raw_category = value.get("category")
        if not isinstance(raw_category, str):
            raise VentilationStateError("category must be a known internal string")
        try:
            category = RespiratoryCategory(raw_category)
        except ValueError as error:
            raise VentilationStateError(
                "unknown respiratory category; refusing to map it to OFF"
            ) from error
        return cls(
            stay_id=value.get("stay_id"),  # type: ignore[arg-type]
            category=category,
            interval_start=timestamp("interval_start"),
            interval_end=timestamp("interval_end"),
            source_state_ref=value.get("source_state_ref"),  # type: ignore[arg-type]
            concept_version=value.get("concept_version"),  # type: ignore[arg-type]
            adapter_version=value.get("adapter_version"),  # type: ignore[arg-type]
            normalization_provenance_ref=value.get(
                "normalization_provenance_ref"
            ),  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class VentilationStateResult:
    stay_id: Identifier
    cutoff: datetime
    invasive_state: InvasiveVentilationState
    invasive_on: bool
    vent_at_risk: bool
    active_categories: Tuple[RespiratoryCategory, ...]
    supporting_invasive_interval_refs: Tuple[str, ...]
    event_dictionary_version: str
    event_dictionary_sha256: str
    state_rule_version: str
    concept_name: str
    concept_version: str
    adapter_version: str
    interval_boundary_policy: str
    sofa_alignment_status: str
    execution_mode: ExecutionMode


def _valid_identifier(value: object) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return True
    return isinstance(value, str) and bool(value.strip())


def _instant(value: datetime) -> datetime:
    if value.utcoffset() is None:
        return value
    return value.astimezone(timezone.utc)


def _validate_interval(interval: NormalizedVentilationInterval) -> None:
    if not isinstance(interval, NormalizedVentilationInterval):
        raise VentilationStateError(
            "records must be NormalizedVentilationInterval instances"
        )
    if not _valid_identifier(interval.stay_id):
        raise VentilationStateError("stay_id must be a non-empty int or string")
    if not isinstance(interval.category, RespiratoryCategory):
        raise VentilationStateError(
            "unknown respiratory category; refusing to map it to OFF"
        )
    for field, value in (
        ("source_state_ref", interval.source_state_ref),
        ("concept_version", interval.concept_version),
        ("adapter_version", interval.adapter_version),
        ("normalization_provenance_ref", interval.normalization_provenance_ref),
    ):
        if not isinstance(value, str) or not value.strip():
            raise VentilationStateError(field + " must be a non-empty string")
    if not isinstance(interval.interval_start, datetime):
        raise VentilationStateError("interval_start must be a datetime")
    if not isinstance(interval.interval_end, datetime):
        raise VentilationStateError("interval_end must be a datetime")
    if (interval.interval_start.utcoffset() is None) != (
        interval.interval_end.utcoffset() is None
    ):
        raise VentilationStateError(
            "interval timestamps must use consistent timezone awareness"
        )
    if _instant(interval.interval_end) <= _instant(interval.interval_start):
        raise VentilationStateError("normalized ventilation interval end must be after start")


def _is_active(
    interval: NormalizedVentilationInterval, cutoff: datetime, policy: str
) -> bool:
    start = _instant(interval.interval_start)
    end = _instant(interval.interval_end)
    point = _instant(cutoff)
    if policy == "CLOSED_OPEN":
        return start <= point < end
    if policy == "CLOSED_CLOSED":
        return start <= point <= end
    raise VentilationStateError("unsupported ventilation interval-boundary policy")


def query_invasive_ventilation_state(
    intervals: Iterable[NormalizedVentilationInterval],
    *,
    stay_id: Identifier,
    cutoff: datetime,
    event_dictionary: EventDictionary,
    execution_mode: ExecutionMode,
) -> VentilationStateResult:
    """Return invasive ventilation state at ``cutoff`` without future outcomes."""

    if not _valid_identifier(stay_id):
        raise VentilationStateError("query stay_id must be a non-empty int or string")
    if not isinstance(cutoff, datetime):
        raise VentilationStateError("cutoff must be a datetime")
    if not isinstance(execution_mode, ExecutionMode):
        raise VentilationStateError("execution_mode must be explicit")

    contract = event_dictionary.ventilation
    if execution_mode is ExecutionMode.SCIENTIFIC:
        if not contract.is_frozen_for_scientific_use:
            raise EventDictionaryError(
                "BLOCKED — VERIFIED INVASIVE-VENTILATION CONCEPT REQUIRED"
            )
        if (
            contract.scientific_concept_name is None
            or contract.scientific_concept_version is None
            or contract.scientific_adapter_version is None
            or contract.scientific_boundary_policy is None
        ):
            raise EventDictionaryError("frozen ventilation concept metadata is incomplete")
        concept_name = contract.scientific_concept_name
        concept_version = contract.scientific_concept_version
        adapter_version = contract.scientific_adapter_version
        boundary_policy = contract.scientific_boundary_policy
        invasive_categories = frozenset({RespiratoryCategory.INVASIVE})
    else:
        concept_name = contract.synthetic_concept_name
        concept_version = contract.synthetic_concept_version
        adapter_version = contract.synthetic_adapter_version
        boundary_policy = contract.synthetic_boundary_policy
        invasive_categories = frozenset(
            RespiratoryCategory(value)
            for value in contract.synthetic_invasive_categories
        )

    materialized = tuple(intervals)
    cutoff_aware = cutoff.utcoffset() is not None
    seen_refs = set()
    seen_exact = set()
    active_categories = set()
    invasive_refs = []
    for interval in materialized:
        _validate_interval(interval)
        if interval.concept_version != concept_version:
            raise VentilationStateError("ventilation concept version mismatch")
        if interval.adapter_version != adapter_version:
            raise VentilationStateError("ventilation adapter version mismatch")
        if (interval.interval_start.utcoffset() is not None) != cutoff_aware:
            raise VentilationStateError(
                "cutoff and intervals must use consistent timezone awareness"
            )
        exact_key = (
            type(interval.stay_id),
            interval.stay_id,
            interval.category,
            interval.interval_start,
            interval.interval_end,
            interval.concept_version,
            interval.adapter_version,
            interval.normalization_provenance_ref,
        )
        if interval.source_state_ref in seen_refs:
            raise VentilationStateError("duplicate/conflicting source_state_ref")
        if exact_key in seen_exact:
            raise VentilationStateError("exact duplicate normalized ventilation interval")
        seen_refs.add(interval.source_state_ref)
        seen_exact.add(exact_key)

        if type(interval.stay_id) is not type(stay_id) or interval.stay_id != stay_id:
            continue
        if not _is_active(interval, cutoff, boundary_policy):
            continue
        active_categories.add(interval.category)
        if interval.category in invasive_categories:
            invasive_refs.append(interval.source_state_ref)

    ordered_categories = tuple(sorted(active_categories, key=lambda item: item.value))
    ordered_refs = tuple(sorted(invasive_refs))
    state = (
        InvasiveVentilationState.ON
        if ordered_refs
        else InvasiveVentilationState.OFF
    )
    return VentilationStateResult(
        stay_id=stay_id,
        cutoff=cutoff,
        invasive_state=state,
        invasive_on=state is InvasiveVentilationState.ON,
        vent_at_risk=state is InvasiveVentilationState.OFF,
        active_categories=ordered_categories,
        supporting_invasive_interval_refs=ordered_refs,
        event_dictionary_version=event_dictionary.event_dictionary_version,
        event_dictionary_sha256=event_dictionary.config_sha256,
        state_rule_version=contract.state_rule_version,
        concept_name=concept_name,
        concept_version=concept_version,
        adapter_version=adapter_version,
        interval_boundary_policy=boundary_policy,
        sofa_alignment_status=contract.sofa_alignment_status,
        execution_mode=execution_mode,
    )

