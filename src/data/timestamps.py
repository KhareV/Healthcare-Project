"""Canonical legal ICU prediction timestamp generation.

This module owns only the prediction clock and purely temporal follow-up
availability. It does not select the first ICU stay, query MIMIC, build
features, calculate labels, assign splits, or inspect outcomes.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterable, List, Sequence, Tuple, Union


Identifier = Union[int, str]

TIMESTAMP_SPEC_VERSION = "timestamp_spec_v1"
EARLIEST_OFFSET_HOURS = 24
SPACING_HOURS = 6
K_MIN = 0
K_MAX = 11
MAX_EXAMPLES_PER_STAY = 12
MINIMUM_TIME_BEFORE_OUTTIME_HOURS = 6
RECOVERY_24H_FOLLOWUP_HOURS = 24
RECOVERY_48H_FOLLOWUP_HOURS = 48
SUPPORT_24H_FULL_FOLLOWUP_HOURS = 24


class TimestampValidationError(ValueError):
    """Raised when retained-stay input or generated output is invalid."""


@dataclass(frozen=True)
class RetainedICUStay:
    """Typed internal input supplied by the upstream cohort layer.

    Field names are project engineering-contract names, not claims about raw
    MIMIC table columns. One instance must already represent the retained
    chronologically first ICU stay for its subject.
    """

    subject_id: Identifier
    stay_id: Identifier
    intime: datetime
    outtime: datetime


@dataclass(frozen=True)
class CandidateCutoff:
    """One of the twelve deterministic candidates before boundary filtering."""

    grid_index: int
    prediction_time: datetime


@dataclass(frozen=True)
class HorizonAvailability:
    """Purely temporal follow-up availability, not final label eligibility."""

    recovery24_followup_available: bool
    recovery48_followup_available: bool
    support24_full_followup_available: bool
    icu_time_temporally_eligible: bool = True


@dataclass(frozen=True)
class PredictionTimestamp:
    """Canonical emitted-row contract for downstream project components."""

    subject_id: Identifier
    stay_id: Identifier
    prediction_time: datetime
    grid_index: int
    icu_elapsed_hours: int
    recovery24_followup_available: bool
    recovery48_followup_available: bool
    support24_full_followup_available: bool
    icu_time_temporally_eligible: bool
    timestamp_spec_version: str = TIMESTAMP_SPEC_VERSION


def _is_supported_identifier(value: object) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return True
    return isinstance(value, str) and bool(value.strip())


def _is_aware(value: datetime) -> bool:
    return value.utcoffset() is not None


def _comparable_instant(value: datetime) -> datetime:
    if _is_aware(value):
        return value.astimezone(timezone.utc)
    return value


def _add_elapsed_hours(value: datetime, hours: int) -> datetime:
    """Add actual elapsed hours while preserving the input timezone form."""

    delta = timedelta(hours=hours)
    if not _is_aware(value):
        return value + delta
    return (value.astimezone(timezone.utc) + delta).astimezone(value.tzinfo)


def _identifier_sort_key(value: Identifier) -> Tuple[int, Union[int, str]]:
    if isinstance(value, int):
        return (0, value)
    return (1, value)


def validate_retained_stay(stay: RetainedICUStay) -> None:
    """Validate one already-retained ICU-stay contract."""

    if not isinstance(stay, RetainedICUStay):
        raise TimestampValidationError("input must be a RetainedICUStay")
    if not _is_supported_identifier(stay.subject_id):
        raise TimestampValidationError("subject_id must be a non-empty int or string")
    if not _is_supported_identifier(stay.stay_id):
        raise TimestampValidationError("stay_id must be a non-empty int or string")
    if not isinstance(stay.intime, datetime):
        raise TimestampValidationError("intime must be a datetime")
    if not isinstance(stay.outtime, datetime):
        raise TimestampValidationError("outtime must be a datetime")
    if _is_aware(stay.intime) != _is_aware(stay.outtime):
        raise TimestampValidationError(
            "intime and outtime must use consistent timezone awareness"
        )
    if _comparable_instant(stay.outtime) <= _comparable_instant(stay.intime):
        raise TimestampValidationError("outtime must be strictly after intime")


def validate_retained_stays(stays: Iterable[RetainedICUStay]) -> Tuple[RetainedICUStay, ...]:
    """Validate the collection and reject duplicate stays or subjects."""

    materialized = tuple(stays)
    seen_subject_ids = set()  # type: set
    seen_stay_ids = set()  # type: set
    for stay in materialized:
        validate_retained_stay(stay)
        subject_key = (type(stay.subject_id), stay.subject_id)
        stay_key = (type(stay.stay_id), stay.stay_id)
        if subject_key in seen_subject_ids:
            raise TimestampValidationError(
                "duplicate subject_id: input must contain one retained ICU stay per subject"
            )
        if stay_key in seen_stay_ids:
            raise TimestampValidationError(
                "duplicate stay_id: duplicate canonical examples are prohibited"
            )
        seen_subject_ids.add(subject_key)
        seen_stay_ids.add(stay_key)
    return materialized


def generate_candidate_cutoffs(stay: RetainedICUStay) -> Tuple[CandidateCutoff, ...]:
    """Generate exactly the frozen k=0..11 candidates without filtering."""

    validate_retained_stay(stay)
    return tuple(
        CandidateCutoff(
            grid_index=k,
            prediction_time=_add_elapsed_hours(
                stay.intime, EARLIEST_OFFSET_HOURS + SPACING_HOURS * k
            ),
        )
        for k in range(K_MIN, K_MAX + 1)
    )


def retain_legal_cutoffs(
    stay: RetainedICUStay, candidates: Sequence[CandidateCutoff]
) -> Tuple[CandidateCutoff, ...]:
    """Apply the inclusive t <= outtime-6h retention boundary."""

    validate_retained_stay(stay)
    legal_boundary = _add_elapsed_hours(
        stay.outtime, -MINIMUM_TIME_BEFORE_OUTTIME_HOURS
    )
    retained_candidates = []  # type: List[CandidateCutoff]
    seen_grid_indices = set()
    for candidate in candidates:
        if not isinstance(candidate, CandidateCutoff):
            raise TimestampValidationError("candidate must be CandidateCutoff")
        if not K_MIN <= candidate.grid_index <= K_MAX:
            raise TimestampValidationError("candidate grid_index is outside 0..11")
        if candidate.grid_index in seen_grid_indices:
            raise TimestampValidationError("duplicate candidate grid_index")
        seen_grid_indices.add(candidate.grid_index)
        expected_time = _add_elapsed_hours(
            stay.intime,
            EARLIEST_OFFSET_HOURS + SPACING_HOURS * candidate.grid_index,
        )
        if candidate.prediction_time != expected_time:
            raise TimestampValidationError("candidate is not on the frozen grid")
        if _comparable_instant(candidate.prediction_time) <= _comparable_instant(
            legal_boundary
        ):
            retained_candidates.append(candidate)

    retained = tuple(sorted(retained_candidates, key=lambda item: item.grid_index))
    if len(retained) > MAX_EXAMPLES_PER_STAY:
        raise TimestampValidationError("more than 12 legal cutoffs were produced")
    return retained


def calculate_horizon_availability(
    prediction_time: datetime, outtime: datetime
) -> HorizonAvailability:
    """Calculate time-only future follow-up flags with inclusive equality."""

    if not isinstance(prediction_time, datetime) or not isinstance(outtime, datetime):
        raise TimestampValidationError("prediction_time and outtime must be datetimes")
    if _is_aware(prediction_time) != _is_aware(outtime):
        raise TimestampValidationError(
            "prediction_time and outtime must use consistent timezone awareness"
        )
    if _comparable_instant(prediction_time) >= _comparable_instant(outtime):
        raise TimestampValidationError("prediction_time must be before outtime")

    def available(hours: int) -> bool:
        horizon_end = _add_elapsed_hours(prediction_time, hours)
        return _comparable_instant(horizon_end) <= _comparable_instant(outtime)

    recovery24 = available(RECOVERY_24H_FOLLOWUP_HOURS)
    return HorizonAvailability(
        recovery24_followup_available=recovery24,
        recovery48_followup_available=available(RECOVERY_48H_FOLLOWUP_HOURS),
        support24_full_followup_available=available(
            SUPPORT_24H_FULL_FOLLOWUP_HOURS
        ),
    )


def generate_prediction_rows_for_stay(
    stay: RetainedICUStay,
) -> Tuple[PredictionTimestamp, ...]:
    """Generate legal canonical rows for one retained ICU stay."""

    candidates = generate_candidate_cutoffs(stay)
    legal_candidates = retain_legal_cutoffs(stay, candidates)
    rows: List[PredictionTimestamp] = []
    for candidate in legal_candidates:
        availability = calculate_horizon_availability(
            candidate.prediction_time, stay.outtime
        )
        rows.append(
            PredictionTimestamp(
                subject_id=stay.subject_id,
                stay_id=stay.stay_id,
                prediction_time=candidate.prediction_time,
                grid_index=candidate.grid_index,
                icu_elapsed_hours=(
                    EARLIEST_OFFSET_HOURS + SPACING_HOURS * candidate.grid_index
                ),
                recovery24_followup_available=(
                    availability.recovery24_followup_available
                ),
                recovery48_followup_available=(
                    availability.recovery48_followup_available
                ),
                support24_full_followup_available=(
                    availability.support24_full_followup_available
                ),
                icu_time_temporally_eligible=(
                    availability.icu_time_temporally_eligible
                ),
            )
        )
    result = tuple(rows)
    validate_prediction_rows(result, (stay,))
    return result


def validate_prediction_rows(
    rows: Sequence[PredictionTimestamp], stays: Sequence[RetainedICUStay]
) -> None:
    """Validate the emitted schema against the frozen mathematical clock."""

    validated_stays = validate_retained_stays(stays)
    stay_by_key = {
        (type(stay.stay_id), stay.stay_id): stay for stay in validated_stays
    }
    seen_rows = set()  # type: set
    counts = {}  # type: dict

    for row in rows:
        if not isinstance(row, PredictionTimestamp):
            raise TimestampValidationError("output row must be PredictionTimestamp")
        stay_key = (type(row.stay_id), row.stay_id)
        stay = stay_by_key.get(stay_key)
        if stay is None or row.subject_id != stay.subject_id:
            raise TimestampValidationError("output row does not match an input stay")
        if not K_MIN <= row.grid_index <= K_MAX:
            raise TimestampValidationError("grid_index is outside 0..11")

        expected_time = _add_elapsed_hours(
            stay.intime, EARLIEST_OFFSET_HOURS + SPACING_HOURS * row.grid_index
        )
        if row.prediction_time != expected_time:
            raise TimestampValidationError("prediction_time is not on the frozen grid")
        if row.icu_elapsed_hours != EARLIEST_OFFSET_HOURS + SPACING_HOURS * row.grid_index:
            raise TimestampValidationError("icu_elapsed_hours does not match grid_index")
        latest_legal = _add_elapsed_hours(
            stay.outtime, -MINIMUM_TIME_BEFORE_OUTTIME_HOURS
        )
        if _comparable_instant(row.prediction_time) > _comparable_instant(latest_legal):
            raise TimestampValidationError("prediction_time exceeds outtime-6h")
        if not row.icu_time_temporally_eligible:
            raise TimestampValidationError(
                "every emitted row must be temporally eligible for ICU-time"
            )
        expected_availability = calculate_horizon_availability(
            row.prediction_time, stay.outtime
        )
        actual_flags = (
            row.recovery24_followup_available,
            row.recovery48_followup_available,
            row.support24_full_followup_available,
            row.icu_time_temporally_eligible,
        )
        expected_flags = (
            expected_availability.recovery24_followup_available,
            expected_availability.recovery48_followup_available,
            expected_availability.support24_full_followup_available,
            expected_availability.icu_time_temporally_eligible,
        )
        if actual_flags != expected_flags:
            raise TimestampValidationError(
                "time-based follow-up flags do not match outtime"
            )
        if row.timestamp_spec_version != TIMESTAMP_SPEC_VERSION:
            raise TimestampValidationError("unexpected timestamp spec version")

        row_key = (stay_key, row.grid_index)
        if row_key in seen_rows:
            raise TimestampValidationError("duplicate canonical prediction row")
        seen_rows.add(row_key)
        counts[stay_key] = counts.get(stay_key, 0) + 1
        if counts[stay_key] > MAX_EXAMPLES_PER_STAY:
            raise TimestampValidationError("more than 12 rows emitted for one stay")


def generate_prediction_timestamps(
    stays: Iterable[RetainedICUStay],
) -> Tuple[PredictionTimestamp, ...]:
    """Generate deterministic rows ordered by subject, stay, then cutoff."""

    materialized = validate_retained_stays(stays)
    ordered_stays = sorted(
        materialized,
        key=lambda stay: (
            _identifier_sort_key(stay.subject_id),
            _identifier_sort_key(stay.stay_id),
        ),
    )
    rows = tuple(
        row
        for stay in ordered_stays
        for row in generate_prediction_rows_for_stay(stay)
    )
    validate_prediction_rows(rows, ordered_stays)
    return rows
