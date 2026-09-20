"""Phase-4 structural cohort selection and Vedant timestamp integration."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from data.timestamps import (
    MAX_EXAMPLES_PER_STAY,
    RetainedICUStay,
    TIMESTAMP_SPEC_VERSION,
    generate_prediction_timestamps,
    validate_prediction_rows,
)

from .validation import SyntheticValidationError, parse_utc


COHORT_VERSION = "synthetic_retained_cohort_v2"
INDEX_SCHEMA_VERSION = "synthetic_structural_index_v2"


class SyntheticCohortError(ValueError):
    """Raised when structural cohort input violates the frozen contract."""


@dataclass(frozen=True)
class CohortBuild:
    retained_cohort: tuple[Mapping[str, Any], ...]
    structural_index: tuple[Mapping[str, Any], ...]
    exclusion_counts: Mapping[str, int]
    generated_subject_count: int
    generated_episode_count: int


def _valid_identifier(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _iso(value: datetime) -> str:
    return value.isoformat(timespec="microseconds").replace("+00:00", "Z")


def _index_row(row) -> Mapping[str, Any]:
    value = asdict(row)
    value["prediction_time"] = _iso(row.prediction_time)
    return value


def _assert_events_within_episodes(events: Sequence[Mapping], episodes: Sequence[Mapping]) -> None:
    bounds = {}
    for episode in episodes:
        stay_id = episode.get("stay_id")
        if _valid_identifier(stay_id) and episode.get("intime") and episode.get("outtime"):
            try:
                bounds[stay_id] = (episode.get("subject_id"), parse_utc(episode["intime"]), parse_utc(episode["outtime"]))
            except SyntheticValidationError:
                continue
    for event in events:
        stay_id = event.get("stay_id")
        if stay_id not in bounds:
            raise SyntheticCohortError(f"UPSTREAM PHASE-3 DEFECT — event references unknown/invalid stay: {stay_id!r}")
        subject_id, start, end = bounds[stay_id]
        if event.get("subject_id") != subject_id:
            raise SyntheticCohortError("UPSTREAM PHASE-3 DEFECT — event subject/stay linkage conflict")
        try:
            event_time = parse_utc(event.get("event_time"))
        except SyntheticValidationError as error:
            raise SyntheticCohortError("UPSTREAM PHASE-3 DEFECT — invalid event timestamp") from error
        if not start <= event_time <= end:
            raise SyntheticCohortError("UPSTREAM PHASE-3 DEFECT — event outside explicit episode boundaries")


def build_cohort(
    subjects: Sequence[Mapping],
    episodes: Sequence[Mapping],
    events: Sequence[Mapping],
    *,
    minimum_age: int,
    allowed_cardiac_groups: Sequence[str],
) -> CohortBuild:
    """Select by static/structural facts and call Vedant's generator for every cutoff."""

    subject_ids = [row.get("subject_id") for row in subjects if _valid_identifier(row.get("subject_id"))]
    if len(subject_ids) != len(set(subject_ids)):
        raise SyntheticCohortError("duplicate subject_id is a generator contract violation")
    stay_owners = {}
    episodes_by_subject = defaultdict(list)
    for episode in episodes:
        stay_id, subject_id = episode.get("stay_id"), episode.get("subject_id")
        if _valid_identifier(stay_id):
            prior = stay_owners.get(stay_id)
            if prior is not None and prior != subject_id:
                raise SyntheticCohortError("cross-subject stay_id collision")
            if prior is not None:
                raise SyntheticCohortError("duplicate stay_id is a generator contract violation")
            stay_owners[stay_id] = subject_id
        episodes_by_subject[subject_id].append(episode)
    known = set(subject_ids)
    if any(subject_id not in known for subject_id in episodes_by_subject):
        raise SyntheticCohortError("episode subject foreign key is invalid")
    if any(len(rows) > 1 for rows in episodes_by_subject.values()):
        raise SyntheticCohortError("multiple episodes per subject violate the frozen one-episode generator contract")
    _assert_events_within_episodes(events, episodes)

    retained, index, exclusions = [], [], Counter()
    allowed = set(allowed_cardiac_groups)
    for subject in sorted(subjects, key=lambda row: str(row.get("subject_id", ""))):
        subject_id = subject.get("subject_id")
        reason = None
        if not _valid_identifier(subject_id): reason = "INVALID_SUBJECT_ID"
        elif not isinstance(subject.get("age_years"), int) or isinstance(subject.get("age_years"), bool): reason = "INVALID_AGE"
        elif subject["age_years"] < minimum_age: reason = "NOT_ADULT"
        elif subject.get("cardiac_condition_group") not in allowed: reason = "OUTSIDE_CARDIAC_SCOPE"
        subject_episodes = episodes_by_subject.get(subject_id, [])
        if reason is None and not subject_episodes: reason = "MISSING_EPISODE"
        if reason is not None:
            exclusions[reason] += 1
            continue
        episode = subject_episodes[0]
        stay_id = episode.get("stay_id")
        if not _valid_identifier(stay_id):
            exclusions["INVALID_STAY_ID"] += 1; continue
        if episode.get("intime") is None:
            exclusions["INVALID_INTIME"] += 1; continue
        if episode.get("outtime") is None:
            exclusions["INVALID_OUTTIME"] += 1; continue
        try:
            intime, outtime = parse_utc(episode["intime"]), parse_utc(episode["outtime"])
        except SyntheticValidationError:
            exclusions["INVALID_EPISODE_TIMESTAMP"] += 1; continue
        if outtime <= intime:
            exclusions["NONPOSITIVE_EPISODE_DURATION"] += 1; continue
        stay = RetainedICUStay(subject_id, stay_id, intime, outtime)
        rows = generate_prediction_timestamps((stay,))
        if not rows:
            exclusions["NO_LEGAL_PREDICTION_CUTOFF"] += 1; continue
        retained.append({"subject_id":subject_id,"stay_id":stay_id,"intime":episode["intime"],"outtime":episode["outtime"],"cohort_version":COHORT_VERSION})
        index.extend(_index_row(row) for row in rows)
    retained.sort(key=lambda row:(row["subject_id"],row["stay_id"]))
    index.sort(key=lambda row:(row["subject_id"],row["stay_id"],row["prediction_time"],row["grid_index"]))
    _validate_outputs(retained, index)
    return CohortBuild(tuple(retained),tuple(index),dict(sorted(exclusions.items())),len(subjects),len(episodes))


def _validate_outputs(cohort: Sequence[Mapping], index: Sequence[Mapping]) -> None:
    cohort_fields = {"subject_id","stay_id","intime","outtime","cohort_version"}
    index_fields = {"subject_id","stay_id","prediction_time","grid_index","icu_elapsed_hours","recovery24_followup_available","recovery48_followup_available","support24_full_followup_available","icu_time_temporally_eligible","timestamp_spec_version"}
    forbidden = {"target","label","sofa","delta_sofa_24","delta_sofa_48","remaining_stay","support_label","split","fold","anchor_year_group"}
    if any(set(row) != cohort_fields or set(row) & forbidden for row in cohort): raise SyntheticCohortError("retained cohort schema contains missing/extra/forbidden fields")
    if any(set(row) != index_fields or set(row) & forbidden for row in index): raise SyntheticCohortError("structural index is not outcome-free")
    if any(row["cohort_version"] != COHORT_VERSION for row in cohort): raise SyntheticCohortError("cohort version mismatch")
    if any(row["timestamp_spec_version"] != TIMESTAMP_SPEC_VERSION for row in index): raise SyntheticCohortError("timestamp spec version mismatch")
    if any(not 0 <= row["grid_index"] < MAX_EXAMPLES_PER_STAY for row in index): raise SyntheticCohortError("grid index outside 0..11")
    identity = [(row["stay_id"],row["prediction_time"]) for row in index]
    grid_identity = [(row["subject_id"],row["stay_id"],row["grid_index"]) for row in index]
    if len(identity) != len(set(identity)) or len(grid_identity) != len(set(grid_identity)): raise SyntheticCohortError("duplicate structural prediction identity")
    stays = [RetainedICUStay(row["subject_id"],row["stay_id"],parse_utc(row["intime"]),parse_utc(row["outtime"])) for row in cohort]
    generated = generate_prediction_timestamps(stays)
    expected = tuple(_index_row(row) for row in generated)
    if tuple(index) != expected: raise SyntheticCohortError("structural index differs from Vedant timestamp generator")
    validate_prediction_rows(generated, stays)


def validate_output_schemas(cohort_schema: Mapping, index_schema: Mapping) -> None:
    for schema in (cohort_schema,index_schema):
        if schema.get("additional_fields_allowed") is not False: raise SyntheticCohortError("output schema must fail closed on additional fields")
        for field in schema.get("fields",[]):
            if field.get("model_eligible") is not False: raise SyntheticCohortError(f"structural field cannot be model eligible: {field.get('name')}")
    prediction = next(field for field in index_schema["fields"] if field["name"]=="prediction_time")
    outtime = next(field for field in cohort_schema["fields"] if field["name"]=="outtime")
    if prediction["role"] != "INDEX_ROUTING" or outtime["role"] != "STRUCTURAL_LABEL_ONLY": raise SyntheticCohortError("prediction_time/outtime roles are unsafe")
