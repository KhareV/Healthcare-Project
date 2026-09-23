"""Validated synthetic/demo-safe dashboard timeline and cutoff catalog."""

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence, Tuple

from data.timestamps import RetainedICUStay, generate_prediction_rows_for_stay
from serving.recovery import CurrentSOFAState


SYNTHETIC_CLASSIFICATION = (
    "SYNTHETIC DASHBOARD DEVELOPMENT FIXTURE — NON-SCIENTIFIC — NOT FINAL DEMO DATA"
)
ALLOWED_CATEGORIES = ("vital", "lab", "support_state")


class DashboardCatalogError(ValueError):
    pass


def _parse_time(value: object, field: str) -> datetime:
    if not isinstance(value, str):
        raise DashboardCatalogError(field + " must be an ISO-8601 string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise DashboardCatalogError(field + " is malformed") from error
    if parsed.utcoffset() is None:
        raise DashboardCatalogError(field + " must be timezone-aware")
    return parsed


def _instant(value: str) -> datetime:
    return _parse_time(value, "timestamp").astimezone(timezone.utc)


@dataclass(frozen=True)
class TimelineEvent:
    prediction_available_time: str
    category: str
    label: str
    value: object
    observed: bool


@dataclass(frozen=True)
class ReplayStay:
    subject_id: str
    stay_id: str
    intime: str
    legal_cutoffs: Tuple[str, ...]
    events: Tuple[TimelineEvent, ...]
    current_sofa_by_cutoff: Mapping[str, float]
    sofa_version: str
    current_sofa_source_version: str = "SYNTHETIC_DASHBOARD_CURRENT_SOFA_PHASE13_V1"
    current_sofa_source_sha256: str = "6" * 64

    def elapsed_hours(self, cutoff: str) -> float:
        if cutoff not in self.legal_cutoffs:
            raise DashboardCatalogError("cutoff is not approved for this stay")
        return (_instant(cutoff) - _instant(self.intime)).total_seconds() / 3600.0


class DashboardCatalog:
    def __init__(self, stays: Sequence[ReplayStay], *, scope: str) -> None:
        if scope not in ("synthetic", "real"):
            raise DashboardCatalogError(
                "dashboard catalog scope must be synthetic or real"
            )
        indexed = {}
        for stay in stays:
            # The Phase-6 mock fixture uses "SYNTHETIC_"-prefixed placeholder
            # identifiers; Stage 4's real catalog uses the accepted cohort's
            # own stay identifiers ("SYN-E-..."), which are legitimately
            # train/validation-only demo-safe subjects (never the sealed
            # final-test partition), not a Phase-6 placeholder claim.
            if scope == "synthetic" and not stay.stay_id.startswith("SYNTHETIC_"):
                raise DashboardCatalogError("synthetic catalog requires demo-safe identifiers")
            if stay.stay_id in indexed:
                raise DashboardCatalogError("duplicate replay stay")
            indexed[stay.stay_id] = stay
        if not indexed:
            raise DashboardCatalogError("dashboard catalog must contain at least one stay")
        self._stays = MappingProxyType(indexed)
        self.scope = scope

    @property
    def stay_ids(self) -> Tuple[str, ...]:
        return tuple(self._stays)

    def stay(self, stay_id: str) -> ReplayStay:
        try:
            return self._stays[stay_id]
        except KeyError as error:
            raise DashboardCatalogError("selected demo stay is unavailable") from error

    def timeline_through(self, stay_id: str, cutoff: str) -> Tuple[TimelineEvent, ...]:
        stay = self.stay(stay_id)
        if cutoff not in stay.legal_cutoffs:
            raise DashboardCatalogError("cutoff is not approved for this stay")
        boundary = _instant(cutoff)
        return tuple(
            event
            for event in stay.events
            if _instant(event.prediction_available_time) <= boundary
        )

    def current_sofa(self, stay_id: str, cutoff: str) -> CurrentSOFAState:
        stay = self.stay(stay_id)
        if cutoff not in stay.legal_cutoffs:
            raise DashboardCatalogError("cutoff is not approved for this stay")
        return CurrentSOFAState(
            stay_id=stay_id,
            prediction_time=cutoff,
            value=stay.current_sofa_by_cutoff[cutoff],
            sofa_version=stay.sofa_version,
            source_version=stay.current_sofa_source_version,
            source_sha256=stay.current_sofa_source_sha256,
            component_observed=None,
        )


def load_synthetic_dashboard_catalog(path: Path) -> DashboardCatalog:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DashboardCatalogError("synthetic dashboard fixture is unreadable") from error
    if (
        not isinstance(payload, dict)
        or payload.get("classification") != SYNTHETIC_CLASSIFICATION
        or payload.get("scope") != "synthetic"
    ):
        raise DashboardCatalogError("dashboard fixture is not explicitly synthetic")
    contract = payload.get("timeline_contract")
    stays_payload = payload.get("stays")
    sofa_version = payload.get("sofa_version")
    if not isinstance(contract, dict) or not isinstance(stays_payload, list):
        raise DashboardCatalogError("synthetic dashboard fixture is malformed")
    required_contract = {
        "event_time_field", "category_field", "label_field", "value_field", "observed_field"
    }
    if set(contract) != required_contract or not isinstance(sofa_version, str):
        raise DashboardCatalogError("timeline field contract is malformed")
    stays = []
    for item in stays_payload:
        if not isinstance(item, dict):
            raise DashboardCatalogError("stay fixture must be an object")
        intime = _parse_time(item.get("intime"), "intime")
        outtime = _parse_time(item.get("outtime"), "outtime")
        canonical = {
            row.prediction_time.isoformat()
            for row in generate_prediction_rows_for_stay(
                RetainedICUStay(
                    subject_id=item.get("subject_id"),
                    stay_id=item.get("stay_id"),
                    intime=intime,
                    outtime=outtime,
                )
            )
        }
        approved = item.get("approved_cutoffs")
        current = item.get("current_sofa")
        if (
            not isinstance(approved, list)
            or len(approved) < 3
            or len(set(approved)) != len(approved)
            or not set(approved).issubset(canonical)
            or not isinstance(current, dict)
            or set(current) != set(approved)
        ):
            raise DashboardCatalogError("approved replay cutoff contract is invalid")
        events = []
        for row in item.get("events", ()):
            if not isinstance(row, dict) or set(row) != set(contract.values()):
                raise DashboardCatalogError("timeline event does not match explicit schema")
            category = row[contract["category_field"]]
            label = row[contract["label_field"]]
            observed = row[contract["observed_field"]]
            _parse_time(row[contract["event_time_field"]], "event time")
            if (
                category not in ALLOWED_CATEGORIES
                or not isinstance(label, str)
                or not label
                or not isinstance(observed, bool)
            ):
                raise DashboardCatalogError("timeline event identity is invalid")
            events.append(
                TimelineEvent(
                    prediction_available_time=row[contract["event_time_field"]],
                    category=category,
                    label=label,
                    value=row[contract["value_field"]],
                    observed=observed,
                )
            )
        current_values = {}
        for cutoff, value in current.items():
            state = CurrentSOFAState(
                stay_id=item["stay_id"], prediction_time=cutoff, value=value,
                sofa_version=sofa_version,
                source_version="SYNTHETIC_DASHBOARD_CURRENT_SOFA_PHASE13_V1",
                source_sha256="6" * 64,
            )
            state.validate()
            current_values[cutoff] = float(value)
        stays.append(
            ReplayStay(
                subject_id=str(item["subject_id"]),
                stay_id=str(item["stay_id"]),
                intime=intime.isoformat(),
                legal_cutoffs=tuple(approved),
                events=tuple(sorted(events, key=lambda event: _instant(event.prediction_available_time))),
                current_sofa_by_cutoff=MappingProxyType(current_values),
                sofa_version=sofa_version,
            )
        )
    return DashboardCatalog(stays, scope="synthetic")
