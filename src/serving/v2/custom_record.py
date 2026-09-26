"""Custom-record intake: let an authenticated user type in their own
vitals/labs (and, optionally, organ-support state) and get a real
prediction through the exact same frozen V2 pipeline used for the demo
subjects, without touching any frozen scientific artifact.

Ephemeral, in-memory-only, process-lifetime storage (see
V2ServingRuntime.register_ephemeral_stay) -- restarting the server discards
every custom record. This is intentional: it is a demo feature, not a
clinical data store, and nothing here is written to any git-tracked path.
Every record is bound to the `owner_user_id` of the request that created it
(a verified Clerk user id, or the fixed local-dev owner when the backend has
no Clerk key configured -- see serving.v2.auth); ownership is enforced by
the API layer (api/v2_app.py), not here.

Deliberately out of scope: urine_output_volume. The renal SOFA component
(data/synthetic/sofa.py::_urine_candidate) requires a *fully gapless* chain
of interval-shaped readings whose union tiles exactly the 24-hour window
before each cutoff -- any gap makes it silently fall back to creatinine
alone, but any malformed/partial row for this concept raises a hard SOFAError
for every cutoff, not just the ones near it. Building genuine gapless 24h
coverage from episodic manual entry would mean fabricating volumes for
hours the user never actually reported, which this project's stance on
never faking data rules out. Renal SOFA still works from creatinine alone.
"""

from __future__ import annotations

import json
import math
import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Mapping, Optional, Sequence

from data.timestamps import RetainedICUStay, generate_prediction_rows_for_stay
from serving.v2.runtime import FEATURE_SCHEMA_PATH, V2ServingRuntime

CUSTOM_RECORD_SCHEMA_VERSION = "custom_record_v1"

MAX_OBSERVATIONS_PER_RECORD = 200
MAX_SUPPORT_INTERVALS_PER_RECORD = 20
MAX_EPHEMERAL_RECORDS = 500
# The legal-cutoff grid (data.timestamps: K_MIN..K_MAX) never extends past
# icu_elapsed_hours = 24 + 6*11 = 90h regardless of how far outtime is
# pushed out, so an observation timed later than that could never be visible
# to any servable cutoff -- capping intake here avoids silently accepting
# data that would otherwise just be invisible forever.
MAX_HOURS_SINCE_ADMISSION = 90.0
# "Currently active" support intervals get an end far enough beyond the grid
# ceiling that they read as active at every legal cutoff; the SOFA support
# provider clamps interval_end to min(interval_end, cutoff) per request, so
# this is cutoff-safe by construction, never a future-data leak.
ONGOING_SUPPORT_END_HOURS = 200.0

# Anchor point-in-time for every custom record. The calendar date itself is
# never meaningful anywhere in this pipeline -- every check is relative
# (elapsed hours, grid alignment) -- fixed for determinism and clean,
# whole-second ISO strings.
_ANCHOR_INTIME = datetime(2024, 1, 1, tzinfo=timezone.utc)

_LABELS: Mapping[str, Mapping[str, str]] = {
    "heart_rate": {"label": "Heart rate", "hint": "typically 60-100"},
    "mean_arterial_pressure": {"label": "Mean arterial pressure", "hint": "typically 65-100"},
    "systolic_blood_pressure": {"label": "Systolic blood pressure", "hint": "typically 90-140"},
    "diastolic_blood_pressure": {"label": "Diastolic blood pressure", "hint": "typically 60-90"},
    "respiratory_rate": {"label": "Respiratory rate", "hint": "typically 12-20"},
    "oxygen_saturation": {"label": "Oxygen saturation (SpO2)", "hint": "typically 95-100"},
    "temperature": {"label": "Temperature (°C)", "hint": "typically 36.5-37.5"},
    "pao2": {"label": "Arterial oxygen (PaO2)", "hint": "typically 80-100"},
    "fio2": {"label": "Fraction of inspired oxygen (FiO2)", "hint": "0.21 = room air, up to 1.0"},
    "creatinine": {"label": "Creatinine", "hint": "typically 0.6-1.3"},
    "platelet_count": {"label": "Platelet count", "hint": "typically 150-400"},
    "bilirubin_total": {"label": "Total bilirubin", "hint": "typically 0.1-1.2"},
    "glasgow_coma_scale": {"label": "Glasgow Coma Scale", "hint": "3 (worst) to 15 (normal)"},
    "lactate": {"label": "Lactate", "hint": "typically 0.5-2.0"},
}

# Which SOFA subcomponent each concept feeds -- used only to report honest
# "what's supported" facts in the readiness summary, never to compute SOFA
# ourselves (that stays entirely inside data/synthetic/sofa.py).
_SOFA_COMPONENT_CONCEPTS: Mapping[str, Sequence[str]] = {
    "respiratory": ("pao2", "fio2"),
    "coagulation": ("platelet_count",),
    "liver": ("bilirubin_total",),
    "cardiovascular": ("mean_arterial_pressure",),
    "central_nervous_system": ("glasgow_coma_scale",),
    "renal": ("creatinine",),  # urine_output_volume intentionally unsupported, see module docstring
}

VASOPRESSOR_AGENTS = ("norepinephrine", "epinephrine", "dopamine", "dobutamine")


class CustomRecordError(ValueError):
    """Raised for any invalid custom-record submission."""


@dataclass(frozen=True)
class _Observation:
    concept: str
    hours_since_admission: float
    value: float


def canonical_concepts(root: Path) -> List[Mapping[str, object]]:
    """The concepts a custom record may report, with the exact unit and
    provenance_id the frozen feature schema requires -- the caller never
    has to know or supply these, only pick a concept and a value."""

    schema = json.loads((Path(root) / FEATURE_SCHEMA_PATH).read_text(encoding="utf-8"))
    out: List[Mapping[str, object]] = []
    for channel in schema["temporal_channels"]:
        if channel["aggregation"] == "STATE_AT_BIN_END":
            continue  # organ-support channels: entered separately, see support_intervals below
        concept = channel["source_concept"]
        meta = _LABELS.get(concept)
        if meta is None:
            continue  # urine_output_volume: deliberately out of scope, see module docstring
        out.append({
            "concept": concept,
            "label": meta["label"],
            "hint": meta["hint"],
            "unit": channel["unit"],
            "provenance_id": channel["provenance_id"],
        })
    return sorted(out, key=lambda item: str(item["label"]))


def _parse_observations(observations: Sequence[Mapping[str, object]], concept_index: Mapping[str, Mapping[str, object]]) -> List[_Observation]:
    parsed: List[_Observation] = []
    for row in observations:
        concept = str(row.get("concept", ""))
        if concept not in concept_index:
            raise CustomRecordError(f"unsupported canonical concept: {concept!r}")
        try:
            hours = float(row["hours_since_admission"])
            value = float(row["value"])
        except (KeyError, TypeError, ValueError) as exc:
            raise CustomRecordError("hours_since_admission and value must be numeric") from exc
        # Must be strictly > 0: the feature builder excludes any event at or
        # before intime as structural padding (feature_schema_v2.json's
        # exact_intime_event_policy) -- an hour-0 observation would silently
        # never appear in any prediction, so it is rejected upfront instead.
        if not math.isfinite(hours) or hours <= 0 or hours > MAX_HOURS_SINCE_ADMISSION:
            raise CustomRecordError(f"hours_since_admission must be greater than 0 and at most {MAX_HOURS_SINCE_ADMISSION:.0f}")
        if not math.isfinite(value):
            raise CustomRecordError("value must be a finite number")
        if concept == "glasgow_coma_scale" and value != int(value):
            raise CustomRecordError("glasgow_coma_scale must be a whole number (3-15)")
        parsed.append(_Observation(concept=concept, hours_since_admission=hours, value=value))
    return parsed


def _iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _build_support_rows(*, stay_id: str, subject_id: str, intime: datetime, support_intervals: Sequence[Mapping[str, object]]) -> List[dict]:
    """Raw support_intervals.jsonl-shaped rows from user-entered intervals.

    Boundary convention matches the frozen event dictionary exactly
    (configs/event_dict_v2.yaml: intervals.convention = CLOSED_OPEN,
    explicit_end_required = true) -- active iff start <= cutoff < end, and
    every interval must have a concrete end (see ONGOING_SUPPORT_END_HOURS
    for how "still active" is represented)."""

    rows: List[dict] = []
    for index, item in enumerate(support_intervals):
        kind = str(item.get("kind", ""))
        try:
            start_hour = float(item["start_hour"])
        except (KeyError, TypeError, ValueError) as exc:
            raise CustomRecordError("support interval start_hour must be numeric") from exc
        if not math.isfinite(start_hour) or start_hour <= 0 or start_hour > MAX_HOURS_SINCE_ADMISSION:
            raise CustomRecordError(f"support interval start_hour must be greater than 0 and at most {MAX_HOURS_SINCE_ADMISSION:.0f}")

        end_hour_raw = item.get("end_hour")
        if end_hour_raw is None:
            end_hour = ONGOING_SUPPORT_END_HOURS
        else:
            try:
                end_hour = float(end_hour_raw)
            except (TypeError, ValueError) as exc:
                raise CustomRecordError("support interval end_hour must be numeric or null") from exc
        if not math.isfinite(end_hour) or end_hour <= start_hour:
            raise CustomRecordError("support interval end_hour must be greater than start_hour")

        start_dt = intime + timedelta(hours=start_hour)
        end_dt = intime + timedelta(hours=end_hour)
        event_id = f"{stay_id}-SUPPORT-{index:04d}"

        if kind == "vasopressor":
            agent = str(item.get("agent", ""))
            if agent not in VASOPRESSOR_AGENTS:
                raise CustomRecordError(f"unsupported vasopressor agent: {agent!r} (must be one of {VASOPRESSOR_AGENTS})")
            try:
                rate = float(item["rate"])
            except (KeyError, TypeError, ValueError) as exc:
                raise CustomRecordError("vasopressor rate (ug/kg/min) must be numeric") from exc
            if not math.isfinite(rate) or rate <= 0:
                raise CustomRecordError("vasopressor rate must be a positive number")
            rows.append({
                "stay_id": stay_id, "subject_id": subject_id, "agent_key": agent,
                "interval_start": _iso(start_dt), "interval_end": _iso(end_dt),
                "support_event_id": event_id, "support_type": "VASOPRESSOR",
                "rate_value": rate, "rate_unit": "ug/kg/min",
            })
        elif kind == "ventilation":
            rows.append({
                "stay_id": stay_id, "subject_id": subject_id, "respiratory_category": "INVASIVE",
                "interval_start": _iso(start_dt), "interval_end": _iso(end_dt),
                "support_event_id": event_id, "support_type": "RESPIRATORY",
                "normalization_provenance_ref": f"{stay_id}-manual-entry-v1",
            })
        else:
            raise CustomRecordError(f"unsupported support interval kind: {kind!r} (must be 'vasopressor' or 'ventilation')")
    return rows


def build_custom_record(
    runtime: V2ServingRuntime,
    *,
    owner_user_id: str,
    patient_alias: str,
    age_years: int,
    sex_category: str,
    observations: Sequence[Mapping[str, object]],
    support_intervals: Sequence[Mapping[str, object]] = (),
) -> Mapping[str, object]:
    if not owner_user_id:
        raise CustomRecordError("owner_user_id is required")
    if runtime.ephemeral_count() >= MAX_EPHEMERAL_RECORDS:
        raise CustomRecordError("too many custom records registered this session; restart the server to clear them")
    if not patient_alias or not str(patient_alias).strip():
        raise CustomRecordError("patient_alias is required")
    if not observations:
        raise CustomRecordError("at least one observation is required")
    if len(observations) > MAX_OBSERVATIONS_PER_RECORD:
        raise CustomRecordError(f"at most {MAX_OBSERVATIONS_PER_RECORD} observations are supported per record")
    if len(support_intervals) > MAX_SUPPORT_INTERVALS_PER_RECORD:
        raise CustomRecordError(f"at most {MAX_SUPPORT_INTERVALS_PER_RECORD} support intervals are supported per record")

    concept_index = {item["concept"]: item for item in canonical_concepts(runtime.root)}
    parsed = _parse_observations(observations, concept_index)

    stay_id = f"CUSTOM-{secrets.token_hex(6).upper()}"
    subject_id = f"CUSTOM-SUBJ-{secrets.token_hex(6).upper()}"
    intime = _ANCHOR_INTIME

    support_rows = _build_support_rows(stay_id=stay_id, subject_id=subject_id, intime=intime, support_intervals=support_intervals)

    last_observation_hour = max(item.hours_since_admission for item in parsed)
    last_support_hour = max((_hours_between(intime, row["interval_end"]) for row in support_rows), default=0.0)
    outtime_hours = max(30.0, last_observation_hour + 6.0, min(last_support_hour, MAX_HOURS_SINCE_ADMISSION) + 6.0)
    outtime = intime + timedelta(hours=outtime_hours)

    events = []
    for index, item in enumerate(parsed):
        meta = concept_index[item.concept]
        # SOFA's central-nervous-system component requires an integer GCS
        # (data/synthetic/sofa.py::_validate_point, integer_required=True);
        # every other concept is a plain float.
        value_numeric: object = int(item.value) if item.concept == "glasgow_coma_scale" else float(item.value)
        events.append({
            "stay_id": stay_id,
            "subject_id": subject_id,
            "event_time": _iso(intime + timedelta(hours=item.hours_since_admission)),
            "canonical_concept": item.concept,
            "event_id": f"{stay_id}-EV-{index:04d}",
            "unit": meta["unit"],
            "provenance_id": meta["provenance_id"],
            "value_numeric": value_numeric,
            "event_kind": "point",
        })

    statics_row = {
        "stay_id": stay_id,
        "subject_id": subject_id,
        "intime": _iso(intime),
        "outtime": _iso(outtime),
        "age_years": int(age_years),
        "sex_category": str(sex_category),
        "cardiac_condition_group": "CUSTOM_RECORD",
        "patient_alias": str(patient_alias).strip(),
    }

    stay = RetainedICUStay(subject_id=subject_id, stay_id=stay_id, intime=intime, outtime=outtime)
    rows = generate_prediction_rows_for_stay(stay)
    legal_cutoffs = tuple(_iso(row.prediction_time) for row in rows)

    runtime.register_ephemeral_stay(
        stay_id=stay_id, subject_id=subject_id, statics_row=statics_row, events=events,
        legal_cutoffs=legal_cutoffs, owner_user_id=owner_user_id,
    )
    if support_rows:
        runtime.register_ephemeral_support(stay_id=stay_id, support_rows=support_rows)

    return _describe_record(runtime, stay_id=stay_id, statics=statics_row, entry_legal_cutoffs=legal_cutoffs, support_rows=support_rows)


def _hours_between(intime: datetime, iso_timestamp: str) -> float:
    dt = datetime.fromisoformat(iso_timestamp.replace("Z", "+00:00"))
    return (dt - intime).total_seconds() / 3600.0


def _describe_record(
    runtime: V2ServingRuntime, *, stay_id: str, statics: Mapping[str, object], entry_legal_cutoffs: Sequence[str], support_rows: Sequence[Mapping[str, object]],
) -> Mapping[str, object]:
    """Builds the response payload: identity + a genuine, non-fake-confidence
    readiness summary. Where possible this reuses the real prediction path
    (running one prediction at the first legal cutoff) rather than
    re-deriving bin/observation math separately, so the numbers reported here
    can never drift from what /predict itself would say."""

    concept_index = canonical_concepts(runtime.root)
    events = runtime._events_by_stay.get(stay_id, [])
    covered = sorted({row["canonical_concept"] for row in events})
    hours_entered = sorted({_hours_between(_ANCHOR_INTIME, row["event_time"]) for row in events})

    warnings: List[str] = []
    data_quality = None
    if entry_legal_cutoffs:
        try:
            prediction = runtime.predict(stay_id, entry_legal_cutoffs[0])
            mask = prediction.feature_row["observation_mask"]
            padding = prediction.feature_row.get("padding_mask", [False] * len(mask))
            total_bins = len(mask)
            observed_bins = sum(1 for bin_row, p in zip(mask, padding) if not p and any(bin_row))
            data_quality = {"observed_bins": observed_bins, "total_bins": total_bins}
        except Exception:  # noqa: BLE001 - readiness reporting must never itself fail record creation
            data_quality = None

    if hours_entered and not any(0 < h <= 24.0 for h in hours_entered):
        warnings.append(
            "No observations fall within the first legal cutoff's 48-hour lookback window "
            "(0-24h since admission) -- the first prediction will show low data completeness."
        )
    concept_index_names = {c["concept"] for c in concept_index}
    missing_concepts = sorted(concept_index_names - set(covered))
    if missing_concepts:
        warnings.append(f"No observations at all for: {', '.join(missing_concepts)} -- these channels will show as missing at every cutoff.")

    sofa_observed = [name for name, concepts in _SOFA_COMPONENT_CONCEPTS.items() if set(concepts) & set(covered)]
    sofa_missing = [name for name in _SOFA_COMPONENT_CONCEPTS if name not in sofa_observed]

    has_vasopressor = any(r["support_type"] == "VASOPRESSOR" for r in support_rows)
    has_ventilation = any(r["support_type"] == "RESPIRATORY" for r in support_rows)
    if not has_vasopressor and not has_ventilation:
        warnings.append("No organ-support state entered -- every cutoff is modeled as no active vasopressor or invasive ventilation.")

    readiness_summary = {
        "observations_entered": len(events),
        "concepts_represented": len(covered),
        "concepts_total": len(concept_index_names),
        "earliest_observation_hour": hours_entered[0] if hours_entered else None,
        "latest_observation_hour": hours_entered[-1] if hours_entered else None,
        "legal_cutoffs": len(entry_legal_cutoffs),
        "first_cutoff_observed_bins": data_quality["observed_bins"] if data_quality else None,
        "first_cutoff_total_bins": data_quality["total_bins"] if data_quality else None,
        "sofa_components_observed": sofa_observed,
        "sofa_components_missing": sofa_missing,
        "support_state_entered": {"vasopressor": has_vasopressor, "invasive_ventilation": has_ventilation},
        "missing_concepts": missing_concepts,
        "urine_output_supported": False,
    }

    return {
        "stay_id": stay_id,
        "subject_id": statics["subject_id"],
        "patient_alias": statics.get("patient_alias", stay_id),
        "cardiac_condition_group": "CUSTOM_RECORD",
        "age_years": statics.get("age_years"),
        "sex_category": statics.get("sex_category"),
        "intime": statics.get("intime"),
        "outtime": statics.get("outtime"),
        "legal_cutoffs": list(entry_legal_cutoffs),
        "n_legal_cutoffs": len(entry_legal_cutoffs),
        "data_readiness": "READY",
        "warnings": warnings,
        "concept_coverage": {"observed": len(covered), "total": len(concept_index_names), "concepts": covered},
        "readiness_summary": readiness_summary,
        "source": "CUSTOM_RECORD_DIRECT_ENTRY",
    }


def get_custom_record(runtime: V2ServingRuntime, stay_id: str) -> Optional[Mapping[str, object]]:
    entry = runtime.ephemeral_entry(stay_id)
    if entry is None:
        return None
    statics = runtime._statics_by_stay.get(stay_id, {})
    support_rows = runtime._supports_by_stay.get(stay_id, [])
    return _describe_record(runtime, stay_id=stay_id, statics=statics, entry_legal_cutoffs=entry["legal_cutoffs"], support_rows=support_rows)
