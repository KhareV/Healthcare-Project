"""Custom-record intake: let a user type in their own vitals/labs and get a
real prediction through the exact same frozen V2 pipeline used for the demo
subjects, without touching any frozen scientific artifact.

Ephemeral, in-memory-only, process-lifetime storage (see
V2ServingRuntime.register_ephemeral_stay) -- restarting the server discards
every custom record. This is intentional: it is a demo feature, not a
clinical data store, and nothing here is written to any git-tracked path.

Deliberately out of scope for this pass: urine_output_volume (the SOFA
renal-component computation wants an interval-shaped event distinct from the
point-in-time shape the feature builder wants for the same concept; renal
SOFA still works from creatinine alone) and organ-support entry
(vasopressor/ventilation) -- a custom record is always modeled as "no organ
support observed," which is a real, valid state, just not representable for
every real patient in this first pass.
"""

from __future__ import annotations

import json
import math
import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Mapping, Sequence

from data.timestamps import RetainedICUStay, generate_prediction_rows_for_stay
from serving.v2.runtime import FEATURE_SCHEMA_PATH, V2ServingRuntime

MAX_OBSERVATIONS_PER_RECORD = 200
MAX_EPHEMERAL_RECORDS = 500
# The legal-cutoff grid (data.timestamps: K_MIN..K_MAX) never extends past
# icu_elapsed_hours = 24 + 6*11 = 90h regardless of how far outtime is
# pushed out, so an observation timed later than that could never be visible
# to any servable cutoff -- capping intake here avoids silently accepting
# data that would otherwise just be invisible forever.
MAX_HOURS_SINCE_ADMISSION = 90.0

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
            continue  # organ-support channels: out of scope for direct entry, see module docstring
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


def build_custom_record(
    runtime: V2ServingRuntime,
    *,
    patient_alias: str,
    age_years: int,
    sex_category: str,
    observations: Sequence[Mapping[str, object]],
) -> Mapping[str, object]:
    if runtime.ephemeral_count() >= MAX_EPHEMERAL_RECORDS:
        raise CustomRecordError("too many custom records registered this session; restart the server to clear them")
    if not patient_alias or not str(patient_alias).strip():
        raise CustomRecordError("patient_alias is required")
    if not observations:
        raise CustomRecordError("at least one observation is required")
    if len(observations) > MAX_OBSERVATIONS_PER_RECORD:
        raise CustomRecordError(f"at most {MAX_OBSERVATIONS_PER_RECORD} observations are supported per record")

    concept_index = {item["concept"]: item for item in canonical_concepts(runtime.root)}

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

    stay_id = f"CUSTOM-{secrets.token_hex(6).upper()}"
    subject_id = f"CUSTOM-SUBJ-{secrets.token_hex(6).upper()}"

    last_hour = max(item.hours_since_admission for item in parsed)
    outtime_hours = max(30.0, last_hour + 6.0)
    intime = _ANCHOR_INTIME
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
            "event_time": (intime + timedelta(hours=item.hours_since_admission)).strftime("%Y-%m-%dT%H:%M:%SZ"),
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
        "intime": intime.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "outtime": outtime.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "age_years": int(age_years),
        "sex_category": str(sex_category),
        "cardiac_condition_group": "CUSTOM_RECORD",
        "patient_alias": str(patient_alias).strip(),
    }

    stay = RetainedICUStay(subject_id=subject_id, stay_id=stay_id, intime=intime, outtime=outtime)
    rows = generate_prediction_rows_for_stay(stay)
    legal_cutoffs = tuple(row.prediction_time.strftime("%Y-%m-%dT%H:%M:%SZ") for row in rows)

    runtime.register_ephemeral_stay(
        stay_id=stay_id, subject_id=subject_id, statics_row=statics_row, events=events, legal_cutoffs=legal_cutoffs,
    )

    covered = sorted({item.concept for item in parsed})

    # Legal cutoffs are structurally guaranteed once any observation exists
    # (the outtime floor above always makes the first grid cutoff at +24h
    # legal), so "readiness" here is never a fake pass/fail gate -- it is
    # always READY, with genuine structural warnings surfaced instead of a
    # blocking verdict. The model is explicitly designed to tolerate sparse/
    # missing bins (the same observation_mask/data_quality machinery every
    # other prediction already reports), so a sparse custom record is a real,
    # valid, low-completeness case -- not an error.
    first_cutoff_hours = 24.0
    warnings: list = []
    if not any(0 < item.hours_since_admission <= first_cutoff_hours for item in parsed):
        warnings.append(
            f"No observations fall within the first legal cutoff's 48-hour lookback window "
            f"(0-{first_cutoff_hours:.0f}h since admission) -- the first prediction will show low data completeness."
        )
    if len(covered) < len(concept_index):
        missing = sorted(set(concept_index) - set(covered))
        warnings.append(f"No observations at all for: {', '.join(missing)} -- these channels will show as missing at every cutoff.")

    return {
        "stay_id": stay_id,
        "subject_id": subject_id,
        "patient_alias": str(patient_alias).strip(),
        "cardiac_condition_group": "CUSTOM_RECORD",
        "age_years": int(age_years),
        "sex_category": str(sex_category),
        "intime": statics_row["intime"],
        "outtime": statics_row["outtime"],
        "legal_cutoffs": list(legal_cutoffs),
        "n_legal_cutoffs": len(legal_cutoffs),
        "data_readiness": "READY",
        "warnings": warnings,
        "concept_coverage": {"observed": len(covered), "total": len(concept_index), "concepts": covered},
        "source": "CUSTOM_RECORD_DIRECT_ENTRY",
    }


def get_custom_record(runtime: V2ServingRuntime, stay_id: str) -> Mapping[str, object] | None:
    entry = runtime.ephemeral_entry(stay_id)
    if entry is None:
        return None
    statics = runtime._statics_by_stay.get(stay_id, {})
    events = runtime._events_by_stay.get(stay_id, [])
    covered = sorted({row["canonical_concept"] for row in events})
    concept_index = canonical_concepts(runtime.root)
    return {
        "stay_id": stay_id,
        "subject_id": entry["subject_id"],
        "patient_alias": statics.get("patient_alias", stay_id),
        "cardiac_condition_group": "CUSTOM_RECORD",
        "age_years": statics.get("age_years"),
        "sex_category": statics.get("sex_category"),
        "intime": statics.get("intime"),
        "outtime": statics.get("outtime"),
        "legal_cutoffs": list(entry["legal_cutoffs"]),
        "n_legal_cutoffs": len(entry["legal_cutoffs"]),
        "data_readiness": "READY",
        "concept_coverage": {"observed": len(covered), "total": len(concept_index), "concepts": covered},
        "source": "CUSTOM_RECORD_DIRECT_ENTRY",
    }
