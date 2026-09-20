from datetime import datetime, timedelta, timezone
from pathlib import Path

from data.synthetic.sofa import SOFABindings, VasoactiveExposure, load_sofa_spec
from data.synthetic.sofa_provider import InMemorySOFASupportProvider, VentilationInterval

ROOT = Path(__file__).resolve().parents[1]
SPEC = load_sofa_spec(ROOT / "configs/synthetic/sofa_spec_v1.json")
CUTOFF = datetime(2100, 1, 2, 12, tzinfo=timezone.utc)


def iso(value):
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


STAY = {
    "subject_id": "subject-A", "stay_id": "stay-A",
    "intime": iso(CUTOFF - timedelta(hours=48)),
    "outtime": iso(CUTOFF + timedelta(hours=72)),
}
CONTRACT_VERSION = "ENGINEERING_SYNTHETIC_FIXTURE_SUPPORT_V1"
CONTRACT_HASH = "a" * 64
BINDINGS = SOFABindings(
    processed_schema_version="synthetic_processed_timeline_v1",
    processed_schema_sha256="b" * 64,
    concept_map_version="synthetic_concept_map_v1",
    concept_map_sha256="c" * 64,
    provenance_sha256="d" * 64,
    timeline_source_version="fixture_timeline_v1",
    timeline_source_sha256="e" * 64,
    support_contract_version=CONTRACT_VERSION,
    support_contract_sha256=CONTRACT_HASH,
)

UNITS = {
    "pao2": "mmHg", "fio2": "fraction", "platelet_count": "10^9/L",
    "bilirubin_total": "mg/dL", "mean_arterial_pressure": "mmHg",
    "glasgow_coma_scale": "score", "creatinine": "mg/dL",
    "urine_output_volume": "mL",
}


def event(concept, value, *, when=None, event_id=None, stay_id="stay-A", subject_id="subject-A", unit=None):
    when = when or CUTOFF
    return {
        "event_id": event_id or concept + "-event", "subject_id": subject_id,
        "stay_id": stay_id, "event_time": iso(when),
        "interval_start": None, "interval_end": None,
        "canonical_concept": concept, "value_numeric": value,
        "unit": unit or UNITS[concept], "event_kind": "point",
    }


def urine(value, start, end, ref):
    return {
        "event_id": ref, "subject_id": "subject-A", "stay_id": "stay-A",
        "event_time": iso(end), "interval_start": iso(start),
        "interval_end": iso(end), "canonical_concept": "urine_output_volume",
        "value_numeric": value, "unit": "mL", "event_kind": "quantity_interval",
    }


def provider(*, ventilated=False, vaso=(), known=True, contract_hash=CONTRACT_HASH, extra_vent=()):
    intervals = list(extra_vent)
    if ventilated:
        intervals.append(VentilationInterval(
            "stay-A", CUTOFF - timedelta(hours=2), CUTOFF,
            "fixture-invasive-ventilation",
        ))
    return InMemorySOFASupportProvider(
        contract_version=CONTRACT_VERSION, contract_sha256=contract_hash,
        ventilation_intervals=intervals, vasoactive_exposures=vaso,
        ventilation_coverage_known=known, vasoactive_coverage_known=known,
    )


def normal_history(when=None):
    when = when or CUTOFF
    return [
        event("pao2", 100, when=when, event_id="pao2-normal"),
        event("fio2", .21, when=when, event_id="fio2-normal"),
        event("platelet_count", 200, when=when),
        event("bilirubin_total", 1.0, when=when),
        event("mean_arterial_pressure", 75, when=when),
        event("glasgow_coma_scale", 15, when=when),
        event("creatinine", 1.0, when=when),
    ]


def vaso(agent, rate, *, start=None, end=None, stay_id="stay-A", ref=None):
    return VasoactiveExposure(
        stay_id=stay_id, agent=agent, rate=rate, unit="ug/kg/min",
        interval_start=start or CUTOFF - timedelta(hours=2),
        interval_end=end or CUTOFF - timedelta(hours=1),
        source_ref=ref or agent + "-exposure",
    )
