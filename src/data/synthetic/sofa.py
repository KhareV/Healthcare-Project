"""One deterministic, cutoff-safe six-component SOFA implementation.

The versioned spec is intentionally draft until its rolling-window/pairing
rules and the Phase-9 support contract are reviewed.  The engine is complete
for explicit engineering evidence and fails closed when support provenance is
unavailable; it does not authorize final synthetic scoring.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Protocol, Sequence, Tuple

from .provenance import sha256_file
from .validation import parse_utc


COMPONENTS = (
    "respiratory", "coagulation", "liver", "cardiovascular",
    "central_nervous_system", "renal",
)
SOFA_IMPLEMENTATION_VERSION = "synthetic_sofa_at_v1"


class SOFAError(ValueError):
    """SOFA evidence or provenance violates the frozen interface."""


class SOFADependencyUnavailable(SOFAError):
    """A pipeline dependency is unavailable, not clinically unobserved."""


class SupportState(str, Enum):
    ACTIVE = "ACTIVE"
    INACTIVE = "INACTIVE"


@dataclass(frozen=True)
class VentilationObservation:
    observation_time: datetime
    state: SupportState
    source_ref: str


@dataclass(frozen=True)
class VasoactiveExposure:
    stay_id: object
    agent: str
    rate: float
    unit: str
    interval_start: datetime
    interval_end: datetime
    source_ref: str


@dataclass(frozen=True)
class SupportWindowEvidence:
    stay_id: object
    window_start: datetime
    cutoff: datetime
    ventilation: Tuple[VentilationObservation, ...]
    vasoactive: Tuple[VasoactiveExposure, ...]
    ventilation_coverage_known: bool
    vasoactive_coverage_known: bool
    contract_version: str
    contract_sha256: str


class SOFASupportProvider(Protocol):
    def evidence(
        self, *, stay_id: object, window_start: datetime, cutoff: datetime,
        respiratory_times: Tuple[datetime, ...],
    ) -> SupportWindowEvidence:
        """Return cutoff-truncated, versioned support evidence."""


@dataclass(frozen=True)
class SOFABindings:
    processed_schema_version: str
    processed_schema_sha256: str
    concept_map_version: str
    concept_map_sha256: str
    provenance_sha256: str
    timeline_source_version: str
    timeline_source_sha256: str
    support_contract_version: str
    support_contract_sha256: str

    def validate(self) -> None:
        for field in (
            "processed_schema_sha256", "concept_map_sha256", "provenance_sha256",
            "timeline_source_sha256", "support_contract_sha256",
        ):
            value = getattr(self, field)
            if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise SOFAError(field + " must be a lowercase SHA-256")
        for field in (
            "processed_schema_version", "concept_map_version",
            "timeline_source_version", "support_contract_version",
        ):
            if not getattr(self, field):
                raise SOFAError(field + " is required")

    @property
    def source_identity_sha256(self) -> str:
        payload = json.dumps(self.__dict__, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class LoadedSOFASpec:
    payload: Mapping[str, Any]
    path: Path
    sha256: str

    @property
    def version(self) -> str:
        return str(self.payload["specification_version"])


@dataclass(frozen=True)
class ComponentResult:
    component_name: str
    score: int
    observed: bool
    evidence_refs: Tuple[str, ...]
    source_timestamps: Tuple[str, ...]
    specification_version: str

    def validate(self) -> None:
        if self.component_name not in COMPONENTS:
            raise SOFAError("unknown SOFA component")
        if isinstance(self.score, bool) or not isinstance(self.score, int) or not 0 <= self.score <= 4:
            raise SOFAError("component score must be integer 0..4")
        if not isinstance(self.observed, bool):
            raise SOFAError("component observed flag must be boolean")
        if not self.observed and (self.score != 0 or self.evidence_refs or self.source_timestamps):
            raise SOFAError("unobserved component must use zero with no selected evidence")


@dataclass(frozen=True)
class SOFAResult:
    subject_id: object
    stay_id: object
    cutoff_time: str
    total_score: int
    components: Tuple[ComponentResult, ...]
    sofa_spec_version: str
    sofa_spec_sha256: str
    implementation_version: str
    processed_schema_version: str
    processed_schema_sha256: str
    concept_map_version: str
    concept_map_sha256: str
    provenance_sha256: str
    timeline_source_version: str
    timeline_source_sha256: str
    support_contract_version: str
    support_contract_sha256: str
    source_identity_sha256: str

    def validate(self) -> None:
        if tuple(item.component_name for item in self.components) != COMPONENTS:
            raise SOFAError("SOFA result must contain exactly the ordered six components")
        for item in self.components:
            item.validate()
        expected = sum(item.score for item in self.components)
        if isinstance(self.total_score, bool) or self.total_score != expected or not 0 <= expected <= 24:
            raise SOFAError("SOFA total must be the exact six-component sum in 0..24")

    @property
    def component_scores(self) -> Tuple[int, ...]:
        return tuple(item.score for item in self.components)

    @property
    def component_observed(self) -> Tuple[bool, ...]:
        return tuple(item.observed for item in self.components)


def load_sofa_spec(path: Path) -> LoadedSOFASpec:
    payload = json.loads(path.read_text())
    if payload.get("component_order") != list(COMPONENTS):
        raise SOFAError("SOFA spec must define exactly the canonical six components")
    if payload.get("total_score") != {"minimum": 0, "maximum": 24}:
        raise SOFAError("SOFA spec total range must be 0..24")
    window = payload.get("rolling_window", {})
    if window.get("duration_hours") != 24 or window.get("interval") != "(T-24h,T]" or window.get("left_inclusive") is not False or window.get("right_inclusive") is not True:
        raise SOFAError("unsupported SOFA rolling-window contract")
    missing = payload.get("missing_component_convention", {})
    if missing.get("score") != 0 or missing.get("observed") is not False or missing.get("majority_or_fifty_percent_exclusion_rule") is not False:
        raise SOFAError("unsupported SOFA missing-component convention")
    for section in ("respiratory", "coagulation", "liver", "central_nervous_system"):
        scores = payload.get(section, {}).get("scores", [])
        if sorted(item.get("score") for item in scores) != [0, 1, 2, 3, 4]:
            raise SOFAError(section + " must define scores 0..4 exactly once")
    if sorted(item.get("score") for item in payload["renal"]["creatinine_scores"]) != [0, 1, 2, 3, 4]:
        raise SOFAError("renal creatinine rules must define scores 0..4")
    return LoadedSOFASpec(payload=payload, path=path, sha256=sha256_file(path))


def validate_spec_against_concept_map(spec: LoadedSOFASpec, concept_map: Mapping[str, Any]) -> None:
    """Prove that SOFA names, units, domains and event kinds match Phase 5."""
    p = spec.payload
    expected = {
        "pao2": (p["respiratory"]["concepts"]["pao2"], "point"),
        "fio2": (p["respiratory"]["concepts"]["fio2"], "point"),
        p["coagulation"]["concept"]: ({"unit": p["coagulation"]["unit"], "valid_domain": p["coagulation"]["valid_domain"]}, "point"),
        p["liver"]["concept"]: ({"unit": p["liver"]["unit"], "valid_domain": p["liver"]["valid_domain"]}, "point"),
        p["cardiovascular"]["map_concept"]: ({"unit": p["cardiovascular"]["map_unit"], "valid_domain": p["cardiovascular"]["map_valid_domain"]}, "point"),
        p["central_nervous_system"]["concept"]: ({"unit": p["central_nervous_system"]["unit"], "valid_domain": p["central_nervous_system"]["valid_domain"]}, "point"),
        p["renal"]["creatinine_concept"]: ({"unit": p["renal"]["creatinine_unit"], "valid_domain": p["renal"]["creatinine_valid_domain"]}, "point"),
        p["renal"]["urine_concept"]: ({"unit": p["renal"]["urine_unit"], "valid_domain": p["renal"]["urine_valid_domain"]}, "quantity_interval"),
    }
    mappings = concept_map.get("mappings", {})
    if set(expected) - set(mappings):
        raise SOFAError("Phase-5 concept map lacks mandatory SOFA dependencies")
    for concept, (source, kind) in expected.items():
        mapping = mappings[concept]
        if mapping.get("canonical_concept") != concept or mapping.get("canonical_unit") != source["unit"] or mapping.get("domain") != source["valid_domain"] or mapping.get("event_kind") != kind:
            raise SOFAError("SOFA spec/Phase-5 concept-map mismatch for " + concept)


def _utc(value: object, field: str) -> datetime:
    if isinstance(value, datetime):
        result = value
    elif isinstance(value, str):
        result = parse_utc(value)
    else:
        raise SOFAError(field + " must be an ISO timestamp or datetime")
    if result.utcoffset() is None:
        raise SOFAError(field + " must be timezone-aware")
    return result.astimezone(timezone.utc)


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _threshold_score(value: float, rules: Sequence[Mapping[str, Any]]) -> int:
    for rule in rules:
        if rule.get("otherwise") is True:
            return int(rule["score"])
        if "less_than" in rule and value < float(rule["less_than"]):
            return int(rule["score"])
        if "greater_than" in rule and value > float(rule["greater_than"]):
            return int(rule["score"])
        if "greater_than_or_equal" in rule and value >= float(rule["greater_than_or_equal"]):
            return int(rule["score"])
        if rule.get("any_positive_rate") is True and value > 0:
            return int(rule["score"])
    raise SOFAError("threshold table has no matching rule")


def _missing(name: str, spec: LoadedSOFASpec) -> ComponentResult:
    return ComponentResult(name, 0, False, (), (), spec.version)


def _selected(name: str, candidates: Sequence[Tuple[int, str, datetime]], spec: LoadedSOFASpec) -> ComponentResult:
    if not candidates:
        return _missing(name, spec)
    worst = max(item[0] for item in candidates)
    tied = sorted((ref, when) for score, ref, when in candidates if score == worst)
    return ComponentResult(name, worst, True, tuple(item[0] for item in tied), tuple(_iso(item[1]) for item in tied), spec.version)


def _validate_point(
    row: Mapping[str, Any], unit: str,
    valid_domain: Optional[Sequence[float]] = None,
    integer_required: bool = False,
) -> Tuple[float, datetime, str]:
    if row.get("event_kind") != "point":
        raise SOFAError("SOFA point concept has incompatible event_kind")
    if row.get("unit") != unit:
        raise SOFAError("invalid canonical unit for " + str(row.get("canonical_concept")))
    value = row.get("value_numeric")
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise SOFAError("SOFA input value must be finite numeric")
    if integer_required and not isinstance(value, int):
        raise SOFAError("SOFA input requires an integer canonical value")
    if valid_domain is not None and not float(valid_domain[0]) <= float(value) <= float(valid_domain[1]):
        raise SOFAError("SOFA input value violates the Phase-5 canonical domain")
    ref = row.get("event_id")
    if not isinstance(ref, str) or not ref:
        raise SOFAError("SOFA evidence event_id is required")
    return float(value), _utc(row.get("event_time"), "event_time"), ref


def _point_candidates(
    rows: Sequence[Mapping[str, Any]], concept: str, unit: str,
    rules: Sequence[Mapping[str, Any]], name: str, spec: LoadedSOFASpec,
    valid_domain: Optional[Sequence[float]] = None,
    integer_required: bool = False,
) -> ComponentResult:
    candidates = []
    for row in rows:
        if row.get("canonical_concept") == concept:
            value, when, ref = _validate_point(row, unit, valid_domain, integer_required)
            candidates.append((_threshold_score(value, rules), ref, when))
    return _selected(name, candidates, spec)


def _validate_support(evidence: SupportWindowEvidence, *, stay_id: object, start: datetime, cutoff: datetime, bindings: SOFABindings) -> None:
    if evidence.stay_id != stay_id or evidence.window_start != start or evidence.cutoff != cutoff:
        raise SOFAError("support evidence is bound to another stay/window")
    if evidence.contract_version != bindings.support_contract_version or evidence.contract_sha256 != bindings.support_contract_sha256:
        raise SOFAError("support contract version/hash mismatch")
    if not evidence.ventilation_coverage_known or not evidence.vasoactive_coverage_known:
        raise SOFADependencyUnavailable("BLOCKED — REQUIRED SOFA DEPENDENCY UNAVAILABLE")


def _respiratory(rows: Sequence[Mapping[str, Any]], support: SupportWindowEvidence, spec: LoadedSOFASpec) -> ComponentResult:
    section = spec.payload["respiratory"]
    pao2: Dict[datetime, list] = {}
    fio2: Dict[datetime, list] = {}
    for row in rows:
        concept = row.get("canonical_concept")
        if concept not in ("pao2", "fio2"):
            continue
        source = section["concepts"][concept]
        value, when, ref = _validate_point(row, source["unit"], source["valid_domain"])
        if concept == "fio2" and value <= 0:
            raise SOFAError("FiO2 must be positive")
        (pao2 if concept == "pao2" else fio2).setdefault(when, []).append((value, ref))
    vent: Dict[datetime, VentilationObservation] = {}
    for item in support.ventilation:
        when = _utc(item.observation_time, "ventilation observation_time")
        if when in vent:
            raise SOFAError("duplicate ventilation state at one time")
        if not isinstance(item.state, SupportState) or not item.source_ref:
            raise SOFAError("invalid ventilation support evidence")
        vent[when] = item
    candidates = []
    for when in sorted(set(pao2) & set(fio2)):
        if when not in vent:
            raise SOFADependencyUnavailable("BLOCKED — RESPIRATORY SUPPORT STATE UNAVAILABLE AT P/F TIME")
        active = vent[when].state is SupportState.ACTIVE
        for p_value, p_ref in pao2[when]:
            for f_value, f_ref in fio2[when]:
                ratio = p_value / f_value
                score = 0
                for rule in section["scores"]:
                    if rule.get("otherwise") is True:
                        score = int(rule["score"]); break
                    if ratio < float(rule["ratio_less_than"]) and (not rule["invasive_respiratory_support_required"] or active):
                        score = int(rule["score"]); break
                ref = "+".join(sorted((p_ref, f_ref, vent[when].source_ref)))
                candidates.append((score, ref, when))
    return _selected("respiratory", candidates, spec)


def _cardiovascular(rows: Sequence[Mapping[str, Any]], support: SupportWindowEvidence, spec: LoadedSOFASpec) -> ComponentResult:
    section = spec.payload["cardiovascular"]
    candidates = []
    for row in rows:
        if row.get("canonical_concept") == section["map_concept"]:
            value, when, ref = _validate_point(row, section["map_unit"], section["map_valid_domain"])
            candidates.append((1 if value < float(section["map_score_one_less_than"]) else 0, ref, when))
    for item in support.vasoactive:
        if item.stay_id != support.stay_id:
            raise SOFAError("vasoactive evidence is bound to another stay")
        start = _utc(item.interval_start, "vasoactive interval_start")
        end = _utc(item.interval_end, "vasoactive interval_end")
        if end <= start or end <= support.window_start or start > support.cutoff or end > support.cutoff:
            raise SOFAError("vasoactive evidence must overlap and remain cutoff-safe for the queried window")
        effective_start = max(start, support.window_start)
        duration = (end - effective_start).total_seconds() / 60.0
        if duration < float(section["minimum_exposure_minutes"]):
            continue
        if item.unit != section["vasoactive_rate_unit"]:
            raise SOFAError("invalid vasoactive rate unit")
        if item.agent not in section["agents"]:
            raise SOFAError("unknown vasoactive agent")
        if isinstance(item.rate, bool) or not isinstance(item.rate, (int, float)) or not math.isfinite(float(item.rate)) or item.rate <= 0:
            raise SOFAError("vasoactive rate must be finite and positive")
        score = _threshold_score(float(item.rate), section["agents"][item.agent])
        candidates.append((score, item.source_ref, end))
    return _selected("cardiovascular", candidates, spec)


def _urine_candidate(rows: Sequence[Mapping[str, Any]], start: datetime, cutoff: datetime, spec: LoadedSOFASpec) -> Optional[Tuple[int, str, datetime]]:
    section = spec.payload["renal"]
    intervals = []
    for row in rows:
        if row.get("canonical_concept") != section["urine_concept"]:
            continue
        if row.get("event_kind") != "quantity_interval" or row.get("unit") != section["urine_unit"]:
            raise SOFAError("invalid urine-output event kind or unit")
        value = row.get("value_numeric")
        domain = section["urine_valid_domain"]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or not float(domain[0]) <= float(value) <= float(domain[1]):
            raise SOFAError("invalid urine-output value")
        left = _utc(row.get("interval_start"), "urine interval_start")
        right = _utc(row.get("interval_end"), "urine interval_end")
        available = _utc(row.get("event_time"), "urine event_time")
        if not left < right or right != available:
            raise SOFAError("invalid urine-output availability interval")
        if left >= start and right <= cutoff:
            intervals.append((left, right, float(value), str(row.get("event_id"))))
    intervals.sort(key=lambda item: (item[0], item[1], item[3]))
    if not intervals or intervals[0][0] != start or intervals[-1][1] != cutoff:
        return None
    cursor = start
    total = 0.0
    refs = []
    for left, right, value, ref in intervals:
        if left != cursor:
            return None
        cursor = right; total += value; refs.append(ref)
    if cursor != cutoff:
        return None
    score = _threshold_score(total, section["urine_scores"])
    return score, "+".join(refs), cutoff


def _renal(rows: Sequence[Mapping[str, Any]], start: datetime, cutoff: datetime, spec: LoadedSOFASpec) -> ComponentResult:
    section = spec.payload["renal"]
    candidates = []
    for row in rows:
        if row.get("canonical_concept") == section["creatinine_concept"]:
            value, when, ref = _validate_point(row, section["creatinine_unit"], section["creatinine_valid_domain"])
            candidates.append((_threshold_score(value, section["creatinine_scores"]), ref, when))
    urine = _urine_candidate(rows, start, cutoff, spec)
    if urine is not None:
        candidates.append(urine)
    return _selected("renal", candidates, spec)


def sofa_at(
    history: Sequence[Mapping[str, Any]], stay: Mapping[str, Any], cutoff: object,
    *, spec: LoadedSOFASpec, bindings: SOFABindings,
    support_provider: Optional[SOFASupportProvider],
) -> SOFAResult:
    """Score one stay at any within-episode cutoff; cutoff is the only horizon input."""
    bindings.validate()
    if support_provider is None:
        raise SOFADependencyUnavailable("BLOCKED — REQUIRED SOFA DEPENDENCY UNAVAILABLE")
    subject_id, stay_id = stay.get("subject_id"), stay.get("stay_id")
    if subject_id is None or stay_id is None:
        raise SOFAError("stay identity is required")
    intime = _utc(stay.get("intime"), "intime")
    outtime = _utc(stay.get("outtime"), "outtime")
    point = _utc(cutoff, "cutoff")
    if not intime <= point <= outtime:
        raise SOFAError("SOFA cutoff must be within the episode")
    start = point - timedelta(hours=24)
    relevant = []
    for row in history:
        if row.get("stay_id") != stay_id or row.get("subject_id") != subject_id:
            continue
        when = _utc(row.get("event_time"), "event_time")
        if when > point or when <= start:
            continue
        if when < intime or when > outtime:
            raise SOFAError("scoreable event lies outside episode bounds")
        relevant.append(row)
    respiratory_times = tuple(sorted({
        _utc(row.get("event_time"), "event_time") for row in relevant
        if row.get("canonical_concept") in ("pao2", "fio2")
    }))
    support = support_provider.evidence(
        stay_id=stay_id, window_start=start, cutoff=point,
        respiratory_times=respiratory_times,
    )
    _validate_support(support, stay_id=stay_id, start=start, cutoff=point, bindings=bindings)
    p = spec.payload
    parts = (
        _respiratory(relevant, support, spec),
        _point_candidates(relevant, p["coagulation"]["concept"], p["coagulation"]["unit"], p["coagulation"]["scores"], "coagulation", spec, p["coagulation"]["valid_domain"]),
        _point_candidates(relevant, p["liver"]["concept"], p["liver"]["unit"], p["liver"]["scores"], "liver", spec, p["liver"]["valid_domain"]),
        _cardiovascular(relevant, support, spec),
        _point_candidates(relevant, p["central_nervous_system"]["concept"], p["central_nervous_system"]["unit"], p["central_nervous_system"]["scores"], "central_nervous_system", spec, p["central_nervous_system"]["valid_domain"], p["central_nervous_system"]["integer_required"]),
        _renal(relevant, start, point, spec),
    )
    result = SOFAResult(
        subject_id=subject_id, stay_id=stay_id, cutoff_time=_iso(point),
        total_score=sum(item.score for item in parts), components=parts,
        sofa_spec_version=spec.version, sofa_spec_sha256=spec.sha256,
        implementation_version=SOFA_IMPLEMENTATION_VERSION,
        processed_schema_version=bindings.processed_schema_version,
        processed_schema_sha256=bindings.processed_schema_sha256,
        concept_map_version=bindings.concept_map_version,
        concept_map_sha256=bindings.concept_map_sha256,
        provenance_sha256=bindings.provenance_sha256,
        timeline_source_version=bindings.timeline_source_version,
        timeline_source_sha256=bindings.timeline_source_sha256,
        support_contract_version=bindings.support_contract_version,
        support_contract_sha256=bindings.support_contract_sha256,
        source_identity_sha256=bindings.source_identity_sha256,
    )
    result.validate()
    return result
