"""Phase-6 adapters: explicit fixture support and Pulkit CurrentSOFAProvider."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import hashlib
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence, Tuple

from serving.recovery import CurrentSOFAState
from labels.event_dict import EventDictionary
from labels.support_state import ExecutionMode
from labels.ventilation_state import (
    InvasiveVentilationState, NormalizedVentilationInterval,
    query_invasive_ventilation_state,
)

from .processed_manifest import validate_processed_manifest
from .provenance import sha256_file
from .sofa import (
    LoadedSOFASpec, SOFABindings, SOFADependencyUnavailable, SOFAError, SOFASupportProvider,
    SupportState, SupportWindowEvidence, VasoactiveExposure,
    VentilationObservation, load_sofa_spec, sofa_at,
    validate_spec_against_concept_map,
)
from .validation import load_jsonl, parse_utc


def _instant(value: datetime) -> datetime:
    if value.utcoffset() is None:
        raise SOFAError("support timestamps must be timezone-aware")
    return value.astimezone(timezone.utc)


@dataclass(frozen=True)
class VentilationInterval:
    stay_id: object
    interval_start: datetime
    interval_end: datetime
    source_ref: str


class InMemorySOFASupportProvider(SOFASupportProvider):
    """Explicit known-coverage provider for engineering fixtures only.

    This consumes already-normalized intervals/exposures.  It does not derive
    Pulkit's support states or define Phase-9 source mappings.
    """

    def __init__(
        self, *, contract_version: str, contract_sha256: str,
        ventilation_intervals: Iterable[VentilationInterval] = (),
        vasoactive_exposures: Iterable[VasoactiveExposure] = (),
        ventilation_coverage_known: bool = True,
        vasoactive_coverage_known: bool = True,
    ) -> None:
        self.contract_version = contract_version
        self.contract_sha256 = contract_sha256
        self.ventilation_intervals = tuple(ventilation_intervals)
        self.vasoactive_exposures = tuple(vasoactive_exposures)
        self.ventilation_coverage_known = ventilation_coverage_known
        self.vasoactive_coverage_known = vasoactive_coverage_known

    def evidence(
        self, *, stay_id: object, window_start: datetime, cutoff: datetime,
        respiratory_times: Tuple[datetime, ...],
    ) -> SupportWindowEvidence:
        observations = []
        for when in respiratory_times:
            active = []
            for interval in self.ventilation_intervals:
                if interval.stay_id != stay_id:
                    continue
                left, right = _instant(interval.interval_start), _instant(interval.interval_end)
                if left > cutoff:
                    continue
                if right > cutoff:
                    raise SOFAError("fixture support interval exposes a future end")
                # The fixture interval is an as-of-cutoff observation span, not
                # a claimed clinical cessation interval.  Its right endpoint
                # is therefore observed/inclusive after cutoff truncation.
                if left <= when <= right:
                    active.append(interval.source_ref)
            observations.append(VentilationObservation(
                observation_time=when,
                state=SupportState.ACTIVE if active else SupportState.INACTIVE,
                source_ref="+".join(sorted(active)) if active else "KNOWN_INACTIVE@" + when.isoformat(),
            ))
        vaso = tuple(
            item for item in self.vasoactive_exposures
            if item.interval_end > window_start and item.interval_start <= cutoff
            and item.stay_id == stay_id
        )
        return SupportWindowEvidence(
            stay_id=stay_id, window_start=window_start, cutoff=cutoff,
            ventilation=tuple(observations), vasoactive=vaso,
            ventilation_coverage_known=self.ventilation_coverage_known,
            vasoactive_coverage_known=self.vasoactive_coverage_known,
            contract_version=self.contract_version,
            contract_sha256=self.contract_sha256,
        )


class PulkitStateSOFASupportProvider(SOFASupportProvider):
    """Adapter that reuses Pulkit's ventilation state engine.

    Vasoactive dose segments remain an injected Phase-9 dependency because
    Pulkit's endpoint state is intentionally binary and cannot represent the
    dose categories required by standard SOFA.
    """

    def __init__(
        self, *, event_dictionary: EventDictionary,
        execution_mode: ExecutionMode,
        ventilation_intervals: Iterable[NormalizedVentilationInterval],
        vasoactive_exposures: Iterable[VasoactiveExposure],
        vasoactive_dose_contract_version: str,
        vasoactive_dose_contract_sha256: str,
        vasoactive_coverage_known: bool,
    ) -> None:
        if not vasoactive_dose_contract_version:
            raise SOFAError("vasoactive dose contract version is required")
        if len(vasoactive_dose_contract_sha256) != 64:
            raise SOFAError("vasoactive dose contract hash is invalid")
        self.event_dictionary = event_dictionary
        self.execution_mode = execution_mode
        self.ventilation_intervals = tuple(ventilation_intervals)
        self.vasoactive_exposures = tuple(vasoactive_exposures)
        self.vasoactive_coverage_known = vasoactive_coverage_known
        self.contract_version = (
            event_dictionary.event_dictionary_version + "+" +
            event_dictionary.ventilation.state_rule_version + "+" +
            vasoactive_dose_contract_version
        )
        payload = json.dumps({
            "event_dictionary_sha256": event_dictionary.config_sha256,
            "ventilation_state_rule": event_dictionary.ventilation.state_rule_version,
            "vasoactive_dose_contract_sha256": vasoactive_dose_contract_sha256,
        }, sort_keys=True, separators=(",", ":"))
        self.contract_sha256 = hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def evidence(
        self, *, stay_id: object, window_start: datetime, cutoff: datetime,
        respiratory_times: Tuple[datetime, ...],
    ) -> SupportWindowEvidence:
        # Full generated cessation bounds are used internally only to answer
        # state-as-of queries; the bound itself is never returned to features
        # or serving before it becomes observable.
        ventilation = []
        for when in respiratory_times:
            result = query_invasive_ventilation_state(
                self.ventilation_intervals, stay_id=stay_id, cutoff=when,
                event_dictionary=self.event_dictionary,
                execution_mode=self.execution_mode,
            )
            ventilation.append(VentilationObservation(
                observation_time=when,
                state=(SupportState.ACTIVE if result.invasive_state is InvasiveVentilationState.ON else SupportState.INACTIVE),
                source_ref=("+".join(result.supporting_invasive_interval_refs) if result.supporting_invasive_interval_refs else "PULKIT_KNOWN_INACTIVE@" + when.isoformat()),
            ))
        vasoactive = tuple(
            VasoactiveExposure(
                stay_id=item.stay_id, agent=item.agent, rate=item.rate, unit=item.unit,
                interval_start=item.interval_start, interval_end=min(item.interval_end, cutoff),
                source_ref=item.source_ref,
            )
            for item in self.vasoactive_exposures
            if item.stay_id == stay_id and item.interval_end > window_start
            and item.interval_start < cutoff
        )
        return SupportWindowEvidence(
            stay_id=stay_id, window_start=window_start, cutoff=cutoff,
            ventilation=tuple(ventilation), vasoactive=vasoactive,
            ventilation_coverage_known=True,
            vasoactive_coverage_known=self.vasoactive_coverage_known,
            contract_version=self.contract_version,
            contract_sha256=self.contract_sha256,
        )


class SyntheticCurrentSOFAProvider:
    """Storage-backed adapter implementing Pulkit's CurrentSOFAProvider seam."""

    def __init__(
        self, *, manifest_path: Path, root: Path, sofa_spec_path: Path,
        support_provider: SOFASupportProvider,
        expected_sofa_spec_sha256: Optional[str] = None,
        legal_cutoffs: Optional[Mapping[object, Sequence[str]]] = None,
    ) -> None:
        manifest = validate_processed_manifest(manifest_path, root)
        self.spec: LoadedSOFASpec = load_sofa_spec(sofa_spec_path)
        if expected_sofa_spec_sha256 is not None and self.spec.sha256 != expected_sofa_spec_sha256:
            raise SOFAError("SOFA-spec hash mismatch")
        artifacts = {item["logical_name"]: item for item in manifest["artifacts"]}
        concept_map = json.loads((root / manifest["concept_map_path"]).read_text())
        validate_spec_against_concept_map(self.spec, concept_map)
        timeline_artifact = artifacts["canonical_timeline"]
        static_artifact = artifacts["canonical_statics"]
        self.history = tuple(load_jsonl(root / timeline_artifact["repository_relative_path"]))
        statics = load_jsonl(root / static_artifact["repository_relative_path"])
        self.stays = {item["stay_id"]: item for item in statics}
        self.support_provider = support_provider
        self.legal_cutoffs = None if legal_cutoffs is None else {
            key: frozenset(values) for key, values in legal_cutoffs.items()
        }
        self.bindings = SOFABindings(
            processed_schema_version=manifest["processed_schema_version"],
            processed_schema_sha256=manifest["processed_schema_sha256"],
            concept_map_version=manifest["concept_map_version"],
            concept_map_sha256=manifest["concept_map_sha256"],
            provenance_sha256=manifest["provenance_sha256"],
            timeline_source_version=timeline_artifact["schema_version"],
            timeline_source_sha256=timeline_artifact["sha256"],
            support_contract_version=getattr(support_provider, "contract_version", ""),
            support_contract_sha256=getattr(support_provider, "contract_sha256", ""),
        )
        self.bindings.validate()

    def score(self, *, stay_id: object, cutoff: str):
        if stay_id not in self.stays:
            raise SOFAError("unknown stay_id")
        if self.legal_cutoffs is not None and cutoff not in self.legal_cutoffs.get(stay_id, frozenset()):
            raise SOFAError("requested serving cutoff is not a legal project cutoff")
        return sofa_at(
            self.history, self.stays[stay_id], cutoff, spec=self.spec,
            bindings=self.bindings, support_provider=self.support_provider,
        )

    def current_sofa(self, *, stay_id: object, prediction_time: str) -> CurrentSOFAState:
        result = self.score(stay_id=stay_id, cutoff=prediction_time)
        state = CurrentSOFAState(
            stay_id=result.stay_id, prediction_time=prediction_time,
            value=float(result.total_score), sofa_version=result.sofa_spec_version,
            source_version=result.timeline_source_version + "+" + result.implementation_version,
            source_sha256=result.source_identity_sha256,
            component_observed=result.component_observed,
        )
        state.validate()
        return state


def bindings_from_processed_manifest(
    manifest_path: Path, root: Path, *, support_contract_version: str,
    support_contract_sha256: str,
) -> SOFABindings:
    manifest = validate_processed_manifest(manifest_path, root)
    timeline = next(item for item in manifest["artifacts"] if item["logical_name"] == "canonical_timeline")
    return SOFABindings(
        processed_schema_version=manifest["processed_schema_version"],
        processed_schema_sha256=manifest["processed_schema_sha256"],
        concept_map_version=manifest["concept_map_version"],
        concept_map_sha256=manifest["concept_map_sha256"],
        provenance_sha256=manifest["provenance_sha256"],
        timeline_source_version=timeline["schema_version"],
        timeline_source_sha256=timeline["sha256"],
        support_contract_version=support_contract_version,
        support_contract_sha256=support_contract_sha256,
    )
