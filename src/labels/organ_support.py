"""Composite new-organ-support label and censoring state machine.

This module composes Phase-1/2 state artifacts. It does not derive raw
vasopressor or ventilation states, and its audit disposition vocabulary is an
internal non-final contract pending review. Real execution is fail-closed.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Iterable, Mapping, Optional, Tuple, Union

from data.support_labels import SupportLabelRecord
from data.timestamps import PredictionTimestamp, TIMESTAMP_SPEC_VERSION
from labels.event_dict import (
    COMPONENT_ONSET_EVIDENCE_VERSION,
    EventDictionary,
    EventDictionaryError,
)
from labels.support_state import ExecutionMode, SupportState, VasopressorStateResult
from labels.ventilation_state import (
    InvasiveVentilationState,
    VentilationStateResult,
)


Identifier = Union[int, str]


class OrganSupportLabelError(ValueError):
    """Raised when label inputs violate temporal or lineage contracts."""


class SupportOnsetExitTieError(OrganSupportLabelError):
    """Raised because onset exactly at ICU exit has no frozen ordering rule."""


class SupportComponent(str, Enum):
    VASOPRESSOR = "VASOPRESSOR"
    INVASIVE_VENTILATION = "INVASIVE_VENTILATION"


class TransitionKind(str, Enum):
    OFF_TO_ON = "OFF_TO_ON"
    NO_COMPONENT_OFF_TO_ON = "NO_COMPONENT_OFF_TO_ON"


class AuditDisposition(str, Enum):
    """Internal non-final reason vocabulary; not yet a frozen interchange enum."""

    POSITIVE_INITIATION = "INTERNAL_POSITIVE_INITIATION"
    NEGATIVE_FULL_FOLLOWUP = "INTERNAL_NEGATIVE_FULL_FOLLOWUP"
    CENSORED_EARLY_EXIT_NO_INITIATION = (
        "INTERNAL_CENSORED_EARLY_EXIT_NO_INITIATION"
    )
    NOT_AT_RISK_BOTH_COMPONENTS_ON = (
        "INTERNAL_NOT_AT_RISK_BOTH_COMPONENTS_ON"
    )


@dataclass(frozen=True)
class SupportLabelCutoff:
    stay_id: Identifier
    prediction_time: datetime
    outtime: datetime
    support24_full_followup_available: bool
    timestamp_spec_version: str

    @classmethod
    def from_prediction_timestamp(
        cls, row: PredictionTimestamp, *, outtime: datetime
    ) -> "SupportLabelCutoff":
        """Bind Vedant's temporal flag to the ICU outtime used for labeling."""

        if not isinstance(row, PredictionTimestamp):
            raise OrganSupportLabelError("row must be a PredictionTimestamp")
        return cls(
            stay_id=row.stay_id,
            prediction_time=row.prediction_time,
            outtime=outtime,
            support24_full_followup_available=(
                row.support24_full_followup_available
            ),
            timestamp_spec_version=row.timestamp_spec_version,
        )


@dataclass(frozen=True)
class ComponentOnsetEvidence:
    """Audit-only normalized transition evidence supplied by a state adapter."""

    stay_id: Identifier
    component: SupportComponent
    onset_time: datetime
    transition_kind: TransitionKind
    source_event_ref: str
    provenance_ref: str
    state_rule_version: str
    definition_version: str
    adapter_version: str
    event_dictionary_version: str
    event_dictionary_sha256: str
    evidence_contract_version: str
    execution_mode: ExecutionMode


@dataclass(frozen=True)
class OrganSupportLabelResult:
    stay_id: Identifier
    prediction_time: datetime
    label: Optional[int]
    organ_support_eligible: bool
    disposition: AuditDisposition
    audit_reason_vocabulary_version: str
    audit_reason_vocabulary_status: str
    vaso_on_at_t: bool
    vaso_at_risk: bool
    vent_on_at_t: bool
    vent_at_risk: bool
    support24_full_followup_available: bool
    outtime: datetime
    vasopressor_onset_time: Optional[datetime]
    vasopressor_onset_ref: Optional[str]
    ventilation_onset_time: Optional[datetime]
    ventilation_onset_ref: Optional[str]
    earliest_composite_onset_time: Optional[datetime]
    earliest_composite_onset_component: Optional[SupportComponent]
    event_dictionary_version: str
    event_dictionary_sha256: str
    vasopressor_state_rule_version: str
    vasopressor_mapping_provenance_version: str
    ventilation_state_rule_version: str
    ventilation_concept_version: str
    ventilation_adapter_version: str
    timestamp_spec_version: str
    label_rule_version: str
    execution_mode: ExecutionMode

    @property
    def key(self) -> Tuple[Identifier, datetime]:
        return self.stay_id, self.prediction_time


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


def _add_elapsed_hours(value: datetime, hours: int) -> datetime:
    delta = timedelta(hours=hours)
    if value.utcoffset() is None:
        return value + delta
    return (value.astimezone(timezone.utc) + delta).astimezone(value.tzinfo)


def _same_identifier(left: Identifier, right: Identifier) -> bool:
    return type(left) is type(right) and left == right


def _validate_cutoff(cutoff: SupportLabelCutoff, horizon_hours: int) -> None:
    if not isinstance(cutoff, SupportLabelCutoff):
        raise OrganSupportLabelError("cutoff must be SupportLabelCutoff")
    if not _valid_identifier(cutoff.stay_id):
        raise OrganSupportLabelError("cutoff stay_id must be a non-empty int or string")
    if not isinstance(cutoff.prediction_time, datetime):
        raise OrganSupportLabelError("prediction_time must be a datetime")
    if not isinstance(cutoff.outtime, datetime):
        raise OrganSupportLabelError("outtime must be a datetime")
    if (cutoff.prediction_time.utcoffset() is None) != (
        cutoff.outtime.utcoffset() is None
    ):
        raise OrganSupportLabelError(
            "prediction_time and outtime must use consistent timezone awareness"
        )
    if _instant(cutoff.outtime) <= _instant(cutoff.prediction_time):
        raise OrganSupportLabelError("outtime must be after prediction_time")
    if not isinstance(cutoff.support24_full_followup_available, bool):
        raise OrganSupportLabelError("temporal follow-up flag must be boolean")
    expected = _instant(cutoff.outtime) >= _instant(
        _add_elapsed_hours(cutoff.prediction_time, horizon_hours)
    )
    if cutoff.support24_full_followup_available != expected:
        raise OrganSupportLabelError(
            "support24_full_followup_available conflicts with outtime"
        )
    if cutoff.timestamp_spec_version != TIMESTAMP_SPEC_VERSION:
        raise OrganSupportLabelError("timestamp specification version mismatch")


def _validate_baseline_states(
    cutoff: SupportLabelCutoff,
    vasopressor: VasopressorStateResult,
    ventilation: VentilationStateResult,
    event_dictionary: EventDictionary,
    execution_mode: ExecutionMode,
) -> None:
    if not isinstance(vasopressor, VasopressorStateResult):
        raise OrganSupportLabelError("vasopressor baseline artifact has wrong type")
    if not isinstance(ventilation, VentilationStateResult):
        raise OrganSupportLabelError("ventilation baseline artifact has wrong type")
    for name, artifact_stay, artifact_cutoff in (
        ("vasopressor", vasopressor.stay_id, vasopressor.cutoff),
        ("ventilation", ventilation.stay_id, ventilation.cutoff),
    ):
        if not _same_identifier(artifact_stay, cutoff.stay_id):
            raise OrganSupportLabelError(name + " baseline stay mismatch")
        if artifact_cutoff != cutoff.prediction_time:
            raise OrganSupportLabelError(name + " baseline cutoff mismatch")
    for name, version, digest, mode in (
        (
            "vasopressor",
            vasopressor.event_dictionary_version,
            vasopressor.event_dictionary_sha256,
            vasopressor.execution_mode,
        ),
        (
            "ventilation",
            ventilation.event_dictionary_version,
            ventilation.event_dictionary_sha256,
            ventilation.execution_mode,
        ),
    ):
        if version != event_dictionary.event_dictionary_version:
            raise OrganSupportLabelError(name + " event-dictionary version mismatch")
        if digest != event_dictionary.config_sha256:
            raise OrganSupportLabelError(name + " event-dictionary hash mismatch")
        if mode is not execution_mode:
            raise OrganSupportLabelError(name + " execution-mode mismatch")
    if vasopressor.state_rule_version != event_dictionary.state_rule_version:
        raise OrganSupportLabelError("vasopressor state-rule version mismatch")
    if ventilation.state_rule_version != event_dictionary.ventilation.state_rule_version:
        raise OrganSupportLabelError("ventilation state-rule version mismatch")
    if vasopressor.vaso_at_risk != (
        vasopressor.support_state is SupportState.OFF
    ):
        raise OrganSupportLabelError("vasopressor state/at-risk inconsistency")
    if ventilation.invasive_on != (
        ventilation.invasive_state is InvasiveVentilationState.ON
    ) or ventilation.vent_at_risk != (
        ventilation.invasive_state is InvasiveVentilationState.OFF
    ):
        raise OrganSupportLabelError("ventilation state/at-risk inconsistency")


def _validate_onset(
    evidence: ComponentOnsetEvidence,
    *,
    cutoff: SupportLabelCutoff,
    vasopressor: VasopressorStateResult,
    ventilation: VentilationStateResult,
    event_dictionary: EventDictionary,
    execution_mode: ExecutionMode,
) -> None:
    if not isinstance(evidence, ComponentOnsetEvidence):
        raise OrganSupportLabelError("onsets must be ComponentOnsetEvidence")
    if not _valid_identifier(evidence.stay_id):
        raise OrganSupportLabelError("onset stay_id must be a non-empty int or string")
    if not isinstance(evidence.component, SupportComponent):
        raise OrganSupportLabelError("onset component is unknown")
    if not isinstance(evidence.transition_kind, TransitionKind):
        raise OrganSupportLabelError("onset transition kind is unknown")
    if not isinstance(evidence.onset_time, datetime):
        raise OrganSupportLabelError("onset_time must be a datetime")
    if (evidence.onset_time.utcoffset() is None) != (
        cutoff.prediction_time.utcoffset() is None
    ):
        raise OrganSupportLabelError(
            "onset and cutoff must use consistent timezone awareness"
        )
    for field, value in (
        ("source_event_ref", evidence.source_event_ref),
        ("provenance_ref", evidence.provenance_ref),
        ("state_rule_version", evidence.state_rule_version),
        ("definition_version", evidence.definition_version),
        ("adapter_version", evidence.adapter_version),
    ):
        if not isinstance(value, str) or not value.strip():
            raise OrganSupportLabelError(field + " must be a non-empty string")
    if evidence.event_dictionary_version != event_dictionary.event_dictionary_version:
        raise OrganSupportLabelError("onset event-dictionary version mismatch")
    if evidence.event_dictionary_sha256 != event_dictionary.config_sha256:
        raise OrganSupportLabelError("onset event-dictionary hash mismatch")
    if evidence.evidence_contract_version != COMPONENT_ONSET_EVIDENCE_VERSION:
        raise OrganSupportLabelError("component-onset evidence version mismatch")
    if evidence.execution_mode is not execution_mode:
        raise OrganSupportLabelError("onset execution-mode mismatch")
    if evidence.component is SupportComponent.VASOPRESSOR:
        if evidence.state_rule_version != vasopressor.state_rule_version:
            raise OrganSupportLabelError("vasopressor onset state-rule mismatch")
        if evidence.definition_version != vasopressor.mapping_provenance_version:
            raise OrganSupportLabelError("vasopressor onset mapping mismatch")
        if evidence.adapter_version != vasopressor.mapping_provenance_version:
            raise OrganSupportLabelError("vasopressor onset adapter mismatch")
    else:
        if evidence.state_rule_version != ventilation.state_rule_version:
            raise OrganSupportLabelError("ventilation onset state-rule mismatch")
        if evidence.definition_version != ventilation.concept_version:
            raise OrganSupportLabelError("ventilation onset concept mismatch")
        if evidence.adapter_version != ventilation.adapter_version:
            raise OrganSupportLabelError("ventilation onset adapter mismatch")


def _earliest_by_component(
    evidence: Iterable[ComponentOnsetEvidence],
) -> Mapping[SupportComponent, ComponentOnsetEvidence]:
    earliest = {}
    for item in evidence:
        existing = earliest.get(item.component)
        key = (_instant(item.onset_time), item.source_event_ref)
        if existing is None or key < (
            _instant(existing.onset_time),
            existing.source_event_ref,
        ):
            earliest[item.component] = item
    return earliest


def generate_organ_support_label(
    *,
    cutoff: SupportLabelCutoff,
    vasopressor_state: VasopressorStateResult,
    ventilation_state: VentilationStateResult,
    onset_evidence: Iterable[ComponentOnsetEvidence],
    event_dictionary: EventDictionary,
    execution_mode: ExecutionMode,
) -> OrganSupportLabelResult:
    """Classify one cutoff as positive, negative, censored, or not at risk."""

    if not isinstance(execution_mode, ExecutionMode):
        raise OrganSupportLabelError("execution_mode must be explicit")
    contract = event_dictionary.organ_support
    if execution_mode is ExecutionMode.SCIENTIFIC:
        raise EventDictionaryError(
            "REAL ORGAN-SUPPORT LABEL GENERATION = BLOCKED — "
            "REAL STATE DEFINITIONS / DATA REQUIRED"
        )
    _validate_cutoff(cutoff, contract.horizon_hours)
    _validate_baseline_states(
        cutoff,
        vasopressor_state,
        ventilation_state,
        event_dictionary,
        execution_mode,
    )

    materialized = tuple(onset_evidence)
    seen_refs = set()
    seen_exact = set()
    for evidence in materialized:
        _validate_onset(
            evidence,
            cutoff=cutoff,
            vasopressor=vasopressor_state,
            ventilation=ventilation_state,
            event_dictionary=event_dictionary,
            execution_mode=execution_mode,
        )
        ref_key = (evidence.component, evidence.source_event_ref)
        exact_key = (
            type(evidence.stay_id),
            evidence.stay_id,
            evidence.component,
            evidence.onset_time,
            evidence.transition_kind,
            evidence.state_rule_version,
            evidence.definition_version,
            evidence.adapter_version,
            evidence.provenance_ref,
        )
        if ref_key in seen_refs:
            raise OrganSupportLabelError("duplicate/conflicting onset source reference")
        if exact_key in seen_exact:
            raise OrganSupportLabelError("exact duplicate normalized onset evidence")
        seen_refs.add(ref_key)
        seen_exact.add(exact_key)

    vaso_at_risk = vasopressor_state.vaso_at_risk
    vent_at_risk = ventilation_state.vent_at_risk
    risk_by_component = {
        SupportComponent.VASOPRESSOR: vaso_at_risk,
        SupportComponent.INVASIVE_VENTILATION: vent_at_risk,
    }
    horizon_end = _add_elapsed_hours(
        cutoff.prediction_time, contract.horizon_hours
    )
    qualifying = []
    for evidence in materialized:
        if not _same_identifier(evidence.stay_id, cutoff.stay_id):
            continue
        if not risk_by_component[evidence.component]:
            continue
        if evidence.transition_kind is not TransitionKind.OFF_TO_ON:
            continue
        onset = _instant(evidence.onset_time)
        if not (_instant(cutoff.prediction_time) < onset <= _instant(horizon_end)):
            continue
        if onset == _instant(cutoff.outtime):
            raise SupportOnsetExitTieError(
                "BLOCKED — SUPPORT ONSET / ICU-EXIT SAME-TIMESTAMP POLICY REQUIRED"
            )
        if onset > _instant(cutoff.outtime):
            continue
        qualifying.append(evidence)

    qualifying.sort(
        key=lambda item: (
            _instant(item.onset_time),
            item.component.value,
            item.source_event_ref,
        )
    )
    by_component = _earliest_by_component(qualifying)
    vaso_onset = by_component.get(SupportComponent.VASOPRESSOR)
    vent_onset = by_component.get(SupportComponent.INVASIVE_VENTILATION)

    if not (vaso_at_risk or vent_at_risk):
        label = None
        eligible = False
        disposition = AuditDisposition.NOT_AT_RISK_BOTH_COMPONENTS_ON
    elif qualifying:
        label = 1
        eligible = True
        disposition = AuditDisposition.POSITIVE_INITIATION
    elif cutoff.support24_full_followup_available:
        label = 0
        eligible = True
        disposition = AuditDisposition.NEGATIVE_FULL_FOLLOWUP
    else:
        label = None
        eligible = False
        disposition = AuditDisposition.CENSORED_EARLY_EXIT_NO_INITIATION

    selected = qualifying[0] if qualifying else None
    return OrganSupportLabelResult(
        stay_id=cutoff.stay_id,
        prediction_time=cutoff.prediction_time,
        label=label,
        organ_support_eligible=eligible,
        disposition=disposition,
        audit_reason_vocabulary_version=contract.audit_reason_vocabulary_version,
        audit_reason_vocabulary_status=contract.audit_reason_vocabulary_status,
        vaso_on_at_t=vasopressor_state.support_state is SupportState.ON,
        vaso_at_risk=vaso_at_risk,
        vent_on_at_t=ventilation_state.invasive_on,
        vent_at_risk=vent_at_risk,
        support24_full_followup_available=(
            cutoff.support24_full_followup_available
        ),
        outtime=cutoff.outtime,
        vasopressor_onset_time=None if vaso_onset is None else vaso_onset.onset_time,
        vasopressor_onset_ref=(
            None if vaso_onset is None else vaso_onset.source_event_ref
        ),
        ventilation_onset_time=None if vent_onset is None else vent_onset.onset_time,
        ventilation_onset_ref=(
            None if vent_onset is None else vent_onset.source_event_ref
        ),
        earliest_composite_onset_time=(
            None if selected is None else selected.onset_time
        ),
        earliest_composite_onset_component=(
            None if selected is None else selected.component
        ),
        event_dictionary_version=event_dictionary.event_dictionary_version,
        event_dictionary_sha256=event_dictionary.config_sha256,
        vasopressor_state_rule_version=vasopressor_state.state_rule_version,
        vasopressor_mapping_provenance_version=(
            vasopressor_state.mapping_provenance_version
        ),
        ventilation_state_rule_version=ventilation_state.state_rule_version,
        ventilation_concept_version=ventilation_state.concept_version,
        ventilation_adapter_version=ventilation_state.adapter_version,
        timestamp_spec_version=cutoff.timestamp_spec_version,
        label_rule_version=contract.label_rule_version,
        execution_mode=execution_mode,
    )


def to_phase7_support_label_record(
    result: OrganSupportLabelResult, *, split: str
) -> SupportLabelRecord:
    """Adapt one result to Vedant's existing label/eligibility handoff record."""

    if not isinstance(result, OrganSupportLabelResult):
        raise OrganSupportLabelError("result must be OrganSupportLabelResult")
    if not isinstance(split, str) or not split.strip():
        raise OrganSupportLabelError("split must be a non-empty string")
    if result.organ_support_eligible and result.label not in (0, 1):
        raise OrganSupportLabelError("eligible support label must be binary")
    if not result.organ_support_eligible and result.label is not None:
        raise OrganSupportLabelError("ineligible support label must remain null")
    audit_metadata = {
        "audit_reason_vocabulary_status": result.audit_reason_vocabulary_status,
        "event_dictionary_sha256": result.event_dictionary_sha256,
        "event_dictionary_version": result.event_dictionary_version,
        "outtime": result.outtime.isoformat(),
        "support24_full_followup_available": (
            result.support24_full_followup_available
        ),
        "vaso_at_risk": result.vaso_at_risk,
        "vaso_on_at_t": result.vaso_on_at_t,
        "vasopressor_onset_ref": result.vasopressor_onset_ref,
        "vasopressor_onset_time": (
            None
            if result.vasopressor_onset_time is None
            else result.vasopressor_onset_time.isoformat()
        ),
        "vent_at_risk": result.vent_at_risk,
        "vent_on_at_t": result.vent_on_at_t,
        "ventilation_onset_ref": result.ventilation_onset_ref,
        "ventilation_onset_time": (
            None
            if result.ventilation_onset_time is None
            else result.ventilation_onset_time.isoformat()
        ),
    }
    return SupportLabelRecord(
        stay_id=result.stay_id,
        prediction_time=result.prediction_time.isoformat(),
        split=split,
        label=result.label,
        eligible=result.organ_support_eligible,
        audit_case=result.disposition.value,
        audit_metadata=audit_metadata,
    )


def build_phase7_handoff_records(
    results: Iterable[OrganSupportLabelResult],
    *,
    split_by_key: Mapping[Tuple[Identifier, datetime], str],
) -> Tuple[SupportLabelRecord, ...]:
    """Build unique existing-interface records in deterministic key order."""

    materialized = tuple(results)
    seen = set()
    records = []
    for result in materialized:
        if not isinstance(result, OrganSupportLabelResult):
            raise OrganSupportLabelError("handoff input has wrong result type")
        if result.key in seen:
            raise OrganSupportLabelError("duplicate support-label canonical key")
        seen.add(result.key)
        if result.key not in split_by_key:
            raise OrganSupportLabelError("support-label key lacks split assignment")
        records.append(
            to_phase7_support_label_record(
                result,
                split=split_by_key[result.key],
            )
        )
    return tuple(
        sorted(
            records,
            key=lambda item: (
                str(type(item.stay_id)),
                str(item.stay_id),
                item.prediction_time,
            ),
        )
    )
