import inspect
import json
from dataclasses import fields, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import yaml

from data.collate import CanonicalBatch
from data.support_labels import SupportLabelRecord
from data.timestamps import PredictionTimestamp, TIMESTAMP_SPEC_VERSION
from labels.event_dict import (
    EventDictionaryError,
    load_event_dictionary,
    validate_event_dictionary,
)
from labels.organ_support import (
    AuditDisposition,
    ComponentOnsetEvidence,
    OrganSupportLabelError,
    OrganSupportLabelResult,
    SupportComponent,
    SupportLabelCutoff,
    SupportOnsetExitTieError,
    TransitionKind,
    build_phase7_handoff_records,
    generate_organ_support_label,
    to_phase7_support_label_record,
)
import labels.organ_support as organ_support_module
from labels.support_state import (
    ExecutionMode,
    NormalizedActiveInterval,
    query_vasopressor_state,
)
from labels.ventilation_state import (
    NormalizedVentilationInterval,
    RespiratoryCategory,
    query_invasive_ventilation_state,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/event_dict_v1.yaml"
FIXTURE = ROOT / "tests/fixtures/organ_support/truth_table_v1.json"
CUTOFF = datetime(2026, 1, 2, 12, tzinfo=timezone.utc)
STAY = "SYNTHETIC_SUPPORT_STAY_A"


@pytest.fixture(scope="module")
def event_dictionary():
    return load_event_dictionary(CONFIG)


def baseline_states(event_dictionary, *, vaso_on=False, vent_on=False, stay_id=STAY):
    vaso_intervals = []
    if vaso_on:
        vaso_intervals.append(
            NormalizedActiveInterval(
                stay_id=stay_id,
                agent_key="QUALIFYING_AGENT_A",
                interval_start=CUTOFF - timedelta(hours=2),
                interval_end=CUTOFF + timedelta(hours=2),
                source_event_ref="SYNTHETIC_BASELINE_VASO",
                normalization_provenance_version=(
                    "SYNTHETIC_NORMALIZER_V1_NOT_MIMIC"
                ),
            )
        )
    vent_intervals = []
    if vent_on:
        vent_intervals.append(
            NormalizedVentilationInterval(
                stay_id=stay_id,
                category=RespiratoryCategory.INVASIVE,
                interval_start=CUTOFF - timedelta(hours=2),
                interval_end=CUTOFF + timedelta(hours=2),
                source_state_ref="SYNTHETIC_BASELINE_VENT",
                concept_version=event_dictionary.ventilation.synthetic_concept_version,
                adapter_version=event_dictionary.ventilation.synthetic_adapter_version,
                normalization_provenance_ref=(
                    "SYNTHETIC_NORMALIZER_V1_NOT_MIMIC_CODE"
                ),
            )
        )
    vaso = query_vasopressor_state(
        vaso_intervals,
        stay_id=stay_id,
        cutoff=CUTOFF,
        event_dictionary=event_dictionary,
        execution_mode=ExecutionMode.SYNTHETIC,
    )
    vent = query_invasive_ventilation_state(
        vent_intervals,
        stay_id=stay_id,
        cutoff=CUTOFF,
        event_dictionary=event_dictionary,
        execution_mode=ExecutionMode.SYNTHETIC,
    )
    return vaso, vent


def cutoff(*, outtime_hours=30, followup=None, stay_id=STAY):
    if followup is None:
        followup = outtime_hours >= 24
    return SupportLabelCutoff(
        stay_id=stay_id,
        prediction_time=CUTOFF,
        outtime=CUTOFF + timedelta(hours=outtime_hours),
        support24_full_followup_available=followup,
        timestamp_spec_version=TIMESTAMP_SPEC_VERSION,
    )


def onset(
    event_dictionary,
    vaso,
    vent,
    *,
    component,
    hours,
    ref=None,
    transition=TransitionKind.OFF_TO_ON,
    stay_id=STAY,
):
    if component is SupportComponent.VASOPRESSOR:
        rule = vaso.state_rule_version
        definition = vaso.mapping_provenance_version
        adapter = vaso.mapping_provenance_version
    else:
        rule = vent.state_rule_version
        definition = vent.concept_version
        adapter = vent.adapter_version
    return ComponentOnsetEvidence(
        stay_id=stay_id,
        component=component,
        onset_time=CUTOFF + timedelta(hours=hours),
        transition_kind=transition,
        source_event_ref=ref or "{}-{}".format(component.value, hours),
        provenance_ref="SYNTHETIC_PHASE3_PROVENANCE_NOT_MIMIC",
        state_rule_version=rule,
        definition_version=definition,
        adapter_version=adapter,
        event_dictionary_version=event_dictionary.event_dictionary_version,
        event_dictionary_sha256=event_dictionary.config_sha256,
        evidence_contract_version=(
            event_dictionary.organ_support.onset_evidence_contract_version
        ),
        execution_mode=ExecutionMode.SYNTHETIC,
    )


def label(
    event_dictionary,
    *,
    vaso_on=False,
    vent_on=False,
    outtime_hours=30,
    onsets=(),
    followup=None,
):
    vaso, vent = baseline_states(
        event_dictionary, vaso_on=vaso_on, vent_on=vent_on
    )
    evidence = [
        onset(
            event_dictionary,
            vaso,
            vent,
            component=SupportComponent(item["component"]),
            hours=item["hours"],
            transition=TransitionKind(
                item.get("transition", TransitionKind.OFF_TO_ON.value)
            ),
            ref=item.get("ref"),
        )
        for item in onsets
    ]
    return generate_organ_support_label(
        cutoff=cutoff(outtime_hours=outtime_hours, followup=followup),
        vasopressor_state=vaso,
        ventilation_state=vent,
        onset_evidence=evidence,
        event_dictionary=event_dictionary,
        execution_mode=ExecutionMode.SYNTHETIC,
    )


def test_synthetic_truth_table(event_dictionary):
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert payload["artifact_kind"].endswith("NOT_REAL_LABELS")
    for case in payload["cases"]:
        result = label(
            event_dictionary,
            vaso_on=case["vaso_on"],
            vent_on=case["vent_on"],
            outtime_hours=case["outtime_hours"],
            onsets=case["onsets"],
        )
        assert result.label == case["expected_label"], case["case"]
        assert result.organ_support_eligible is case["expected_eligible"], case[
            "case"
        ]


def test_positive_before_early_exit_has_precedence(event_dictionary):
    result = label(
        event_dictionary,
        outtime_hours=10,
        onsets=[{"component": "VASOPRESSOR", "hours": 5}],
    )
    assert result.label == 1
    assert result.organ_support_eligible
    assert not result.support24_full_followup_available
    assert result.disposition is AuditDisposition.POSITIVE_INITIATION


def test_partial_followup_without_event_is_censored_not_zero(event_dictionary):
    result = label(event_dictionary, outtime_hours=10)
    assert result.label is None
    assert not result.organ_support_eligible
    assert result.disposition is AuditDisposition.CENSORED_EARLY_EXIT_NO_INITIATION


def test_exact_24h_followup_without_event_is_negative(event_dictionary):
    result = label(event_dictionary, outtime_hours=24)
    assert result.label == 0
    assert result.organ_support_eligible
    assert result.disposition is AuditDisposition.NEGATIVE_FULL_FOLLOWUP


def test_both_components_on_are_not_at_risk_even_with_full_followup(event_dictionary):
    result = label(event_dictionary, vaso_on=True, vent_on=True)
    assert result.label is None
    assert not result.organ_support_eligible
    assert result.disposition is AuditDisposition.NOT_AT_RISK_BOTH_COMPONENTS_ON
    assert result.support24_full_followup_available


def test_one_on_one_off_remains_classifiable(event_dictionary):
    negative = label(event_dictionary, vaso_on=True, vent_on=False)
    positive = label(
        event_dictionary,
        vaso_on=True,
        vent_on=False,
        onsets=[{"component": "INVASIVE_VENTILATION", "hours": 3}],
    )
    assert negative.label == 0
    assert positive.label == 1


def test_dual_initiation_is_one_positive_with_earliest_evidence(event_dictionary):
    result = label(
        event_dictionary,
        onsets=[
            {"component": "INVASIVE_VENTILATION", "hours": 8},
            {"component": "VASOPRESSOR", "hours": 4},
        ],
    )
    assert result.label == 1
    assert result.earliest_composite_onset_component is SupportComponent.VASOPRESSOR
    assert result.vasopressor_onset_time == CUTOFF + timedelta(hours=4)
    assert result.ventilation_onset_time == CUTOFF + timedelta(hours=8)


@pytest.mark.parametrize(
    "component",
    [SupportComponent.VASOPRESSOR, SupportComponent.INVASIVE_VENTILATION],
)
def test_already_on_component_change_cannot_create_positive(
    component, event_dictionary
):
    vaso_on = component is SupportComponent.VASOPRESSOR
    vent_on = component is SupportComponent.INVASIVE_VENTILATION
    result = label(
        event_dictionary,
        vaso_on=vaso_on,
        vent_on=vent_on,
        onsets=[
            {
                "component": component.value,
                "hours": 3,
                "transition": TransitionKind.NO_COMPONENT_OFF_TO_ON.value,
            }
        ],
    )
    assert result.label == 0


def test_second_pressor_while_vaso_on_is_not_initiation(event_dictionary):
    result = label(
        event_dictionary,
        vaso_on=True,
        vent_on=False,
        onsets=[
            {
                "component": "VASOPRESSOR",
                "hours": 2,
                "transition": "NO_COMPONENT_OFF_TO_ON",
            }
        ],
    )
    assert result.label == 0
    assert result.vasopressor_onset_time is None


def test_invasive_mode_change_while_on_is_not_initiation(event_dictionary):
    result = label(
        event_dictionary,
        vaso_on=False,
        vent_on=True,
        onsets=[
            {
                "component": "INVASIVE_VENTILATION",
                "hours": 2,
                "transition": "NO_COMPONENT_OFF_TO_ON",
            }
        ],
    )
    assert result.label == 0
    assert result.ventilation_onset_time is None


def test_future_window_is_left_exclusive_right_inclusive(event_dictionary):
    at_left = label(
        event_dictionary,
        onsets=[{"component": "VASOPRESSOR", "hours": 0}],
    )
    at_right = label(
        event_dictionary,
        outtime_hours=30,
        onsets=[{"component": "VASOPRESSOR", "hours": 24}],
    )
    outside = label(
        event_dictionary,
        outtime_hours=30,
        onsets=[{"component": "VASOPRESSOR", "hours": 24.0001}],
    )
    assert at_left.label == 0
    assert at_right.label == 1
    assert outside.label == 0


def test_post_outtime_event_cannot_make_positive(event_dictionary):
    result = label(
        event_dictionary,
        outtime_hours=5,
        onsets=[{"component": "VASOPRESSOR", "hours": 8}],
    )
    assert result.label is None
    assert not result.organ_support_eligible


def test_onset_equal_to_outtime_is_blocked_pending_policy(event_dictionary):
    with pytest.raises(SupportOnsetExitTieError, match="SAME-TIMESTAMP"):
        label(
            event_dictionary,
            outtime_hours=10,
            onsets=[{"component": "VASOPRESSOR", "hours": 10}],
        )


def test_other_stay_onset_is_ignored(event_dictionary):
    vaso, vent = baseline_states(event_dictionary)
    other = onset(
        event_dictionary,
        vaso,
        vent,
        component=SupportComponent.VASOPRESSOR,
        hours=3,
        stay_id="SYNTHETIC_OTHER_STAY",
    )
    result = generate_organ_support_label(
        cutoff=cutoff(),
        vasopressor_state=vaso,
        ventilation_state=vent,
        onset_evidence=[other],
        event_dictionary=event_dictionary,
        execution_mode=ExecutionMode.SYNTHETIC,
    )
    assert result.label == 0


def test_event_order_invariance(event_dictionary):
    vaso, vent = baseline_states(event_dictionary)
    events = [
        onset(
            event_dictionary,
            vaso,
            vent,
            component=SupportComponent.VASOPRESSOR,
            hours=8,
            ref="z",
        ),
        onset(
            event_dictionary,
            vaso,
            vent,
            component=SupportComponent.INVASIVE_VENTILATION,
            hours=4,
            ref="a",
        ),
    ]
    arguments = {
        "cutoff": cutoff(),
        "vasopressor_state": vaso,
        "ventilation_state": vent,
        "event_dictionary": event_dictionary,
        "execution_mode": ExecutionMode.SYNTHETIC,
    }
    assert generate_organ_support_label(
        onset_evidence=events, **arguments
    ) == generate_organ_support_label(onset_evidence=reversed(events), **arguments)


def test_duplicate_onset_evidence_fails_closed(event_dictionary):
    vaso, vent = baseline_states(event_dictionary)
    first = onset(
        event_dictionary,
        vaso,
        vent,
        component=SupportComponent.VASOPRESSOR,
        hours=3,
        ref="same",
    )
    with pytest.raises(OrganSupportLabelError, match="source reference"):
        generate_organ_support_label(
            cutoff=cutoff(),
            vasopressor_state=vaso,
            ventilation_state=vent,
            onset_evidence=[first, replace(first, onset_time=CUTOFF + timedelta(hours=4))],
            event_dictionary=event_dictionary,
            execution_mode=ExecutionMode.SYNTHETIC,
        )


def test_temporal_flag_mismatch_fails_instead_of_becoming_eligibility(event_dictionary):
    vaso, vent = baseline_states(event_dictionary)
    with pytest.raises(OrganSupportLabelError, match="conflicts with outtime"):
        generate_organ_support_label(
            cutoff=cutoff(outtime_hours=10, followup=True),
            vasopressor_state=vaso,
            ventilation_state=vent,
            onset_evidence=[],
            event_dictionary=event_dictionary,
            execution_mode=ExecutionMode.SYNTHETIC,
        )


def test_prediction_timestamp_adapter_reuses_temporal_flag():
    row = PredictionTimestamp(
        subject_id="SUBJECT",
        stay_id=STAY,
        prediction_time=CUTOFF,
        grid_index=0,
        icu_elapsed_hours=24,
        recovery24_followup_available=True,
        recovery48_followup_available=False,
        support24_full_followup_available=True,
        icu_time_temporally_eligible=True,
    )
    adapted = SupportLabelCutoff.from_prediction_timestamp(
        row, outtime=CUTOFF + timedelta(hours=24)
    )
    assert adapted.support24_full_followup_available
    assert adapted.timestamp_spec_version == TIMESTAMP_SPEC_VERSION


def test_state_and_onset_version_mismatches_fail(event_dictionary):
    vaso, vent = baseline_states(event_dictionary)
    bad_state = replace(vaso, event_dictionary_sha256="0" * 64)
    with pytest.raises(OrganSupportLabelError, match="hash mismatch"):
        generate_organ_support_label(
            cutoff=cutoff(),
            vasopressor_state=bad_state,
            ventilation_state=vent,
            onset_evidence=[],
            event_dictionary=event_dictionary,
            execution_mode=ExecutionMode.SYNTHETIC,
        )
    event = onset(
        event_dictionary,
        vaso,
        vent,
        component=SupportComponent.INVASIVE_VENTILATION,
        hours=2,
    )
    with pytest.raises(OrganSupportLabelError, match="concept mismatch"):
        generate_organ_support_label(
            cutoff=cutoff(),
            vasopressor_state=vaso,
            ventilation_state=vent,
            onset_evidence=[replace(event, definition_version="wrong")],
            event_dictionary=event_dictionary,
            execution_mode=ExecutionMode.SYNTHETIC,
        )


def test_unknown_baseline_state_fails_closed(event_dictionary):
    vaso, vent = baseline_states(event_dictionary)
    attacked = replace(vaso, support_state="UNKNOWN", vaso_at_risk=True)
    with pytest.raises(OrganSupportLabelError, match="state/at-risk"):
        generate_organ_support_label(
            cutoff=cutoff(),
            vasopressor_state=attacked,
            ventilation_state=vent,
            onset_evidence=[],
            event_dictionary=event_dictionary,
            execution_mode=ExecutionMode.SYNTHETIC,
        )


def test_scientific_execution_is_blocked(event_dictionary):
    vaso, vent = baseline_states(event_dictionary)
    with pytest.raises(EventDictionaryError, match="REAL ORGAN-SUPPORT"):
        generate_organ_support_label(
            cutoff=cutoff(),
            vasopressor_state=vaso,
            ventilation_state=vent,
            onset_evidence=[],
            event_dictionary=event_dictionary,
            execution_mode=ExecutionMode.SCIENTIFIC,
        )


def test_phase7_record_preserves_null_for_ineligible_and_audit_separation(
    event_dictionary,
):
    result = label(event_dictionary, outtime_hours=10)
    record = to_phase7_support_label_record(result, split="train")
    assert isinstance(record, SupportLabelRecord)
    assert record.label is None
    assert not record.eligible
    assert "vasopressor_onset_time" in record.audit_metadata
    with pytest.raises(OrganSupportLabelError, match="remain null"):
        to_phase7_support_label_record(
            replace(result, label=0),
            split="train",
        )


def test_phase7_handoff_enforces_canonical_uniqueness(event_dictionary):
    result = label(event_dictionary)
    split = {result.key: "train"}
    records = build_phase7_handoff_records([result], split_by_key=split)
    assert len(records) == 1
    with pytest.raises(OrganSupportLabelError, match="duplicate"):
        build_phase7_handoff_records([result, result], split_by_key=split)


def test_future_outcome_changes_label_not_model_facing_baseline(event_dictionary):
    vaso, vent = baseline_states(event_dictionary)
    event = onset(
        event_dictionary,
        vaso,
        vent,
        component=SupportComponent.VASOPRESSOR,
        hours=3,
    )
    arguments = {
        "cutoff": cutoff(),
        "vasopressor_state": vaso,
        "ventilation_state": vent,
        "event_dictionary": event_dictionary,
        "execution_mode": ExecutionMode.SYNTHETIC,
    }
    positive = generate_organ_support_label(onset_evidence=[event], **arguments)
    negative = generate_organ_support_label(onset_evidence=[], **arguments)
    assert positive.label == 1
    assert negative.label == 0
    assert arguments["vasopressor_state"] is vaso
    assert arguments["ventilation_state"] is vent


def test_audit_fields_are_not_canonical_batch_fields():
    names = {field.name for field in fields(CanonicalBatch)}
    audit_names = {
        "vasopressor_onset_time",
        "ventilation_onset_time",
        "disposition",
        "vaso_at_risk",
        "vent_at_risk",
    }
    assert not names & audit_names


def test_result_contract_has_only_binary_or_null_label(event_dictionary):
    for result in (
        label(event_dictionary),
        label(event_dictionary, outtime_hours=10),
        label(
            event_dictionary,
            onsets=[{"component": "VASOPRESSOR", "hours": 2}],
        ),
    ):
        assert result.label in (None, 0, 1)
        assert isinstance(result, OrganSupportLabelResult)


def test_phase3_module_has_no_prevalence_or_class_weight_path():
    source = inspect.getsource(organ_support_module).lower()
    assert "prevalence" not in source
    assert "pos_weight" not in source
    assert "class_weight" not in source


def test_event_dictionary_phase3_semantics_are_draft_and_locked():
    contract = load_event_dictionary(CONFIG).organ_support
    assert contract.status == "DRAFT_NON_AUTHORIZING"
    assert not contract.real_scientific_execution_authorized
    assert contract.horizon_hours == 24
    assert contract.future_interval_semantics == "(t,t+24h]"
    assert contract.onset_outtime_tie_status.startswith("UNRESOLVED")


def test_event_dictionary_rejects_phase4_freeze_or_semantic_attack():
    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    frozen = json.loads(json.dumps(payload))
    frozen["organ_support_label"]["status"] = "FROZEN_ENDPOINT"
    frozen["organ_support_label"]["real_scientific_execution_authorized"] = True
    with pytest.raises(EventDictionaryError, match="tie policy"):
        validate_event_dictionary(frozen, config_sha256="a" * 64)

    attacked = json.loads(json.dumps(payload))
    attacked["organ_support_label"]["right_boundary_inclusive"] = False
    with pytest.raises(EventDictionaryError, match="right_boundary_inclusive"):
        validate_event_dictionary(attacked, config_sha256="a" * 64)
