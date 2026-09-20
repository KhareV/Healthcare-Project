"""Fail-closed event-dictionary loading for Pulkit's support-state work.

The checked-in dictionary is deliberately draft and non-authorizing.  It may
drive symbolic synthetic tests, but it cannot authorize real MIMIC label or
prevalence generation.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, FrozenSet, Mapping, Optional, Union

import yaml

from vedant_infra.hashing import is_sha256, sha256_file


PathLike = Union[str, Path]
SCHEMA_VERSION = "event_dict_schema_v1"
EVENT_DICTIONARY_VERSION = "event_dict_v1"
STATE_RULE_VERSION = "vasopressor_state_rule_v1"
NORMALIZED_CONTRACT_VERSION = "normalized_active_support_interval_v1"
VENTILATION_STATE_RULE_VERSION = "invasive_ventilation_state_rule_v1"
VENTILATION_NORMALIZED_CONTRACT_VERSION = "normalized_ventilation_interval_v1"
ORGAN_SUPPORT_LABEL_RULE_VERSION = "organ_support_label_rule_v1"
COMPONENT_ONSET_EVIDENCE_VERSION = "component_off_to_on_evidence_v1"
PHASE7_HANDOFF_RECORD_CONTRACT = "SupportLabelRecord_v1_existing_phase7_interface"
ORGAN_SUPPORT_FROZEN_STATUS = "FROZEN_ENDPOINT"
DRAFT_STATUS = "DRAFT_NON_AUTHORIZING"
FROZEN_STATUS = "FROZEN_SCIENTIFIC_CONTRACT"
VENTILATION_FROZEN_STATUS = "FROZEN_REAL"
SUPPORTED_BOUNDARY_POLICIES = frozenset({"CLOSED_OPEN", "CLOSED_CLOSED"})
VENTILATION_INTERNAL_CATEGORIES = frozenset(
    {"INVASIVE", "NON_INVASIVE", "HFNC", "ORDINARY_OXYGEN"}
)
PLACEHOLDER_MARKERS = ("PLACEHOLDER", "TODO", "TBD", "FAKE", "EXAMPLE")
REAL_CONCEPT_PROHIBITED_MARKERS = PLACEHOLDER_MARKERS + (
    "SYNTHETIC",
    "UNKNOWN",
    "UNVERIFIED",
)


class EventDictionaryError(ValueError):
    """Raised when an event dictionary is malformed or cannot authorize use."""


@dataclass(frozen=True)
class VentilationEventContract:
    """Validated invasive-ventilation section of the shared dictionary."""

    status: str
    real_scientific_execution_authorized: bool
    state_rule_version: str
    normalized_contract_version: str
    scientific_concept_name: Optional[str]
    scientific_concept_version: Optional[str]
    scientific_adapter_version: Optional[str]
    scientific_boundary_policy: Optional[str]
    sofa_alignment_status: str
    synthetic_concept_name: str
    synthetic_concept_version: str
    synthetic_adapter_version: str
    synthetic_known_categories: FrozenSet[str]
    synthetic_invasive_categories: FrozenSet[str]
    synthetic_non_invasive_categories: FrozenSet[str]
    synthetic_boundary_policy: str

    @property
    def is_frozen_for_scientific_use(self) -> bool:
        return (
            self.status == VENTILATION_FROZEN_STATUS
            and self.real_scientific_execution_authorized
        )


@dataclass(frozen=True)
class OrganSupportLabelContract:
    """Validated draft composite-label semantics; Phase 4 performs freeze."""

    status: str
    real_scientific_execution_authorized: bool
    label_rule_version: str
    onset_evidence_contract_version: str
    phase7_handoff_record_contract: str
    horizon_hours: int
    future_interval_semantics: str
    onset_outtime_tie_status: str
    unknown_component_state_policy_status: str
    audit_reason_vocabulary_status: str
    audit_reason_vocabulary_version: str


@dataclass(frozen=True)
class EventDictionary:
    """Validated Phase-1/2 portions of ``event_dict_v1.yaml``."""

    schema_version: str
    event_dictionary_version: str
    status: str
    real_scientific_execution_authorized: bool
    state_rule_version: str
    normalized_contract_version: str
    scientific_qualifying_agent_keys: FrozenSet[str]
    scientific_boundary_policy: Optional[str]
    mapping_provenance_version: Optional[str]
    synthetic_qualifying_agent_keys: FrozenSet[str]
    synthetic_boundary_policy: str
    synthetic_mapping_provenance_version: str
    ventilation: VentilationEventContract
    organ_support: OrganSupportLabelContract
    config_sha256: str

    @property
    def is_frozen_for_scientific_use(self) -> bool:
        return self.status == FROZEN_STATUS and self.real_scientific_execution_authorized


def _mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise EventDictionaryError(field + " must be a mapping")
    return value


def _nonempty_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EventDictionaryError(field + " must be a non-empty string")
    return value


def _agent_keys(value: object, field: str, *, allow_empty: bool) -> FrozenSet[str]:
    if not isinstance(value, list):
        raise EventDictionaryError(field + " must be a list")
    keys = []
    for key in value:
        text = _nonempty_string(key, field + " entry")
        if text in keys:
            raise EventDictionaryError(field + " contains duplicate identifiers")
        keys.append(text)
    if not keys and not allow_empty:
        raise EventDictionaryError(field + " cannot be empty in a frozen dictionary")
    return frozenset(keys)


def _contains_placeholder(value: object) -> bool:
    if isinstance(value, str):
        upper = value.upper()
        return any(marker in upper for marker in PLACEHOLDER_MARKERS)
    if isinstance(value, dict):
        return any(_contains_placeholder(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_placeholder(item) for item in value)
    return False


def _contains_real_concept_placeholder(value: object) -> bool:
    if isinstance(value, str):
        upper = value.upper()
        return any(marker in upper for marker in REAL_CONCEPT_PROHIBITED_MARKERS)
    if isinstance(value, dict):
        return any(_contains_real_concept_placeholder(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_real_concept_placeholder(item) for item in value)
    return False


def _string_set(
    value: object,
    field: str,
    *,
    expected_values: Optional[FrozenSet[str]] = None,
) -> FrozenSet[str]:
    result = _agent_keys(value, field, allow_empty=False)
    if expected_values is not None and result != expected_values:
        raise EventDictionaryError(field + " does not match the internal category contract")
    return result


def _validate_exact_commit(value: object) -> str:
    commit = _nonempty_string(value, "invasive_ventilation.concept.source_commit")
    if len(commit) not in (40, 64) or any(
        character not in "0123456789abcdef" for character in commit
    ):
        raise EventDictionaryError("ventilation concept commit must be an exact Git hash")
    return commit


def _validate_ventilation_section(
    value: object,
) -> VentilationEventContract:
    section = _mapping(value, "invasive_ventilation")
    status = section.get("status")
    if status not in (DRAFT_STATUS, VENTILATION_FROZEN_STATUS):
        raise EventDictionaryError("unsupported invasive-ventilation status")
    authorized = section.get("real_scientific_execution_authorized")
    if not isinstance(authorized, bool):
        raise EventDictionaryError(
            "invasive_ventilation.real_scientific_execution_authorized must be boolean"
        )
    if status == DRAFT_STATUS and authorized:
        raise EventDictionaryError(
            "draft ventilation concept cannot authorize scientific execution"
        )
    if section.get("state_rule_version") != VENTILATION_STATE_RULE_VERSION:
        raise EventDictionaryError("unsupported invasive-ventilation state-rule version")
    if (
        section.get("normalized_contract_version")
        != VENTILATION_NORMALIZED_CONTRACT_VERSION
    ):
        raise EventDictionaryError(
            "unsupported normalized ventilation contract version"
        )
    if (
        section.get("endpoint_semantics")
        != "binary_invasive_state_only_not_intensity_or_escalation"
    ):
        raise EventDictionaryError("ventilation endpoint must remain binary invasive state")
    if section.get("any_active_invasive_interval_means_on") is not True:
        raise EventDictionaryError("ANY active invasive interval must mean ON")
    if section.get("already_on_means_at_risk") is not False:
        raise EventDictionaryError("already-ON ventilation must not be at risk")

    concept = _mapping(section.get("concept"), "invasive_ventilation.concept")
    category_mapping = _mapping(
        section.get("category_mapping"), "invasive_ventilation.category_mapping"
    )
    if category_mapping.get("unknown_category_policy") != "HARD_FAILURE":
        raise EventDictionaryError("unknown ventilation categories must fail closed")
    output_semantics = _mapping(
        section.get("scientific_output_semantics"),
        "invasive_ventilation.scientific_output_semantics",
    )
    scientific_boundary = output_semantics.get("interval_convention")
    if (
        scientific_boundary is not None
        and scientific_boundary not in SUPPORTED_BOUNDARY_POLICIES
    ):
        raise EventDictionaryError("unsupported ventilation interval-boundary policy")
    sofa_alignment = _mapping(
        section.get("sofa_alignment"), "invasive_ventilation.sofa_alignment"
    )
    sofa_status = _nonempty_string(sofa_alignment.get("status"), "sofa_alignment.status")

    synthetic = _mapping(
        section.get("synthetic_test_profile"),
        "invasive_ventilation.synthetic_test_profile",
    )
    if synthetic.get("synthetic_only") is not True:
        raise EventDictionaryError("ventilation synthetic profile must be synthetic_only")
    synthetic_name = _nonempty_string(
        synthetic.get("concept_name"), "ventilation synthetic concept_name"
    )
    synthetic_version = _nonempty_string(
        synthetic.get("concept_version"), "ventilation synthetic concept_version"
    )
    synthetic_adapter = _nonempty_string(
        synthetic.get("adapter_version"), "ventilation synthetic adapter_version"
    )
    known_categories = _string_set(
        synthetic.get("known_internal_categories"),
        "synthetic known_internal_categories",
        expected_values=VENTILATION_INTERNAL_CATEGORIES,
    )
    invasive_categories = _string_set(
        synthetic.get("invasive_internal_categories"),
        "synthetic invasive_internal_categories",
        expected_values=frozenset({"INVASIVE"}),
    )
    non_invasive_categories = _string_set(
        synthetic.get("explicitly_non_invasive_categories"),
        "synthetic explicitly_non_invasive_categories",
        expected_values=frozenset(
            {"NON_INVASIVE", "HFNC", "ORDINARY_OXYGEN"}
        ),
    )
    if synthetic.get("unknown_category_policy") != "HARD_FAILURE":
        raise EventDictionaryError("synthetic unknown ventilation categories must fail")
    synthetic_boundary = synthetic.get("interval_convention")
    if synthetic_boundary not in SUPPORTED_BOUNDARY_POLICIES:
        raise EventDictionaryError("unsupported synthetic ventilation boundary policy")

    scientific_name = concept.get("concept_name")
    scientific_version = concept.get("concept_version")
    scientific_adapter = concept.get("adapter_version")
    for field, item in (
        ("concept_name", scientific_name),
        ("concept_version", scientific_version),
        ("adapter_version", scientific_adapter),
    ):
        if item is not None:
            _nonempty_string(item, "invasive_ventilation.concept." + field)

    if status == VENTILATION_FROZEN_STATUS:
        if not authorized:
            raise EventDictionaryError("frozen real ventilation concept must authorize use")
        frozen_fields = {
            "concept": concept,
            "category_mapping": category_mapping,
            "output_semantics": output_semantics,
            "duplicates": section.get("duplicate_overlap_policy"),
            "sofa_alignment": sofa_alignment,
            "reviewers": section.get("reviewers"),
        }
        if _contains_real_concept_placeholder(frozen_fields):
            raise EventDictionaryError(
                "frozen real ventilation concept contains placeholder/synthetic metadata"
            )
        if concept.get("verification_status") != "VERIFIED":
            raise EventDictionaryError("frozen ventilation concept must be VERIFIED")
        for field in (
            "concept_name",
            "source_repository",
            "concept_version",
            "output_schema_version",
            "adapter_version",
        ):
            _nonempty_string(concept.get(field), "invasive_ventilation.concept." + field)
        _validate_exact_commit(concept.get("source_commit"))
        if category_mapping.get("verification_status") != "VERIFIED":
            raise EventDictionaryError("frozen ventilation category mapping must be VERIFIED")
        raw_mapping = _mapping(
            category_mapping.get("raw_to_internal_category_mapping"),
            "raw_to_internal_category_mapping",
        )
        if not raw_mapping:
            raise EventDictionaryError("frozen ventilation category mapping cannot be empty")
        for raw_value, internal_value in raw_mapping.items():
            _nonempty_string(raw_value, "raw ventilation category")
            if internal_value not in VENTILATION_INTERNAL_CATEGORIES:
                raise EventDictionaryError("raw ventilation category has unsupported mapping")
        if "INVASIVE" not in raw_mapping.values():
            raise EventDictionaryError("frozen mapping lacks an invasive category")
        exclusions = _mapping(
            category_mapping.get("explicit_exclusion_semantics"),
            "explicit_exclusion_semantics",
        )
        for category in ("NON_INVASIVE", "HFNC", "ORDINARY_OXYGEN"):
            _nonempty_string(exclusions.get(category), "exclusion_semantics." + category)
        if output_semantics.get("verification_status") != "VERIFIED":
            raise EventDictionaryError("frozen ventilation output semantics must be VERIFIED")
        _nonempty_string(
            output_semantics.get("output_representation"),
            "scientific_output_semantics.output_representation",
        )
        _nonempty_string(
            output_semantics.get("event_time_semantics"),
            "scientific_output_semantics.event_time_semantics",
        )
        if scientific_boundary not in SUPPORTED_BOUNDARY_POLICIES:
            raise EventDictionaryError("frozen ventilation boundary policy is required")
        duplicates = _mapping(
            section.get("duplicate_overlap_policy"), "duplicate_overlap_policy"
        )
        if duplicates.get("verification_status") != "VERIFIED":
            raise EventDictionaryError("frozen ventilation duplicate policy must be VERIFIED")
        _nonempty_string(duplicates.get("real_source_rule"), "duplicate real_source_rule")
        if sofa_status != "VERIFIED_ALIGNED":
            raise EventDictionaryError("SOFA/support ventilation alignment is not verified")
        if sofa_alignment.get("expected_concept_version") != scientific_version:
            raise EventDictionaryError("SOFA/support ventilation concept versions differ")
        _nonempty_string(sofa_alignment.get("evidence_ref"), "sofa_alignment.evidence_ref")
        reviewers = _mapping(section.get("reviewers"), "ventilation reviewers")
        sanskruti = _mapping(reviewers.get("Sanskruti"), "reviewers.Sanskruti")
        if sanskruti.get("status") != "APPROVED":
            raise EventDictionaryError("frozen ventilation concept requires Sanskruti approval")
        _nonempty_string(
            sanskruti.get("evidence_ref"), "reviewers.Sanskruti.evidence_ref"
        )

    return VentilationEventContract(
        status=status,
        real_scientific_execution_authorized=authorized,
        state_rule_version=VENTILATION_STATE_RULE_VERSION,
        normalized_contract_version=VENTILATION_NORMALIZED_CONTRACT_VERSION,
        scientific_concept_name=scientific_name,
        scientific_concept_version=scientific_version,
        scientific_adapter_version=scientific_adapter,
        scientific_boundary_policy=scientific_boundary,
        sofa_alignment_status=sofa_status,
        synthetic_concept_name=synthetic_name,
        synthetic_concept_version=synthetic_version,
        synthetic_adapter_version=synthetic_adapter,
        synthetic_known_categories=known_categories,
        synthetic_invasive_categories=invasive_categories,
        synthetic_non_invasive_categories=non_invasive_categories,
        synthetic_boundary_policy=synthetic_boundary,
    )


def _validate_organ_support_section(value: object) -> OrganSupportLabelContract:
    section = _mapping(value, "organ_support_label")
    status = section.get("status")
    if status not in (DRAFT_STATUS, ORGAN_SUPPORT_FROZEN_STATUS):
        raise EventDictionaryError("unsupported organ-support endpoint status")
    authorized = section.get("real_scientific_execution_authorized")
    if not isinstance(authorized, bool):
        raise EventDictionaryError(
            "organ-support real_scientific_execution_authorized must be boolean"
        )
    if status == DRAFT_STATUS and authorized:
        raise EventDictionaryError(
            "draft organ-support label cannot authorize scientific execution"
        )
    if status == ORGAN_SUPPORT_FROZEN_STATUS and not authorized:
        raise EventDictionaryError("frozen organ-support endpoint must authorize use")
    if section.get("label_rule_version") != ORGAN_SUPPORT_LABEL_RULE_VERSION:
        raise EventDictionaryError("unsupported organ-support label-rule version")
    if (
        section.get("onset_evidence_contract_version")
        != COMPONENT_ONSET_EVIDENCE_VERSION
    ):
        raise EventDictionaryError("unsupported component-onset evidence version")
    if (
        section.get("phase7_handoff_record_contract")
        != PHASE7_HANDOFF_RECORD_CONTRACT
    ):
        raise EventDictionaryError("unsupported Phase-7 handoff record contract")
    expected = {
        "endpoint_name": "new_organ_support_initiation",
        "horizon_hours": 24,
        "future_interval_semantics": "(t,t+24h]",
        "left_boundary_inclusive": False,
        "right_boundary_inclusive": True,
        "component_at_risk_rule": "baseline_component_must_be_OFF",
        "composite_at_risk_rule": "at_least_one_component_OFF_at_t",
        "both_components_on_rule": "ineligible_not_at_risk",
        "positive_rule": "eligible_OFF_to_ON_before_ICU_exit_within_horizon",
        "positive_precedence_over_incomplete_followup": True,
        "negative_rule": "full_24h_followup_and_no_eligible_initiation",
        "censored_rule": "early_ICU_exit_without_prior_eligible_initiation",
        "censored_target": None,
    }
    for field, expected_value in expected.items():
        if section.get(field) != expected_value:
            raise EventDictionaryError(
                "organ-support scientific invariant changed: " + field
            )
    tie = _mapping(
        section.get("onset_outtime_tie_policy"),
        "organ_support_label.onset_outtime_tie_policy",
    )
    if status == DRAFT_STATUS:
        if tie.get("status") != "UNRESOLVED_REQUIRES_VERIFICATION":
            raise EventDictionaryError("onset/outtime tie cannot be silently resolved")
        if tie.get("rule") is not None:
            raise EventDictionaryError("unreviewed onset/outtime tie rule must remain null")
    else:
        if tie.get("status") != "VERIFIED":
            raise EventDictionaryError("frozen onset/outtime tie policy must be VERIFIED")
        _nonempty_string(tie.get("rule"), "onset_outtime_tie_policy.rule")
    unknown = _mapping(
        section.get("unknown_component_state_policy"),
        "organ_support_label.unknown_component_state_policy",
    )
    if unknown.get("current_guard") != "HARD_FAILURE":
        raise EventDictionaryError("unknown support states must fail closed")
    unknown_status = _nonempty_string(
        unknown.get("status"), "unknown_component_state_policy.status"
    )
    if status == ORGAN_SUPPORT_FROZEN_STATUS and unknown_status != "RESOLVED":
        raise EventDictionaryError("frozen unknown-state label policy must be RESOLVED")
    vocabulary = _mapping(
        section.get("audit_reason_vocabulary"),
        "organ_support_label.audit_reason_vocabulary",
    )
    vocabulary_status = _nonempty_string(
        vocabulary.get("status"), "audit_reason_vocabulary.status"
    )
    vocabulary_version = _nonempty_string(
        vocabulary.get("version"), "audit_reason_vocabulary.version"
    )
    if (
        status == ORGAN_SUPPORT_FROZEN_STATUS
        and vocabulary_status != "FROZEN_REVIEWED"
    ):
        raise EventDictionaryError("frozen audit-reason vocabulary must be reviewed")
    reviewers = _mapping(section.get("reviewers"), "organ_support_label.reviewers")
    for reviewer in ("Vedant", "Sanskruti_compatibility"):
        review = _mapping(reviewers.get(reviewer), "organ_support reviewers." + reviewer)
        if review.get("status") not in ("PENDING", "APPROVED"):
            raise EventDictionaryError("unsupported organ-support review status")
        evidence_ref = review.get("evidence_ref")
        if review.get("status") == "APPROVED":
            _nonempty_string(evidence_ref, "organ-support review evidence_ref")
        elif evidence_ref is not None:
            raise EventDictionaryError("pending organ-support review evidence must be null")
        if status == ORGAN_SUPPORT_FROZEN_STATUS and review.get("status") != "APPROVED":
            raise EventDictionaryError(
                "frozen organ-support endpoint requires all reviews"
            )
    artifact = _mapping(
        section.get("real_handoff_artifact"),
        "organ_support_label.real_handoff_artifact",
    )
    if status == DRAFT_STATUS:
        if artifact.get("status") != "BLOCKED_REAL_STATE_DEFINITIONS_AND_DATA_REQUIRED":
            raise EventDictionaryError("draft real organ-support handoff must remain blocked")
        if (
            artifact.get("artifact_ref") is not None
            or artifact.get("artifact_sha256") is not None
        ):
            raise EventDictionaryError("draft organ-support config cannot claim a real artifact")
    elif artifact.get("status") not in (
        "PENDING_POST_FREEZE_GENERATION",
        "FROZEN_COMPATIBLE",
    ):
        raise EventDictionaryError("frozen endpoint has invalid handoff artifact status")
    if status == ORGAN_SUPPORT_FROZEN_STATUS:
        frozen_fields = {
            "tie": tie,
            "unknown": unknown,
            "vocabulary": vocabulary,
            "reviewers": reviewers,
        }
        if _contains_real_concept_placeholder(frozen_fields):
            raise EventDictionaryError(
                "frozen organ-support endpoint contains unresolved/placeholder metadata"
            )
    return OrganSupportLabelContract(
        status=status,
        real_scientific_execution_authorized=authorized,
        label_rule_version=ORGAN_SUPPORT_LABEL_RULE_VERSION,
        onset_evidence_contract_version=COMPONENT_ONSET_EVIDENCE_VERSION,
        phase7_handoff_record_contract=PHASE7_HANDOFF_RECORD_CONTRACT,
        horizon_hours=24,
        future_interval_semantics="(t,t+24h]",
        onset_outtime_tie_status=tie["status"],
        unknown_component_state_policy_status=unknown_status,
        audit_reason_vocabulary_status=vocabulary_status,
        audit_reason_vocabulary_version=vocabulary_version,
    )


def _validate_frozen_provenance(
    vasopressor: Mapping[str, Any], mapping: Mapping[str, Any]
) -> None:
    if mapping.get("verification_status") != "VERIFIED":
        raise EventDictionaryError("frozen qualifying mapping must be VERIFIED")
    source_mapping = _mapping(mapping.get("source_mapping"), "vasopressor.mapping.source_mapping")
    for field in (
        "source_name",
        "agent_identifier_field",
        "event_time_field",
        "interval_start_field",
        "interval_end_field",
        "rate_field",
        "unit_field",
        "derivation_version",
    ):
        _nonempty_string(source_mapping.get(field), "source_mapping." + field)
    _nonempty_string(
        mapping.get("mapping_provenance_version"),
        "vasopressor.mapping.mapping_provenance_version",
    )
    activity = _mapping(
        vasopressor.get("scientific_activity_rule"),
        "vasopressor.scientific_activity_rule",
    )
    if activity.get("verification_status") != "VERIFIED":
        raise EventDictionaryError("frozen activity/rate rule must be VERIFIED")
    _nonempty_string(activity.get("rule_version"), "scientific_activity_rule.rule_version")
    _nonempty_string(
        activity.get("units_and_rate_semantics"),
        "scientific_activity_rule.units_and_rate_semantics",
    )
    duplicate_policy = _mapping(
        vasopressor.get("duplicate_conflict_policy"),
        "vasopressor.duplicate_conflict_policy",
    )
    if duplicate_policy.get("verification_status") != "VERIFIED":
        raise EventDictionaryError("frozen duplicate/conflict policy must be VERIFIED")
    _nonempty_string(
        duplicate_policy.get("real_source_rule"),
        "duplicate_conflict_policy.real_source_rule",
    )
    reviewers = _mapping(vasopressor.get("reviewers"), "vasopressor.reviewers")
    for reviewer in ("Sanskruti", "Vedant"):
        review = _mapping(reviewers.get(reviewer), "reviewers." + reviewer)
        if review.get("status") != "APPROVED":
            raise EventDictionaryError("frozen dictionary requires " + reviewer + " approval")
        _nonempty_string(review.get("evidence_ref"), "reviewers." + reviewer + ".evidence_ref")


def validate_event_dictionary(
    payload: Mapping[str, Any],
    *,
    config_sha256: str,
    require_scientific: bool = False,
) -> EventDictionary:
    """Validate schema and return a typed contract.

    ``require_scientific=True`` is the hard gate for real label/prevalence use.
    A draft dictionary always fails that gate.
    """

    if not is_sha256(config_sha256):
        raise EventDictionaryError("event-dictionary config SHA-256 is malformed")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise EventDictionaryError("unsupported event-dictionary schema version")
    if payload.get("event_dictionary_version") != EVENT_DICTIONARY_VERSION:
        raise EventDictionaryError("unsupported event-dictionary version")
    status = payload.get("status")
    if status not in (DRAFT_STATUS, FROZEN_STATUS):
        raise EventDictionaryError("unsupported event-dictionary status")
    authorized = payload.get("real_scientific_execution_authorized")
    if not isinstance(authorized, bool):
        raise EventDictionaryError("real_scientific_execution_authorized must be boolean")
    if status == DRAFT_STATUS and authorized:
        raise EventDictionaryError("draft dictionary cannot authorize scientific execution")

    vasopressor = _mapping(payload.get("vasopressor"), "vasopressor")
    if vasopressor.get("state_rule_version") != STATE_RULE_VERSION:
        raise EventDictionaryError("unsupported vasopressor state-rule version")
    if vasopressor.get("normalized_contract_version") != NORMALIZED_CONTRACT_VERSION:
        raise EventDictionaryError("unsupported normalized-event contract version")
    if vasopressor.get("endpoint_semantics") != "binary_support_state_only_not_escalation":
        raise EventDictionaryError("vasopressor endpoint must remain binary support state")
    if vasopressor.get("any_qualifying_active_interval_means_on") is not True:
        raise EventDictionaryError("ANY qualifying active interval must mean ON")
    if vasopressor.get("already_on_means_at_risk") is not False:
        raise EventDictionaryError("already-ON support must not be at risk")

    mapping = _mapping(vasopressor.get("mapping"), "vasopressor.mapping")
    scientific_keys = _agent_keys(
        mapping.get("qualifying_agent_keys"),
        "vasopressor.mapping.qualifying_agent_keys",
        allow_empty=status == DRAFT_STATUS,
    )
    mapping_provenance = mapping.get("mapping_provenance_version")
    if mapping_provenance is not None:
        mapping_provenance = _nonempty_string(
            mapping_provenance, "vasopressor.mapping.mapping_provenance_version"
        )
    boundary = _mapping(
        vasopressor.get("scientific_boundary_policy"),
        "vasopressor.scientific_boundary_policy",
    )
    scientific_boundary = boundary.get("interval_convention")
    if scientific_boundary is not None and scientific_boundary not in SUPPORTED_BOUNDARY_POLICIES:
        raise EventDictionaryError("unsupported scientific interval-boundary policy")

    synthetic = _mapping(
        vasopressor.get("synthetic_test_profile"),
        "vasopressor.synthetic_test_profile",
    )
    if synthetic.get("synthetic_only") is not True:
        raise EventDictionaryError("synthetic profile must be marked synthetic_only")
    synthetic_keys = _agent_keys(
        synthetic.get("qualifying_agent_keys"),
        "synthetic_test_profile.qualifying_agent_keys",
        allow_empty=False,
    )
    synthetic_boundary = synthetic.get("interval_convention")
    if synthetic_boundary not in SUPPORTED_BOUNDARY_POLICIES:
        raise EventDictionaryError("unsupported synthetic interval-boundary policy")
    synthetic_provenance = _nonempty_string(
        synthetic.get("mapping_provenance_version"),
        "synthetic_test_profile.mapping_provenance_version",
    )
    ventilation = _validate_ventilation_section(payload.get("invasive_ventilation"))
    organ_support = _validate_organ_support_section(payload.get("organ_support_label"))

    if status == FROZEN_STATUS:
        if not authorized:
            raise EventDictionaryError(
                "frozen dictionary must explicitly authorize scientific execution"
            )
        unresolved = payload.get("unresolved_required_items")
        if not isinstance(unresolved, list) or unresolved:
            raise EventDictionaryError(
                "frozen dictionary cannot retain unresolved required items"
            )
        if scientific_boundary not in SUPPORTED_BOUNDARY_POLICIES:
            raise EventDictionaryError("frozen dictionary requires a verified boundary policy")
        if boundary.get("verification_status") != "VERIFIED":
            raise EventDictionaryError("frozen boundary policy must be VERIFIED")
        _validate_frozen_provenance(vasopressor, mapping)
        frozen_scientific_fields = {
            "mapping": mapping,
            "boundary": boundary,
            "activity": vasopressor.get("scientific_activity_rule"),
            "duplicates": vasopressor.get("duplicate_conflict_policy"),
            "reviewers": vasopressor.get("reviewers"),
        }
        if _contains_placeholder(frozen_scientific_fields):
            raise EventDictionaryError("frozen dictionary contains placeholder identifiers")

    if require_scientific and not (status == FROZEN_STATUS and authorized):
        raise EventDictionaryError(
            "BLOCKED — MIMIC SCHEMA / MAPPING VERIFICATION REQUIRED"
        )

    return EventDictionary(
        schema_version=SCHEMA_VERSION,
        event_dictionary_version=EVENT_DICTIONARY_VERSION,
        status=status,
        real_scientific_execution_authorized=authorized,
        state_rule_version=STATE_RULE_VERSION,
        normalized_contract_version=NORMALIZED_CONTRACT_VERSION,
        scientific_qualifying_agent_keys=scientific_keys,
        scientific_boundary_policy=scientific_boundary,
        mapping_provenance_version=mapping_provenance,
        synthetic_qualifying_agent_keys=synthetic_keys,
        synthetic_boundary_policy=synthetic_boundary,
        synthetic_mapping_provenance_version=synthetic_provenance,
        ventilation=ventilation,
        organ_support=organ_support,
        config_sha256=config_sha256,
    )


def load_event_dictionary(
    path: PathLike, *, require_scientific: bool = False
) -> EventDictionary:
    """Load YAML, validate it, and bind the exact-file SHA-256."""

    config_path = Path(path)
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as error:
        raise EventDictionaryError("event dictionary is unreadable") from error
    if not isinstance(payload, dict):
        raise EventDictionaryError("event dictionary root must be a mapping")
    return validate_event_dictionary(
        payload,
        config_sha256=sha256_file(config_path),
        require_scientific=require_scientific,
    )
