from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from labels.event_dict import (
    DRAFT_STATUS,
    EventDictionaryError,
    VENTILATION_FROZEN_STATUS,
    load_event_dictionary,
    validate_event_dictionary,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/event_dict_v1.yaml"


def payload():
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def validate(value):
    return validate_event_dictionary(value, config_sha256="a" * 64)


def structurally_complete_test_only_frozen_payload():
    """Build an in-memory validator attack; this is never a scientific config."""

    value = payload()
    section = value["invasive_ventilation"]
    section["status"] = VENTILATION_FROZEN_STATUS
    section["real_scientific_execution_authorized"] = True
    section["concept"] = {
        "verification_status": "VERIFIED",
        "concept_name": "TEST_ONLY_CONCEPT",
        "source_repository": "https://invalid.local/test-only",
        "source_commit": "a" * 40,
        "concept_version": "test-only-concept-v1",
        "output_schema_version": "test-only-output-v1",
        "adapter_version": "test-only-adapter-v1",
    }
    section["category_mapping"] = {
        "verification_status": "VERIFIED",
        "raw_to_internal_category_mapping": {
            "test-invasive": "INVASIVE",
            "test-niv": "NON_INVASIVE",
        },
        "invasive_internal_category": "INVASIVE",
        "explicit_exclusion_semantics": {
            "NON_INVASIVE": "test-only explicit exclusion",
            "HFNC": "test-only explicit exclusion",
            "ORDINARY_OXYGEN": "test-only explicit exclusion",
        },
        "unknown_category_policy": "HARD_FAILURE",
    }
    section["scientific_output_semantics"] = {
        "verification_status": "VERIFIED",
        "output_representation": "test-only intervals",
        "interval_convention": "CLOSED_OPEN",
        "event_time_semantics": "test-only event time",
    }
    section["duplicate_overlap_policy"] = {
        "verification_status": "VERIFIED",
        "real_source_rule": "test-only rule",
    }
    section["sofa_alignment"] = {
        "status": "VERIFIED_ALIGNED",
        "expected_concept_version": "test-only-concept-v1",
        "evidence_ref": "docs/test-only-evidence.md",
    }
    section["reviewers"]["Sanskruti"] = {
        "status": "APPROVED",
        "evidence_ref": "docs/test-only-review.md",
    }
    return value


def test_checked_in_ventilation_section_is_draft_and_non_authorizing():
    contract = load_event_dictionary(CONFIG).ventilation
    assert contract.status == DRAFT_STATUS
    assert not contract.real_scientific_execution_authorized
    assert not contract.is_frozen_for_scientific_use
    assert contract.scientific_concept_name is None
    assert contract.scientific_concept_version is None
    assert contract.scientific_boundary_policy is None


def test_phase1_vasopressor_contract_remains_compatible():
    contract = load_event_dictionary(CONFIG)
    assert contract.state_rule_version == "vasopressor_state_rule_v1"
    assert contract.synthetic_qualifying_agent_keys == {
        "QUALIFYING_AGENT_A",
        "QUALIFYING_AGENT_B",
    }


def test_synthetic_category_contract_is_explicit_and_disjoint():
    contract = load_event_dictionary(CONFIG).ventilation
    assert contract.synthetic_known_categories == {
        "INVASIVE",
        "NON_INVASIVE",
        "HFNC",
        "ORDINARY_OXYGEN",
    }
    assert contract.synthetic_invasive_categories == {"INVASIVE"}
    assert contract.synthetic_non_invasive_categories == {
        "NON_INVASIVE",
        "HFNC",
        "ORDINARY_OXYGEN",
    }
    assert not (
        contract.synthetic_invasive_categories
        & contract.synthetic_non_invasive_categories
    )


def test_unknown_ventilation_rule_or_contract_version_fails():
    bad_rule = payload()
    bad_rule["invasive_ventilation"]["state_rule_version"] = "unknown"
    with pytest.raises(EventDictionaryError, match="state-rule version"):
        validate(bad_rule)

    bad_contract = payload()
    bad_contract["invasive_ventilation"]["normalized_contract_version"] = "unknown"
    with pytest.raises(EventDictionaryError, match="normalized ventilation"):
        validate(bad_contract)


def test_draft_ventilation_section_cannot_authorize_real_execution():
    attacked = payload()
    attacked["invasive_ventilation"]["real_scientific_execution_authorized"] = True
    with pytest.raises(EventDictionaryError, match="draft ventilation concept"):
        validate(attacked)


def test_unknown_category_policy_cannot_be_silently_off():
    attacked = payload()
    attacked["invasive_ventilation"]["category_mapping"][
        "unknown_category_policy"
    ] = "MAP_TO_OFF"
    with pytest.raises(EventDictionaryError, match="fail closed"):
        validate(attacked)


def test_synthetic_profile_cannot_be_presented_as_real():
    attacked = payload()
    section = attacked["invasive_ventilation"]
    section["status"] = VENTILATION_FROZEN_STATUS
    section["real_scientific_execution_authorized"] = True
    section["concept"]["concept_name"] = section["synthetic_test_profile"][
        "concept_name"
    ]
    with pytest.raises(EventDictionaryError, match="placeholder/synthetic"):
        validate(attacked)


@pytest.mark.parametrize("commit", ["PLACEHOLDER", "unknown", "TODO", "synthetic"])
def test_placeholder_commit_is_rejected_in_real_mode(commit):
    attacked = payload()
    section = attacked["invasive_ventilation"]
    section["status"] = VENTILATION_FROZEN_STATUS
    section["real_scientific_execution_authorized"] = True
    section["concept"]["source_commit"] = commit
    with pytest.raises(EventDictionaryError, match="placeholder/synthetic"):
        validate(attacked)


def test_exact_commit_shape_is_required_for_structurally_frozen_contract():
    attacked = structurally_complete_test_only_frozen_payload()
    attacked["invasive_ventilation"]["concept"]["source_commit"] = "not-a-git-hash"
    with pytest.raises(EventDictionaryError, match="exact Git hash"):
        validate(attacked)


def test_sofa_support_concept_version_mismatch_fails_closed():
    attacked = structurally_complete_test_only_frozen_payload()
    attacked["invasive_ventilation"]["sofa_alignment"][
        "expected_concept_version"
    ] = "different-concept-v2"
    with pytest.raises(EventDictionaryError, match="concept versions differ"):
        validate(attacked)


def test_sanskruti_approval_is_required_for_real_freeze():
    attacked = structurally_complete_test_only_frozen_payload()
    attacked["invasive_ventilation"]["reviewers"]["Sanskruti"][
        "status"
    ] = "PENDING"
    with pytest.raises(EventDictionaryError, match="Sanskruti approval"):
        validate(attacked)


def test_validation_does_not_mutate_phase2_payload():
    source = payload()
    before = deepcopy(source)
    validate(source)
    assert source == before

