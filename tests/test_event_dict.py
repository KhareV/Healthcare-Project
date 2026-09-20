from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from labels.event_dict import (
    DRAFT_STATUS,
    EventDictionaryError,
    load_event_dictionary,
    validate_event_dictionary,
)
from vedant_infra.hashing import is_sha256, sha256_file


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/event_dict_v1.yaml"


def payload():
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def validate(value, *, require_scientific=False):
    return validate_event_dictionary(
        value,
        config_sha256="a" * 64,
        require_scientific=require_scientific,
    )


def test_checked_in_dictionary_is_draft_non_authorizing_and_hashed():
    contract = load_event_dictionary(CONFIG)
    assert contract.status == DRAFT_STATUS
    assert not contract.real_scientific_execution_authorized
    assert not contract.is_frozen_for_scientific_use
    assert contract.scientific_qualifying_agent_keys == frozenset()
    assert contract.scientific_boundary_policy is None
    assert contract.mapping_provenance_version is None
    assert contract.config_sha256 == sha256_file(CONFIG)
    assert is_sha256(contract.config_sha256)


def test_draft_dictionary_can_drive_only_symbolic_synthetic_profile():
    contract = load_event_dictionary(CONFIG)
    assert contract.synthetic_qualifying_agent_keys == {
        "QUALIFYING_AGENT_A",
        "QUALIFYING_AGENT_B",
    }
    assert contract.synthetic_boundary_policy == "CLOSED_OPEN"
    assert contract.synthetic_mapping_provenance_version.endswith("NOT_MIMIC")


def test_draft_dictionary_denies_real_scientific_execution():
    with pytest.raises(EventDictionaryError, match="MIMIC SCHEMA / MAPPING"):
        load_event_dictionary(CONFIG, require_scientific=True)


def test_unknown_schema_and_state_rule_versions_fail():
    unknown_schema = payload()
    unknown_schema["schema_version"] = "event_dict_schema_v999"
    with pytest.raises(EventDictionaryError, match="schema version"):
        validate(unknown_schema)

    unknown_rule = payload()
    unknown_rule["vasopressor"]["state_rule_version"] = "unknown_rule"
    with pytest.raises(EventDictionaryError, match="state-rule version"):
        validate(unknown_rule)


def test_malformed_config_hash_fails():
    with pytest.raises(EventDictionaryError, match="SHA-256"):
        validate_event_dictionary(payload(), config_sha256="not-a-hash")


def test_draft_cannot_claim_scientific_authorization():
    attacked = payload()
    attacked["real_scientific_execution_authorized"] = True
    with pytest.raises(EventDictionaryError, match="draft dictionary cannot authorize"):
        validate(attacked)


def test_frozen_dictionary_requires_nonempty_mapping_and_provenance():
    attacked = payload()
    attacked["status"] = "FROZEN_SCIENTIFIC_CONTRACT"
    attacked["real_scientific_execution_authorized"] = True
    with pytest.raises(EventDictionaryError, match="cannot be empty"):
        validate(attacked)

    attacked["vasopressor"]["mapping"]["qualifying_agent_keys"] = [
        "SYNTHETIC_KEY_NOT_MIMIC"
    ]
    with pytest.raises(EventDictionaryError, match="unresolved required items"):
        validate(attacked)

    attacked["unresolved_required_items"] = []
    with pytest.raises(EventDictionaryError, match="verified boundary policy"):
        validate(attacked)


def test_unverified_mapping_cannot_be_frozen_even_with_symbolic_key():
    attacked = payload()
    attacked["status"] = "FROZEN_SCIENTIFIC_CONTRACT"
    attacked["real_scientific_execution_authorized"] = True
    attacked["unresolved_required_items"] = []
    attacked["vasopressor"]["mapping"]["qualifying_agent_keys"] = [
        "SYNTHETIC_KEY_NOT_MIMIC"
    ]
    with pytest.raises(EventDictionaryError, match="verified boundary policy"):
        validate(attacked)


def test_synthetic_profile_must_be_explicit_and_nonempty():
    attacked = payload()
    attacked["vasopressor"]["synthetic_test_profile"]["synthetic_only"] = False
    with pytest.raises(EventDictionaryError, match="synthetic_only"):
        validate(attacked)

    empty = payload()
    empty["vasopressor"]["synthetic_test_profile"]["qualifying_agent_keys"] = []
    with pytest.raises(EventDictionaryError, match="cannot be empty"):
        validate(empty)


def test_already_on_semantics_cannot_be_reversed_in_config():
    attacked = payload()
    attacked["vasopressor"]["already_on_means_at_risk"] = True
    with pytest.raises(EventDictionaryError, match="must not be at risk"):
        validate(attacked)


def test_unsupported_boundary_policy_fails():
    attacked = payload()
    attacked["vasopressor"]["synthetic_test_profile"][
        "interval_convention"
    ] = "UNDECLARED_POLICY"
    with pytest.raises(EventDictionaryError, match="boundary policy"):
        validate(attacked)


def test_loader_rejects_non_mapping_yaml(tmp_path):
    candidate = tmp_path / "event_dict.yaml"
    candidate.write_text("- not\n- a\n- mapping\n", encoding="utf-8")
    with pytest.raises(EventDictionaryError, match="root must be a mapping"):
        load_event_dictionary(candidate)


def test_validation_does_not_mutate_payload():
    source = payload()
    before = deepcopy(source)
    validate(source)
    assert source == before
