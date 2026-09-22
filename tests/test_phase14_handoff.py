import copy
import json
from pathlib import Path

import pytest

from data.synthetic.provenance import sha256_file
from integration.phase14 import (
    HANDOFF_PATH,
    HandoffError,
    build_handoff,
    pulkit_receiver_acceptance,
    validate_handoff,
    vedant_receiver_acceptance,
)


ROOT = Path(__file__).resolve().parents[1]


def _manifest():
    return json.loads((ROOT / HANDOFF_PATH).read_text(encoding="utf-8"))


def _write(tmp_path, payload):
    path = tmp_path / "handoff.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_phase14_handoff_is_valid_and_governance_locked():
    handoff = validate_handoff(ROOT)
    assert handoff["status"] == "SANSKRUTI_TRACK_HANDOFF_FROZEN"
    assert handoff["test_accessed"] is False
    assert handoff["g3_created"] is False
    assert handoff["selected_models_manifest_created"] is False
    assert handoff["final_model_selection"] == "PENDING_DOWNSTREAM_VEDANT"
    # The handoff records Phase 14's historical state; authorized downstream
    # selection does not rewrite or invalidate it.
    assert (ROOT / "artifacts/models/selected_models_v1.json").is_file()


def test_phase14_dimensions_and_counts_are_artifact_derived():
    handoff = validate_handoff(ROOT)
    assert handoff["dimensions"] == {
        "T": 8,
        "F": 21,
        "raw_static_S": 3,
        "transformed_static_S": 8,
        "xgb_flat_D": 520,
    }
    assert handoff["row_counts"] == {
        "total": 12222,
        "train": 8626,
        "validation": 1754,
        "test": 1842,
    }


@pytest.mark.parametrize(
    ("parent", "replacement"),
    (
        ("g1_acceptance", "docs/governance/project_scope_v2.md"),
        ("feature_schema", "configs/synthetic/feature_schema_v1.json"),
        ("preprocessor", "configs/synthetic/feature_schema_v2.json"),
        ("split", "configs/synthetic/feature_schema_v2.json"),
        ("event_dictionary", "configs/event_dict_v1.yaml"),
        ("sofa_spec", "configs/synthetic/feature_schema_v2.json"),
        ("flat_feature_map", "configs/synthetic/feature_schema_v2.json"),
    ),
)
def test_non_authoritative_scientific_parent_fails_closed(tmp_path, parent, replacement):
    payload = _manifest()
    payload["parents"][parent]["path"] = replacement
    payload["parents"][parent]["sha256"] = sha256_file(ROOT / replacement)
    with pytest.raises(HandoffError):
        validate_handoff(ROOT, _write(tmp_path, payload))


def test_wrong_g1_hash_fails_closed(tmp_path):
    payload = _manifest()
    payload["g1_sha256"] = "0" * 64
    with pytest.raises(HandoffError, match="G1 binding"):
        validate_handoff(ROOT, _write(tmp_path, payload))


def test_modified_parent_hash_fails_closed(tmp_path):
    payload = _manifest()
    payload["parents"]["preprocessor"]["sha256"] = "0" * 64
    with pytest.raises(HandoffError, match="parent mismatch"):
        validate_handoff(ROOT, _write(tmp_path, payload))


def test_missing_parent_fails_closed(tmp_path):
    payload = _manifest()
    payload["parents"]["preprocessor"]["path"] = "missing.json"
    with pytest.raises(HandoffError, match="parent mismatch"):
        validate_handoff(ROOT, _write(tmp_path, payload))


def test_test_accessed_attack_fails_closed(tmp_path):
    payload = _manifest()
    payload["test_accessed"] = True
    with pytest.raises(HandoffError, match="test/G3"):
        validate_handoff(ROOT, _write(tmp_path, payload))


def test_reordered_feature_schema_hash_attack_fails_closed(tmp_path):
    payload = _manifest()
    payload["parents"]["feature_schema"]["sha256"] = "f" * 64
    with pytest.raises(HandoffError, match="parent mismatch"):
        validate_handoff(ROOT, _write(tmp_path, payload))


def test_vedant_end_to_end_receiver_acceptance():
    result = vedant_receiver_acceptance(ROOT, ROOT / HANDOFF_PATH)
    assert result["status"] == "VEDANT_RECEIVER_READY"
    assert result["information_parity"] == "PASS_EXACT_FLATTEN_UNFLATTEN"
    assert result["tensor_shape"] == [8, 21]
    assert result["preprocessor_fit_calls"] == 0
    assert result["scientific_training_performed"] is False
    assert all(value.startswith("PASS_") for value in result["best_xgb_artifacts"].values())


def test_pulkit_end_to_end_receiver_acceptance():
    result = pulkit_receiver_acceptance(ROOT, ROOT / HANDOFF_PATH)
    assert result["status"] == "PULKIT_DATA_RECEIVER_READY"
    assert result["direct_vs_serving_equality"] == "PASS_EXACT"
    assert result["support_state_compatibility"] == "PASS_EXISTING_PULKIT_ENGINES"
    assert result["current_sofa_equality"] == "PASS_EXACT"
    assert result["serving_fit_calls"] == 0
    assert result["selected_model_bundle_status"] == "NOT_YET_FROZEN_DOWNSTREAM"


def test_phase14_artifacts_are_portable_and_secret_free():
    files = list((ROOT / "artifacts/handoffs").glob("*.json"))
    assert files
    forbidden = ("/Users/", "/home/", "C:\\Users\\", "api_key", "password", "secret_key")
    for path in files:
        text = path.read_text(encoding="utf-8").lower()
        assert not any(token.lower() in text for token in forbidden), path


def test_handoff_generation_is_deterministic():
    frozen_commit = validate_handoff(ROOT)["code_commit"]
    path = build_handoff(ROOT, code_commit=frozen_commit)
    first = path.read_bytes()
    assert build_handoff(ROOT, code_commit=frozen_commit).read_bytes() == first


def test_phase14_status_is_not_final_project_completion():
    status = json.loads((ROOT / "artifacts/handoffs/sanskruti_phase14_status_v1.json").read_text())
    assert status["status"] == "SANSKRUTI_TRACK_COMPLETE"
    assert status["g2_readiness"] == "READY_FOR_DOWNSTREAM_MODEL_INTEGRATION"
    assert status["final_model_selection"] == "PENDING_DOWNSTREAM_VEDANT"
    assert status["g3"] == "NOT_CREATED"
    assert status["final_test"] == "NEVER_OPENED"
