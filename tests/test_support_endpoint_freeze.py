import json
import shutil
from pathlib import Path

import pytest

from labels.endpoint_freeze import (
    ACTIVE_SYNTHETIC_STATUS,
    EndpointFreezeError,
    audit_real_endpoint_freeze,
    create_real_endpoint_freeze,
    create_synthetic_endpoint_freeze,
    validate_freeze_record,
    write_blocked_real_freeze_audit,
)
from vedant_infra.hashing import sha256_file


ROOT = Path(__file__).resolve().parents[1]


def _mini_root(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    for relative in (
        "configs/event_dict_v1.yaml",
        "configs/timestamp_spec_v1.yaml",
        "configs/support_endpoint_signoff_v1.json",
        "src/labels/organ_support.py",
    ):
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
    return root


def test_real_freeze_audit_is_blocked_without_accessing_labels():
    audit = audit_real_endpoint_freeze(ROOT)
    assert audit["overall"] == "BLOCKED"
    assert audit["label_or_prevalence_data_accessed"] is False
    assert "phase1_vasopressor_real_definition" in audit["blockers"]
    assert "phase2_ventilation_real_definition" in audit["blockers"]
    assert "phase3_composite_real_definition" in audit["blockers"]
    assert "team_endpoint_signoff" in audit["blockers"]
    assert "repository_commit_provenance" in audit["blockers"]


def test_real_freeze_creation_refuses_and_writes_nothing(tmp_path):
    target = tmp_path / "freeze.json"
    with pytest.raises(EndpointFreezeError, match="ENDPOINT FREEZE = BLOCKED"):
        create_real_endpoint_freeze(ROOT, target)
    assert not target.exists()


def test_blocked_audit_record_never_authorizes_prevalence(tmp_path):
    root = _mini_root(tmp_path)
    target = root / "artifacts/labels/support_endpoint_freeze_v1.json"
    digest = write_blocked_real_freeze_audit(root, target)
    record = json.loads(target.read_text())
    assert digest == sha256_file(target)
    assert record["freeze_active"] is False
    assert record["prevalence_access_authorized"] is False
    assert record["audit"]["label_or_prevalence_data_accessed"] is False


def test_synthetic_freeze_is_active_but_cannot_authorize_real(tmp_path):
    root = _mini_root(tmp_path)
    target = root / "artifacts/labels/synthetic/freeze.json"
    digest = create_synthetic_endpoint_freeze(root, target)
    record = validate_freeze_record(
        root, target, expected_sha256=digest, requested_scope="synthetic"
    )
    assert record["freeze_status"] == ACTIVE_SYNTHETIC_STATUS
    assert record["real_data_access_authorized"] is False
    with pytest.raises(EndpointFreezeError, match="scope"):
        validate_freeze_record(root, target, expected_sha256=digest, requested_scope="real")


def test_exact_freeze_hash_and_dependency_mutation_are_enforced(tmp_path):
    root = _mini_root(tmp_path)
    target = root / "artifacts/labels/synthetic/freeze.json"
    digest = create_synthetic_endpoint_freeze(root, target)
    with pytest.raises(EndpointFreezeError, match="hash mismatch"):
        validate_freeze_record(root, target, expected_sha256="0" * 64, requested_scope="synthetic")
    event_path = root / "configs/event_dict_v1.yaml"
    event_path.write_text(event_path.read_text() + "\n# mutation\n")
    with pytest.raises(EndpointFreezeError, match="changed after endpoint freeze"):
        validate_freeze_record(root, target, expected_sha256=digest, requested_scope="synthetic")


def test_freeze_artifacts_are_immutable(tmp_path):
    root = _mini_root(tmp_path)
    target = root / "artifacts/labels/synthetic/freeze.json"
    create_synthetic_endpoint_freeze(root, target)
    payload = json.loads(target.read_text())
    payload["freeze_active"] = False
    target.write_text(json.dumps(payload))
    with pytest.raises(EndpointFreezeError, match="different bytes"):
        create_synthetic_endpoint_freeze(root, target)


def test_signoff_file_contains_no_fabricated_approvals():
    signoff = json.loads((ROOT / "configs/support_endpoint_signoff_v1.json").read_text())
    assert signoff["schema_status"].startswith("UNLOCKED_")
    assert {entry["decision"] for entry in signoff["reviewers"].values()} == {"PENDING"}

