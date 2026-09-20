import json
from pathlib import Path

import pytest

from data.acceptance import (
    DataAcceptanceError,
    audit_real_data,
    require_accepted_real_data,
    write_acceptance_report,
)
from real_data_helpers import build_real_handoff_fixture
from vedant_infra.hashing import sha256_file


CONFIG = Path(__file__).resolve().parents[1] / "configs/data_acceptance_v1.json"


def test_complete_fixture_produces_reproducible_accepted_report(tmp_path):
    manifest = build_real_handoff_fixture(tmp_path)
    first = audit_real_data(tmp_path, manifest_path=manifest, config_path=CONFIG)
    second = audit_real_data(tmp_path, manifest_path=manifest, config_path=CONFIG)
    assert first == second
    assert first["overall_status"] == "ACCEPTED"
    assert first["test_labels_accessed"] is False
    path = write_acceptance_report(tmp_path / "accepted.json", first)
    assert require_accepted_real_data(path, sha256_file(path))["overall_status"] == "ACCEPTED"


def test_report_hash_tampering_blocks_phase19(tmp_path):
    manifest = build_real_handoff_fixture(tmp_path)
    report = audit_real_data(tmp_path, manifest_path=manifest, config_path=CONFIG)
    path = write_acceptance_report(tmp_path / "accepted.json", report)
    expected = sha256_file(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["phase19_authorized"] = False
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(DataAcceptanceError, match="hash mismatch"):
        require_accepted_real_data(path, expected)


def test_missing_real_inputs_produce_blocked_report_with_support_blocker(tmp_path):
    report = audit_real_data(tmp_path, config_path=CONFIG)
    assert report["overall_status"] == "BLOCKED"
    assert "BLOCKED — PULKIT REAL ORGAN-SUPPORT LABEL ARTIFACT REQUIRED" in report["unresolved_blockers"]
    assert report["phase19_authorized"] is False


def test_missing_support_handoff_blocks_only_support_in_complete_fixture(tmp_path):
    manifest = build_real_handoff_fixture(tmp_path, include_support=False)
    report = audit_real_data(tmp_path, manifest_path=manifest, config_path=CONFIG)
    assert report["overall_status"] == "BLOCKED"
    assert report["unresolved_blockers"] == [
        "BLOCKED — PULKIT REAL ORGAN-SUPPORT LABEL ARTIFACT REQUIRED"
    ]
    by_name = {item["name"]: item["status"] for item in report["checks"]}
    assert by_name["recovery_labels"] == "PASS"
    assert by_name["icu_time_labels"] == "PASS"
    assert by_name["support_handoff"] == "BLOCKED"
