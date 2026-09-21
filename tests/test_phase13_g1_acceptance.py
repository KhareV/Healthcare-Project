import json
from pathlib import Path

from acceptance.phase13 import Gate, _hash_inventory, audit_production


ROOT = Path(__file__).resolve().parents[1]


def test_phase13_gate_is_fail_closed():
    gate = Gate()
    gate.require("X", "test", True, "ok", "PHASE_13")
    assert gate.accepted
    gate.require("Y", "test", False, "bad", "PHASE_13")
    assert not gate.accepted


def test_phase13_inventory_resolves_real_current_artifacts():
    inventory = _hash_inventory(ROOT)
    assert len(inventory) >= 20
    assert all((ROOT / item["path"]).is_file() and len(item["sha256"]) == 64 for item in inventory.values())


def test_phase13_production_audit_passes_against_clean_focused_junit(tmp_path):
    junit = tmp_path / "focused.xml"
    junit.write_text('<testsuite tests="1" failures="0" errors="0" skipped="0"><testcase name="sentinel"/></testsuite>')
    gate, qa, golden = audit_production(ROOT, junit)
    assert gate.accepted, [item for item in gate.checks if item.status != "PASS"]
    assert qa["subjects"] == 2000 and qa["canonical_rows"] == 12222
    assert qa["test_accessed"] is False and qa["phase12_performance_used_for_acceptance"] is False
    assert len(golden) == 10 and all(case["status"] == "PASS" for case in golden)


def test_authorizing_marker_is_never_partial_if_present():
    marker = ROOT / "artifacts/acceptance/g1_synthetic_data_freeze_v1.json"
    if marker.exists():
        payload = json.loads(marker.read_text())
        assert payload["status"] == "ACCEPTED_SYNTHETIC_DATA_LABEL_FREEZE"
        assert payload["test_accessed"] is False
        assert payload["phase12_results_status"] == "VALID_UNCHANGED_PARENT_ARTIFACTS"
        assert payload["model_performance_acceptance_threshold_used"] is False
