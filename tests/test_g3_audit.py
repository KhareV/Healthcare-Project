from pathlib import Path

from g3_helpers import build_complete_g3_fixture
from vedant_infra.g3 import audit_g3


ROOT = Path(__file__).resolve().parents[1]


def test_real_repository_audit_is_blocked_without_opening_test():
    report = audit_g3(ROOT, scope="real")
    assert report.overall == "BLOCKED"
    assert report.test_data_accessed is False
    assert any("selected_models_manifest" in item for item in report.blockers)
    assert any("explanation_adapters" in item for item in report.blockers)
    assert not (ROOT / "artifacts/governance/g3_freeze.json").exists()


def test_complete_synthetic_prerequisites_pass(tmp_path):
    build_complete_g3_fixture(tmp_path)
    report = audit_g3(tmp_path, scope="synthetic_test")
    assert report.overall == "PASS"
    assert all(item.status == "PASS" for item in report.items)
