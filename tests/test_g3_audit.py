from pathlib import Path

from g3_helpers import build_complete_g3_fixture
from vedant_infra.g3 import audit_g3, validate_g3_marker


ROOT = Path(__file__).resolve().parents[1]


def test_real_repository_g3_readiness_tracks_governed_stage():
    marker = ROOT / "artifacts/governance/g3_freeze.json"
    selected = ROOT / "artifacts/models/selected_models_v1.json"
    report = audit_g3(ROOT, scope="real", allow_authorized_state=marker.exists())
    assert report.test_data_accessed is False
    if marker.exists():
        assert report.overall == "PASS"
        validate_g3_marker(marker, ROOT, expected_scope="real")
    elif selected.exists():
        assert report.overall == "PASS"
    else:
        assert report.overall == "BLOCKED"


def test_complete_synthetic_prerequisites_pass(tmp_path):
    build_complete_g3_fixture(tmp_path)
    report = audit_g3(tmp_path, scope="synthetic_test")
    assert report.overall == "PASS"
    assert all(item.status == "PASS" for item in report.items)
