from pathlib import Path

from vedant_infra.g3 import audit_g3, validate_g3_marker


ROOT = Path(__file__).resolve().parents[1]


def test_official_g3_readiness_and_test_nonuse():
    marker = ROOT / "artifacts/governance/g3_freeze.json"
    report = audit_g3(ROOT, scope="real", allow_authorized_state=marker.exists())
    by_name = {item.name: item for item in report.items}
    assert by_name["test_nonuse"].status == "PASS"
    if marker.exists():
        assert report.overall == "PASS"
        validate_g3_marker(marker, ROOT, expected_scope="real")
    elif (ROOT / "artifacts/models/selected_models_v1.json").exists():
        assert report.overall == "PASS"
    else:
        assert report.overall == "BLOCKED"
