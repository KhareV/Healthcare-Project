from pathlib import Path

from vedant_infra.g3 import audit_g3


ROOT = Path(__file__).resolve().parents[1]


def test_official_g3_stays_blocked_and_test_nonuse_passes():
    report = audit_g3(ROOT, scope="real")
    assert report.overall == "BLOCKED"
    by_name = {item.name: item for item in report.items}
    assert by_name["test_nonuse"].status == "PASS"
    assert "PULKIT EXPLANATION ADAPTER" in by_name["explanation_adapters"].detail
    assert not (ROOT / "artifacts/governance/g3_freeze.json").exists()
