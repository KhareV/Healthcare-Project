import json
from pathlib import Path

import pytest

from vedant_infra.g3 import G3FreezeError, audit_g3, validate_g3_marker


ROOT = Path(__file__).resolve().parents[1]


def _final_test_access_consumed() -> bool:
    path = ROOT / "artifacts/governance/test_access_state.json"
    return path.is_file() and json.loads(path.read_text()).get("state") != "AUTHORIZED_NOT_RUN"


def test_official_g3_readiness_and_test_nonuse():
    marker = ROOT / "artifacts/governance/g3_freeze.json"
    report = audit_g3(ROOT, scope="real", allow_authorized_state=marker.exists())
    by_name = {item.name: item for item in report.items}
    if marker.exists() and _final_test_access_consumed():
        # Stage 5's one-time final-test evaluation has legitimately been
        # consumed. "test_nonuse" is defined to fail once access has been
        # granted/consumed -- this is the expected post-test transition
        # (G4 is authoritative from this point; see
        # artifacts/governance/g4_test_evaluation_freeze_v1.json), not
        # corruption of G3 itself.
        assert by_name["test_nonuse"].status == "BLOCKED"
        assert report.overall == "BLOCKED"
        with pytest.raises(G3FreezeError):
            validate_g3_marker(marker, ROOT, expected_scope="real")
        return
    assert by_name["test_nonuse"].status == "PASS"
    if marker.exists():
        assert report.overall == "PASS"
        validate_g3_marker(marker, ROOT, expected_scope="real")
    elif (ROOT / "artifacts/models/selected_models_v1.json").exists():
        assert report.overall == "PASS"
    else:
        assert report.overall == "BLOCKED"
