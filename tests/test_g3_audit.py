import json
from pathlib import Path

import pytest

from g3_helpers import build_complete_g3_fixture
from vedant_infra.g3 import G3FreezeError, audit_g3, validate_g3_marker


ROOT = Path(__file__).resolve().parents[1]


def _final_test_access_consumed() -> bool:
    path = ROOT / "artifacts/governance/test_access_state.json"
    return path.is_file() and json.loads(path.read_text()).get("state") != "AUTHORIZED_NOT_RUN"


def test_real_repository_g3_readiness_tracks_governed_stage():
    marker = ROOT / "artifacts/governance/g3_freeze.json"
    selected = ROOT / "artifacts/models/selected_models_v1.json"
    report = audit_g3(ROOT, scope="real", allow_authorized_state=marker.exists())
    assert report.test_data_accessed is False
    if marker.exists() and _final_test_access_consumed():
        # Stage 5's one-time final-test evaluation has legitimately been
        # consumed: experiments/artifacts.csv/registry.csv now bind the
        # registered final-test results, so the live readiness audit
        # correctly stops passing (G4 is authoritative from this point;
        # see artifacts/governance/g4_test_evaluation_freeze_v1.json). This
        # is the expected post-test transition, not corruption.
        assert report.overall == "BLOCKED"
        with pytest.raises(G3FreezeError):
            validate_g3_marker(marker, ROOT, expected_scope="real")
    elif marker.exists():
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
