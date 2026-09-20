import json

import pytest

from g3_helpers import build_complete_g3_fixture, rewrite_manifest_hash
from vedant_infra.g3 import G3FreezeError, audit_g3, freeze_g3


def test_incomplete_state_cannot_create_marker(tmp_path):
    paths = build_complete_g3_fixture(tmp_path)
    paths["threshold"].unlink()
    marker = tmp_path / "synthetic-marker.json"
    with pytest.raises(G3FreezeError, match="G3 PREREQUISITE"):
        freeze_g3(
            tmp_path,
            marker_path=marker,
            scope="synthetic_test",
        )
    assert not marker.exists()


def test_synthetic_active_flow_uses_mixed_families(tmp_path):
    paths = build_complete_g3_fixture(tmp_path)
    marker = freeze_g3(
        tmp_path,
        marker_path=tmp_path / "synthetic-marker.json",
        scope="synthetic_test",
        created_at_utc="2026-09-18T00:00:00Z",
    )
    payload = json.loads(marker.read_text(encoding="utf-8"))
    assert payload["freeze_status"] == "ACTIVE"
    assert payload["freeze_scope"] == "synthetic_test"
    assert {
        task: entry["family"]
        for task, entry in payload["tasks"].items()
    } == {
        "recovery": "xgboost",
        "icu_stay_time": "gru",
        "organ_support": "xgboost",
    }
    assert not (
        tmp_path / "artifacts/governance/g3_freeze.json"
    ).exists()


def test_missing_task_and_wrong_explanation_route_block(tmp_path):
    paths = build_complete_g3_fixture(tmp_path)
    rewrite_manifest_hash(
        paths["manifest"],
        lambda payload: payload["tasks"].pop("organ_support"),
    )
    assert audit_g3(
        tmp_path, scope="synthetic_test"
    ).overall == "BLOCKED"

    other = tmp_path / "other"
    paths = build_complete_g3_fixture(other)
    rewrite_manifest_hash(
        paths["manifest"],
        lambda payload: payload["tasks"]["recovery"].update(
            {"explanation_method": "integrated_gradients"}
        ),
    )
    assert audit_g3(
        other, scope="synthetic_test"
    ).overall == "BLOCKED"
