import pytest

from g3_helpers import build_complete_g3_fixture
from vedant_infra.g3 import (
    G3FreezeError,
    freeze_g3,
    guarded_test_access,
    invalidate_for_reset,
    load_access_state,
    require_development_change_allowed,
)


def test_reset_preserves_history_and_requires_new_freeze(tmp_path):
    build_complete_g3_fixture(tmp_path)
    marker = freeze_g3(
        tmp_path,
        marker_path=tmp_path / "synthetic-marker.json",
        scope="synthetic_test",
    )
    guarded_test_access(
        tmp_path,
        lambda: "result",
        marker_path=marker,
        expected_scope="synthetic_test",
    )
    for change in ("model_selection", "threshold"):
        with pytest.raises(G3FreezeError, match="formal reset"):
            require_development_change_allowed(tmp_path, change)
    archived = invalidate_for_reset(
        tmp_path,
        reason="synthetic scientific defect",
        marker_path=marker,
    )
    assert archived.is_file()
    assert not marker.exists()
    state = load_access_state(
        tmp_path / "artifacts/governance/test_access_state.json"
    )
    assert state["state"] == "INVALIDATED_BY_RESET"
    assert [entry["event"] for entry in state["history"]] == [
        "G3_FREEZE_CREATED",
        "FINAL_TEST_ACCESS_CONSUMED",
        "SCIENTIFIC_RESET_INVALIDATED",
    ]
    require_development_change_allowed(tmp_path, "threshold")
