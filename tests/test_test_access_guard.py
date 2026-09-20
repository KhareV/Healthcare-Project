import pytest

from g3_helpers import build_complete_g3_fixture
from vedant_infra.g3 import (
    G3FreezeError,
    freeze_g3,
    guarded_test_access,
    load_access_state,
)


def test_guard_runs_before_loader_and_allows_exactly_one_run(tmp_path):
    build_complete_g3_fixture(tmp_path)
    marker = freeze_g3(
        tmp_path,
        marker_path=tmp_path / "synthetic-marker.json",
        scope="synthetic_test",
    )
    events = []

    def loader():
        events.append("loader")
        return "loaded"

    assert guarded_test_access(
        tmp_path,
        loader,
        marker_path=marker,
        expected_scope="synthetic_test",
    ) == "loaded"
    assert events == ["loader"]
    assert load_access_state(
        tmp_path / "artifacts/governance/test_access_state.json"
    )["state"] == "FINAL_RUN_COMPLETED"
    with pytest.raises(G3FreezeError):
        guarded_test_access(
            tmp_path,
            loader,
            marker_path=marker,
            expected_scope="synthetic_test",
        )
    assert events == ["loader"]


def test_missing_or_malformed_marker_never_calls_loader(tmp_path):
    called = []
    with pytest.raises(G3FreezeError):
        guarded_test_access(
            tmp_path,
            lambda: called.append(True),
            marker_path=tmp_path / "missing.json",
            expected_scope="synthetic_test",
        )
    malformed = tmp_path / "bad.json"
    malformed.write_text('{"freeze_status":"ACTIVE"}', encoding="utf-8")
    with pytest.raises(G3FreezeError):
        guarded_test_access(
            tmp_path,
            lambda: called.append(True),
            marker_path=malformed,
            expected_scope="synthetic_test",
        )
    assert called == []
