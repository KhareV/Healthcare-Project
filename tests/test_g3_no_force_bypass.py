import inspect

import pytest

from g3_helpers import build_complete_g3_fixture, rewrite_manifest_hash
from vedant_infra.g3 import (
    G3FreezeError,
    freeze_g3,
    guarded_test_access,
)


def test_freeze_has_no_force_or_ignore_parameter():
    parameters = inspect.signature(freeze_g3).parameters
    assert not {"force", "unsafe", "ignore_errors", "skip_validation"} & set(
        parameters
    )


def test_test_performance_fields_are_rejected(tmp_path):
    paths = build_complete_g3_fixture(tmp_path)
    rewrite_manifest_hash(
        paths["manifest"],
        lambda payload: payload.update({"test_mae": 0.1}),
    )
    with pytest.raises(G3FreezeError, match="test performance"):
        freeze_g3(
            tmp_path,
            marker_path=tmp_path / "synthetic-marker.json",
            scope="synthetic_test",
        )


def test_synthetic_marker_cannot_authorize_real_path(tmp_path):
    build_complete_g3_fixture(tmp_path)
    marker = freeze_g3(
        tmp_path,
        marker_path=tmp_path / "synthetic-marker.json",
        scope="synthetic_test",
    )
    called = []
    with pytest.raises(G3FreezeError, match="scope"):
        guarded_test_access(
            tmp_path,
            lambda: called.append(True),
            marker_path=marker,
            expected_scope="real",
        )
    assert called == []
