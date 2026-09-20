import pytest

from g3_helpers import build_complete_g3_fixture
from vedant_infra.g3 import (
    G3FreezeError,
    freeze_g3,
    validate_g3_marker,
)


@pytest.mark.parametrize(
    "target",
    ("manifest", "model", "calibrator", "threshold", "split"),
)
def test_mutation_invalidates_marker(tmp_path, target):
    paths = build_complete_g3_fixture(tmp_path)
    marker = freeze_g3(
        tmp_path,
        marker_path=tmp_path / "synthetic-marker.json",
        scope="synthetic_test",
    )
    path = (
        paths["models"]["recovery"]
        if target == "model"
        else paths[target]
    )
    path.write_bytes(path.read_bytes() + b"x")
    with pytest.raises(G3FreezeError):
        validate_g3_marker(
            marker, tmp_path, expected_scope="synthetic_test"
        )
