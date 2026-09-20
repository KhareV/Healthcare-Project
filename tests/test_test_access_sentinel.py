import pytest

from vedant_infra.g3 import G3FreezeError, guarded_test_access


def test_invalid_g3_never_opens_hostile_test_source(tmp_path):
    def hostile_loader():
        raise AssertionError("TEST_DATA_OPENED")

    with pytest.raises(G3FreezeError):
        guarded_test_access(
            tmp_path,
            hostile_loader,
            expected_scope="real",
        )
