import pytest

from registry_helpers import run
from vedant_infra.registry import RegistryValidationError, upsert_registry_record, validate_registry


def test_finalized_run_is_idempotent_but_immutable(tmp_path):
    path = tmp_path / "registry.csv"
    original = run()
    upsert_registry_record(path, original)
    upsert_registry_record(path, original)
    assert validate_registry(path) == 1
    changed = dict(original, split_hash="d" * 64)
    with pytest.raises(RegistryValidationError, match="immutable"):
        upsert_registry_record(path, changed)


def test_restricted_metadata_is_rejected(tmp_path):
    with pytest.raises(RegistryValidationError, match="prohibited"):
        upsert_registry_record(tmp_path / "registry.csv", run(notes="api_token=fixture-secret"))
