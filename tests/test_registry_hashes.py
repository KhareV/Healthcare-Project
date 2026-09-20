from experiments.search_governance import canonical_sha256
from experiments.lineage import LineageValidationError, validate_artifact_lineage
from registry_helpers import chain, run
import pytest


def test_config_hash_is_order_independent_and_parameter_sensitive():
    assert canonical_sha256({"a": 1, "b": 2}) == canonical_sha256({"b": 2, "a": 1})
    assert canonical_sha256({"a": 1}) != canonical_sha256({"a": 2})


def test_hash_on_read_detects_one_byte_drift_without_updating(tmp_path):
    selected = run()
    records = chain(tmp_path, selected)
    validate_artifact_lineage(records, (selected,), repository_root=tmp_path)
    old_hash = records[0].artifact_sha256
    (tmp_path / records[0].artifact_path).write_bytes(b"model!")
    with pytest.raises(LineageValidationError, match="SHA-256 mismatch"):
        validate_artifact_lineage(records, (selected,), repository_root=tmp_path)
    assert records[0].artifact_sha256 == old_hash
