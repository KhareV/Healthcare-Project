from dataclasses import replace

import pytest

from experiments.lineage import LineageValidationError, validate_artifact_lineage
from registry_helpers import chain, run


def test_full_chain_validates_and_unknown_producer_fails(tmp_path):
    selected = run()
    records = chain(tmp_path, selected)
    validate_artifact_lineage(records, (selected,), repository_root=tmp_path)
    with pytest.raises(LineageValidationError, match="unknown producing run"):
        validate_artifact_lineage((replace(records[0], producing_run_id="missing"),), (selected,), repository_root=tmp_path)


@pytest.mark.parametrize("field", ("task", "model_family", "split_hash", "feature_version", "label_version"))
def test_run_artifact_compatibility_fails_closed(tmp_path, field):
    selected = run()
    records = list(chain(tmp_path, selected))
    incompatible = "f" * 64 if field == "split_hash" else "incompatible"
    records[0] = replace(records[0], **{field: incompatible})
    with pytest.raises(LineageValidationError, match="mismatch"):
        validate_artifact_lineage(records, (selected,), repository_root=tmp_path)


def test_duplicate_path_unknown_parent_cycle_and_missing_file_fail(tmp_path):
    selected = run()
    records = list(chain(tmp_path, selected))
    with pytest.raises(LineageValidationError, match="duplicate immutable"):
        validate_artifact_lineage((records[0], replace(records[0], artifact_id="other")), (selected,), repository_root=tmp_path)
    with pytest.raises(LineageValidationError, match="unknown parent"):
        validate_artifact_lineage((replace(records[0], parent_artifact_ids="missing"),), (selected,), repository_root=tmp_path)
    a = replace(records[0], artifact_id="a", artifact_path="artifacts/model.bin", parent_artifact_ids="b")
    b_file = tmp_path / "artifacts/b.bin"
    b_file.write_bytes(b"b")
    from vedant_infra.hashing import sha256_file
    b = replace(records[0], artifact_id="b", artifact_path="artifacts/b.bin", artifact_sha256=sha256_file(b_file), parent_artifact_ids="a")
    with pytest.raises(LineageValidationError, match="cycle"):
        validate_artifact_lineage((a, b), (selected,), repository_root=tmp_path)
    (tmp_path / records[0].artifact_path).unlink()
    with pytest.raises(LineageValidationError, match="MISSING ARTIFACT"):
        validate_artifact_lineage((records[0],), (selected,), repository_root=tmp_path)
