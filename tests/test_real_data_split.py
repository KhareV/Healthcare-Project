import json

import pytest

from data.real_adapter import RealDataContractError, load_real_bundle
from real_data_helpers import build_real_handoff_fixture, refresh_manifest_hash, write_json


def test_split_annotation_mismatch_is_hard_failure(tmp_path):
    manifest = build_real_handoff_fixture(tmp_path)
    path = tmp_path / "artifacts/data/canonical_structural_index_v1.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["rows"][0]["partition"] = "validation"
    write_json(path, payload)
    refresh_manifest_hash(tmp_path, "structural_index")
    with pytest.raises(RealDataContractError, match="split"):
        load_real_bundle(tmp_path, manifest)


def test_duplicate_canonical_identity_is_hard_failure(tmp_path):
    manifest = build_real_handoff_fixture(tmp_path)
    path = tmp_path / "artifacts/data/canonical_structural_index_v1.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["rows"].append(dict(payload["rows"][0]))
    write_json(path, payload)
    refresh_manifest_hash(tmp_path, "structural_index")
    with pytest.raises(RealDataContractError, match="CANONICAL ROW IDENTITY"):
        load_real_bundle(tmp_path, manifest)
