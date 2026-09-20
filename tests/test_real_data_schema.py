import json

import pytest

from data.real_adapter import RealDataContractError, load_real_bundle
from real_data_helpers import build_real_handoff_fixture, refresh_manifest_hash, write_json


def _payload(root, name):
    path = root / "artifacts/data" / name
    return path, json.loads(path.read_text(encoding="utf-8"))


def test_schema_version_mismatch_is_blocked(tmp_path):
    manifest = build_real_handoff_fixture(tmp_path)
    path, payload = _payload(tmp_path, "processed_schema_v1.json")
    payload["schema_version"] = "incompatible"
    write_json(path, payload)
    refresh_manifest_hash(tmp_path, "schema_sidecar")
    with pytest.raises(RealDataContractError, match="version mismatch"):
        load_real_bundle(tmp_path, manifest)


def test_feature_order_attack_fails_even_when_shape_is_unchanged(tmp_path):
    manifest = build_real_handoff_fixture(tmp_path)
    path, payload = _payload(tmp_path, "feature_dataset_v1.json")
    payload["dynamic_feature_names"] = list(reversed(payload["dynamic_feature_names"]))
    write_json(path, payload)
    refresh_manifest_hash(tmp_path, "canonical_dataset")
    with pytest.raises(RealDataContractError, match="feature order"):
        load_real_bundle(tmp_path, manifest)


def test_frozen_range_violation_is_not_silently_repaired(tmp_path):
    manifest = build_real_handoff_fixture(tmp_path)
    path, payload = _payload(tmp_path, "feature_dataset_v1.json")
    row = payload["rows"][0]
    first_valid = row["padding"].index(False)
    row["sequence"][first_valid][0] = 1000.0
    write_json(path, payload)
    refresh_manifest_hash(tmp_path, "canonical_dataset")
    with pytest.raises(RealDataContractError, match="frozen feature range"):
        load_real_bundle(tmp_path, manifest)


def test_static_feature_order_attack_fails(tmp_path):
    manifest = build_real_handoff_fixture(tmp_path)
    path, payload = _payload(tmp_path, "feature_dataset_v1.json")
    payload["static_feature_names"] = ["wrong_static"]
    write_json(path, payload)
    refresh_manifest_hash(tmp_path, "canonical_dataset")
    with pytest.raises(RealDataContractError, match="static feature order"):
        load_real_bundle(tmp_path, manifest)


def test_validation_statistics_in_preprocessing_provenance_fail(tmp_path):
    manifest = build_real_handoff_fixture(tmp_path)
    path, payload = _payload(tmp_path, "processed_schema_v1.json")
    payload["preprocessing"]["validation_or_test_statistics_used"] = True
    write_json(path, payload)
    refresh_manifest_hash(tmp_path, "schema_sidecar")
    with pytest.raises(RealDataContractError, match="train-only"):
        load_real_bundle(tmp_path, manifest)
