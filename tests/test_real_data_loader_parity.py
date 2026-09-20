import json

from data.acceptance import audit_real_data
from data.real_adapter import load_real_bundle, loader_projection
from real_data_helpers import build_real_handoff_fixture, refresh_manifest_hash, write_json


CONFIG = __import__("pathlib").Path(__file__).resolve().parents[1] / "configs/data_acceptance_v1.json"


def test_receiver_loader_projection_preserves_every_model_field(tmp_path):
    manifest = build_real_handoff_fixture(tmp_path)
    bundle = load_real_bundle(tmp_path, manifest)
    for example in bundle.examples:
        loaded = loader_projection(example, bundle.dynamic_feature_names)
        assert loaded["history_values"] == example.history_values
        assert loaded["observation_mask"] == example.observation_mask
        assert loaded["targets"] == example.targets


def test_row_reordering_preserves_semantic_acceptance_hash(tmp_path):
    manifest = build_real_handoff_fixture(tmp_path)
    first = audit_real_data(tmp_path, manifest_path=manifest, config_path=CONFIG)
    for name, inventory_name in (
        ("feature_dataset_v1.json", "canonical_dataset"),
        ("canonical_structural_index_v1.json", "structural_index"),
        ("organ_support_handoff_v1.json", "support_handoff"),
    ):
        path = tmp_path / "artifacts/data" / name
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["rows"] = list(reversed(payload["rows"]))
        write_json(path, payload)
        refresh_manifest_hash(tmp_path, inventory_name)
    second = audit_real_data(tmp_path, manifest_path=manifest, config_path=CONFIG)
    assert first["overall_status"] == second["overall_status"] == "ACCEPTED"
    assert first["acceptance_decision_sha256"] == second["acceptance_decision_sha256"]
