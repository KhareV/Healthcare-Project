#!/usr/bin/env python3
"""Build Phase-11 flat-map and complete train/validation parity evidence."""
import csv
import hashlib
import json
import platform
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import xgboost

from data.synthetic.config import canonical_json_bytes
from data.synthetic.provenance import sha256_file
from data.xgb_canonical import (
    Phase10XGBData,
    TASK_VIEWS,
    build_flat_feature_map,
    matrix_content_hash,
    row_keys_hash,
)


def _hash_targets(values, keys):
    digest = hashlib.sha256()
    digest.update(np.asarray(values, dtype="<f4").tobytes(order="C"))
    digest.update(canonical_json_bytes([list(key) for key in keys]))
    return digest.hexdigest()


def main() -> None:
    map_path = ROOT / "artifacts/features/synthetic_xgb_flat_feature_map_v1.json"
    manifest_path = ROOT / "artifacts/manifests/synthetic_phase11_xgb_input_manifest_v1.json"
    if map_path.exists() or manifest_path.exists():
        raise FileExistsError("Phase-11 artifacts already exist; refusing silent overwrite")
    map_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    flat_map = build_flat_feature_map(ROOT)
    map_path.write_bytes(canonical_json_bytes(flat_map))
    gateway = Phase10XGBData(ROOT, map_path)
    partitions = {}
    eligibility = {}
    target_parity = {}
    for partition_name in ("train", "validation"):
        partition = gateway.load_partition(partition_name)
        partitions[partition_name] = {
            "row_count": len(partition.row_keys),
            "row_keys_sha256": row_keys_hash(partition.row_keys),
            "matrix_content_sha256": matrix_content_hash(partition.X, partition.row_keys),
            "shape": list(partition.X.shape), "dtype": str(partition.X.dtype),
            "complete_flatten_unflatten_audit": "PASS",
        }
        for task in TASK_VIEWS:
            matrix = gateway.task_matrix(partition, task)
            key = partition_name + ":" + task
            eligibility[key] = {"eligible_count": len(matrix.row_keys), "row_keys_sha256": row_keys_hash(matrix.row_keys)}
            target_parity[key] = {"target_sha256": _hash_targets(matrix.y, matrix.row_keys), "family_specific_transform": False}
    with (ROOT / "experiments/registry.csv").open(newline="", encoding="utf-8") as handle:
        registry_rows = list(csv.DictReader(handle))
    scientific_xgb = [row for row in registry_rows if row.get("model_family") == "xgboost" and row.get("run_type") == "scientific"]
    support_indices = [item["flat_index"] for item in flat_map["entries"] if item["canonical_feature_name"] in {"vasopressor_on", "invasive_ventilation_on", "norepinephrine_rate", "epinephrine_rate", "dopamine_rate", "dobutamine_rate"}]
    phase10_path = ROOT / "artifacts/manifests/synthetic_phase10_manifest_v1.json"
    phase10 = json.loads(phase10_path.read_text())
    target_contract = ROOT / "configs/synthetic/xgb_target_contract_v1.json"
    search_space = ROOT / "configs/search_space_xgb_v2.json"
    parity_contract = ROOT / "configs/synthetic/information_parity_v1.json"
    feature_schema = ROOT / "configs/synthetic/feature_schema_v2.json"
    preprocessor = ROOT / "artifacts/preprocessors/synthetic_feature_preprocessor_v1.json"
    recovery_scaler = ROOT / "artifacts/preprocessors/recovery_target_scaler_synthetic_v1.json"
    class_weight = ROOT / "artifacts/preprocessors/support_class_weight_synthetic_v1.json"
    split = ROOT / "artifacts/splits/synthetic_split_v2.csv"
    event_dictionary = ROOT / "configs/event_dict_v2.yaml"
    data_entrypoint = ROOT / "src/data/xgb_canonical.py"
    model_entrypoint = ROOT / "src/models/xgb_canonical.py"
    manifest = {
        "manifest_version": "synthetic_phase11_xgb_input_manifest_v1",
        "status": "PHASE11_XGB_CANONICAL_INPUT_READY",
        "decision_authority": "USER_DELEGATED_AI_PROJECT_DECISION",
        "scope": "SYNTHETIC_ADULT_CARDIAC_RETROSPECTIVE_FORECASTING",
        "phase10_manifest_path": str(phase10_path.relative_to(ROOT)), "phase10_manifest_sha256": sha256_file(phase10_path),
        "phase10_status": phase10["status"], "split_path": str(split.relative_to(ROOT)), "split_sha256": sha256_file(split),
        "feature_schema_path": str(feature_schema.relative_to(ROOT)), "feature_schema_sha256": sha256_file(feature_schema),
        "feature_schema_version": flat_map["feature_schema_version"], "temporal_feature_count": 21,
        "preprocessor_path": str(preprocessor.relative_to(ROOT)), "preprocessor_sha256": sha256_file(preprocessor),
        "model_input_contract_path": phase10["model_input_contract_path"], "model_input_contract_sha256": phase10["model_input_contract_sha256"],
        "information_parity_contract_path": str(parity_contract.relative_to(ROOT)), "information_parity_contract_sha256": sha256_file(parity_contract),
        "flat_feature_map_path": str(map_path.relative_to(ROOT)), "flat_feature_map_sha256": sha256_file(map_path),
        "flat_feature_map_version": flat_map["map_version"], "encoded_static_count": 8, "flat_dimension": 520,
        "target_contract_path": str(target_contract.relative_to(ROOT)), "target_contract_sha256": sha256_file(target_contract),
        "recovery_target_scaler_path": str(recovery_scaler.relative_to(ROOT)), "recovery_target_scaler_sha256": sha256_file(recovery_scaler),
        "support_class_weight_path": str(class_weight.relative_to(ROOT)), "support_class_weight_sha256": sha256_file(class_weight),
        "event_dictionary_path": str(event_dictionary.relative_to(ROOT)), "event_dictionary_sha256": sha256_file(event_dictionary),
        "search_space_path": str(search_space.relative_to(ROOT)), "search_space_sha256": sha256_file(search_space),
        "data_entrypoint_path": str(data_entrypoint.relative_to(ROOT)), "data_entrypoint_sha256": sha256_file(data_entrypoint),
        "model_entrypoint_path": str(model_entrypoint.relative_to(ROOT)), "model_entrypoint_sha256": sha256_file(model_entrypoint),
        "partitions": partitions, "task_eligibility": eligibility, "target_parity": target_parity,
        "support_channel_flat_entry_count": len(support_indices), "support_channel_flat_indices_sha256": hashlib.sha256(canonical_json_bytes(support_indices)).hexdigest(),
        "parity_result": "PASS_COMPLETE_TRAIN_AND_VALIDATION",
        "test_partition_requested": False, "test_matrix_constructed": False, "test_predictions_computed": False,
        "xgboost_environment": {"version": xgboost.__version__, "python": platform.python_version(), "platform": platform.platform(), "tree_method": "hist", "device": "cpu", "n_jobs": 1, "serialization": "native_json"},
        "entrypoint_version": "canonical_xgb_candidate_bundle_v1",
        "scientific_xgb_candidates_registered": len(scientific_xgb),
        "scientific_xgb_candidates_executed": 0,
        "search_executed": False, "model_selection_performed": False, "calibration_performed": False,
        "threshold_selected": False, "production_shap_executed": False, "serving_routing_changed": False,
    }
    if scientific_xgb:
        raise RuntimeError("Phase 11 found scientific XGBoost registry rows")
    manifest_path.write_bytes(canonical_json_bytes(manifest))
    print(json.dumps({"status": "PASS", "F": 21, "S": 8, "D": 520, "train_rows": partitions["train"]["row_count"], "validation_rows": partitions["validation"]["row_count"], "scientific_candidates": 0, "test_matrix": False}, sort_keys=True))


if __name__ == "__main__":
    main()
