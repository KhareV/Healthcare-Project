#!/usr/bin/env python3
"""Fail-closed Phase-11 artifact and governance audit."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.synthetic.provenance import sha256_file
from data.xgb_canonical import Phase10XGBData, XGBCanonicalContractError


def main() -> None:
    manifest_path = ROOT / "artifacts/manifests/synthetic_phase11_xgb_input_manifest_v1.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["status"] != "PHASE11_XGB_CANONICAL_INPUT_READY" or manifest["parity_result"] != "PASS_COMPLETE_TRAIN_AND_VALIDATION":
        raise RuntimeError("Phase-11 readiness/parity status failed")
    for path_field, hash_field in (
        ("phase10_manifest_path", "phase10_manifest_sha256"), ("split_path", "split_sha256"),
        ("feature_schema_path", "feature_schema_sha256"), ("preprocessor_path", "preprocessor_sha256"),
        ("model_input_contract_path", "model_input_contract_sha256"),
        ("information_parity_contract_path", "information_parity_contract_sha256"),
        ("flat_feature_map_path", "flat_feature_map_sha256"), ("target_contract_path", "target_contract_sha256"),
        ("recovery_target_scaler_path", "recovery_target_scaler_sha256"),
        ("support_class_weight_path", "support_class_weight_sha256"),
        ("event_dictionary_path", "event_dictionary_sha256"), ("search_space_path", "search_space_sha256"),
        ("data_entrypoint_path", "data_entrypoint_sha256"),
        ("model_entrypoint_path", "model_entrypoint_sha256"),
    ):
        if sha256_file(ROOT / manifest[path_field]) != manifest[hash_field]:
            raise RuntimeError("Phase-11 lineage hash mismatch: " + path_field)
    if (manifest["flat_dimension"], manifest["temporal_feature_count"], manifest["encoded_static_count"]) != (520, 21, 8):
        raise RuntimeError("Phase-11 dimensions changed")
    if manifest["scientific_xgb_candidates_executed"] or manifest["search_executed"] or manifest["model_selection_performed"] or manifest["calibration_performed"] or manifest["threshold_selected"] or manifest["test_matrix_constructed"]:
        raise RuntimeError("Phase 12+ activity leaked into Phase 11")
    gateway = Phase10XGBData(ROOT, ROOT / manifest["flat_feature_map_path"])
    try:
        gateway.load_partition("test")
    except XGBCanonicalContractError:
        pass
    else:
        raise RuntimeError("test partition guard failed")
    print(json.dumps({"status": "PASS", "F": 21, "S": 8, "D": 520, "train_rows": manifest["partitions"]["train"]["row_count"], "validation_rows": manifest["partitions"]["validation"]["row_count"], "scientific_candidates": 0, "test_matrix": False}, sort_keys=True))


if __name__ == "__main__":
    main()
