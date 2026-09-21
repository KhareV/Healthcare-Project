#!/usr/bin/env python3
"""Freeze all 90 Stage-1 GRU candidates before scientific training."""
from __future__ import annotations

import importlib.metadata
import json
import platform
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.gru_canonical import Phase10GRUDataset
from data.synthetic.config import canonical_json_bytes
from data.synthetic.provenance import sha256_file
from experiments.gru_final import (
    BUDGET, MASTER_SEED, SAMPLER_VERSION, SEARCH_VERSION, TASKS,
    candidate_manifest, canonical_sha256, derived_seed, validate_search_space,
)

SEARCH_ROOT = ROOT / "artifacts/search/gru/final_v2"
MANIFEST_ROOT = SEARCH_ROOT / "manifests"


def _load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write_new(path, payload):
    if path.exists():
        raise RuntimeError("refusing to overwrite frozen final-GRU artifact: " + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(payload))


def _git(*args):
    return subprocess.check_output(("git",) + args, cwd=ROOT, text=True).strip()


def main():
    if _git("status", "--porcelain"):
        raise RuntimeError("GRU candidate freeze requires a clean committed worktree")
    commit = _git("rev-parse", "HEAD")
    g1_path = ROOT / "artifacts/acceptance/g1_synthetic_data_freeze_v1.json"
    handoff_path = ROOT / "artifacts/handoffs/sanskruti_phase14_handoff_v1.json"
    g1, handoff = _load(g1_path), _load(handoff_path)
    if (g1.get("status") != "ACCEPTED_SYNTHETIC_DATA_LABEL_FREEZE"
            or g1.get("test_accessed") is not False
            or handoff.get("status") != "SANSKRUTI_TRACK_HANDOFF_FROZEN"
            or handoff.get("test_accessed") is not False
            or handoff.get("g3_created") is not False):
        raise RuntimeError("accepted G1/Phase-14 scientific lineage is required")
    space_path = ROOT / "configs/search_space_gru_final_v1.json"
    space = _load(space_path); validate_search_space(space)
    train, validation = Phase10GRUDataset(ROOT, "train"), Phase10GRUDataset(ROOT, "validation")
    environment = {
        "environment_version": "vedant_final_gru_environment_v2",
        "status": "FROZEN_FOR_FINAL_GRU_VALIDATION_SEARCH", "implementation_commit": commit,
        "python": platform.python_version(), "torch": importlib.metadata.version("torch"),
        "numpy": importlib.metadata.version("numpy"), "platform": platform.platform(),
        "device_policy": "cpu_deterministic", "test_accessed": False,
    }
    environment_path = SEARCH_ROOT / "environment_v1.json"; _write_new(environment_path, environment)
    entries = {}
    for task in TASKS:
        payload = candidate_manifest(space, task, commit)
        path = MANIFEST_ROOT / f"{task}_candidates_v1.json"; _write_new(path, payload)
        entries[task] = {"path": str(path.relative_to(ROOT)), "sha256": sha256_file(path),
                         "candidate_count": BUDGET, "candidate_list_hash": payload["candidate_list_hash"]}
    parents = handoff["parents"]
    master = {
        "manifest_version": "vedant_final_gru_search_manifest_v1", "search_version": SEARCH_VERSION,
        "status": "FROZEN_READY_TO_RUN", "decision_authority": "USER_DELEGATED_AI_PROJECT_DECISION",
        "family": "gru", "tasks": list(TASKS), "scientific_candidate_count": 90,
        "master_seed": MASTER_SEED, "sampler": SAMPLER_VERSION,
        "task_sampler_seeds": {task: derived_seed(task, "sampler") for task in TASKS},
        "search_space_path": str(space_path.relative_to(ROOT)), "search_space_sha256": canonical_sha256(space),
        "search_space_file_sha256": sha256_file(space_path), "candidate_manifests": entries,
        "g1_path": str(g1_path.relative_to(ROOT)), "g1_sha256": sha256_file(g1_path),
        "phase14_handoff_path": str(handoff_path.relative_to(ROOT)), "phase14_handoff_sha256": sha256_file(handoff_path),
        "feature_schema_sha256": parents["feature_schema"]["sha256"],
        "split_sha256": parents["split"]["sha256"], "preprocessor_sha256": parents["preprocessor"]["sha256"],
        "recovery_target_scaler_sha256": parents["recovery_target_scaler"]["sha256"],
        "support_class_weight_sha256": parents["support_class_weight"]["sha256"],
        "feature_schema_version": train.feature_schema_version,
        "tensor_contract_version": train.tensor_contract_version,
        "dimensions": {"T": 8, "F": len(train.feature_names), "S_model": len(train.static_feature_names)},
        "row_counts": {"train": len(train), "validation": len(validation)},
        "validation_row_keys_sha256": canonical_sha256([
            [row["subject_id"], row["stay_id"], row["prediction_time"], row["grid_index"]]
            for row in validation.rows
        ]),
        "validation_objectives": {
            "recovery": "MIN_STAY_BALANCED_MAE24_ORIGINAL_DELTA_SOFA_UNITS",
            "icu_time": "MIN_STAY_BALANCED_WEIGHTED_MEDIAN_ABSOLUTE_ERROR_HOURS",
            "organ_support": "MAX_STAY_BALANCED_AUPRC_RAW_UNCALIBRATED"},
        "environment_path": str(environment_path.relative_to(ROOT)), "environment_sha256": sha256_file(environment_path),
        "implementation_commit": commit, "execution_commit": None, "max_attempts": 3,
        "test_accessed": False, "g3_created": False, "family_selection_performed": False,
        "support_calibrated": False, "support_threshold": None,
    }
    master["immutable_contract_sha256"] = canonical_sha256({key: value for key, value in master.items() if key not in ("status", "execution_commit")})
    path = MANIFEST_ROOT / "search_manifest_v1.json"; _write_new(path, master)
    print(json.dumps({"status": master["status"], "commit": commit, "candidates": 90,
                      "candidate_list_hashes": {k: v["candidate_list_hash"] for k, v in entries.items()},
                      "test_accessed": False}, sort_keys=True))


if __name__ == "__main__":
    main()
