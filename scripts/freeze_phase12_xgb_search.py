#!/usr/bin/env python3
"""Materialize and freeze all Phase-12 scientific candidates before training."""
from __future__ import annotations

import importlib.metadata
import json
import platform
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.synthetic.config import canonical_json_bytes
from data.synthetic.provenance import sha256_file
from data.xgb_canonical import Phase10XGBData
from experiments.xgb_phase12 import (
    MASTER_SEARCH_SEED,
    MAX_ATTEMPTS,
    SAMPLER_VERSION,
    SEARCH_VERSION,
    TASKS,
    candidate_manifest,
    canonical_sha256,
    derived_seed,
    immutable_contract,
    validate_candidate_manifest,
    validate_search_space,
)


SEARCH_ROOT = ROOT / "artifacts/search/xgb/phase12"
MANIFEST_ROOT = SEARCH_ROOT / "manifests"


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write_new(path: Path, value) -> None:
    if path.exists():
        raise FileExistsError("refusing to overwrite frozen Phase-12 artifact: " + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value))


def _git(*args: str) -> str:
    return subprocess.check_output(("git",) + args, cwd=ROOT, text=True).strip()


def _subject_hash(partition) -> str:
    subjects = sorted({str(key[0]) for key in partition.row_keys})
    return canonical_sha256(subjects)


def _environment(implementation_commit: str):
    packages = {}
    for name in ("numpy", "scikit-learn", "torch", "xgboost"):
        packages[name] = importlib.metadata.version(name)
    try:
        libomp = subprocess.check_output(
            ("brew", "list", "--versions", "libomp"), text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        libomp = "UNAVAILABLE"
    return {
        "environment_version": "phase12_xgb_scientific_environment_v1",
        "status": "FROZEN_FOR_PHASE12_XGB_SCIENTIFIC_SEARCH",
        "decision_authority": "USER_DELEGATED_AI_PROJECT_DECISION",
        "scope": "PHASE12_XGBOOST_VALIDATION_SEARCH_ONLY_NOT_FINAL_SERVING_LOCK",
        "implementation_commit": implementation_commit,
        "python": {"implementation": platform.python_implementation(), "version": platform.python_version()},
        "platform": {"system": platform.system(), "release": platform.release(), "machine": platform.machine()},
        "packages": packages,
        "native_runtime": {"libomp": libomp},
        "execution": {"tree_method": "hist", "device": "cpu", "n_jobs": 1},
    }


def main() -> None:
    if _git("status", "--porcelain"):
        raise RuntimeError("freeze requires a clean committed implementation worktree")
    implementation_commit = _git("rev-parse", "HEAD")
    space_path = ROOT / "configs/search_space_xgb_phase12_v1.json"
    space = _load(space_path)
    validate_search_space(space)
    phase11_path = ROOT / "artifacts/manifests/synthetic_phase11_xgb_input_manifest_v1.json"
    phase11 = _load(phase11_path)
    if phase11.get("status") != "PHASE11_XGB_CANONICAL_INPUT_READY" or phase11.get("parity_result") != "PASS_COMPLETE_TRAIN_AND_VALIDATION":
        raise RuntimeError("BLOCKED — COMPLETE PHASE-11 XGBOOST SEARCH READINESS REQUIRED")
    gateway = Phase10XGBData(ROOT, ROOT / phase11["flat_feature_map_path"])
    train = gateway.load_partition("train")
    validation = gateway.load_partition("validation")

    environment_path = SEARCH_ROOT / "phase12_xgb_scientific_environment_v1.json"
    _write_new(environment_path, _environment(implementation_commit))
    manifest_entries = {}
    for task in TASKS:
        payload = candidate_manifest(space, task, implementation_commit=implementation_commit)
        validate_candidate_manifest(payload, space)
        path = MANIFEST_ROOT / f"{task}_candidates_v1.json"
        _write_new(path, payload)
        manifest_entries[task] = {
            "path": str(path.relative_to(ROOT)),
            "sha256": sha256_file(path),
            "candidate_list_hash": payload["candidate_list_hash"],
            "candidate_count": 30,
        }

    master = {
        "manifest_version": "synthetic_xgb_phase12_search_manifest_v1",
        "search_version": SEARCH_VERSION,
        "status": "FROZEN_READY_TO_RUN",
        "decision_authority": "USER_DELEGATED_AI_PROJECT_DECISION",
        "scope": "SYNTHETIC_ADULT_CARDIAC_RETROSPECTIVE_FORECASTING",
        "family": "xgboost",
        "tasks": list(TASKS),
        "master_search_seed": MASTER_SEARCH_SEED,
        "sampler": SAMPLER_VERSION,
        "task_sampler_seeds": {task: derived_seed(task) for task in TASKS},
        "search_space_path": str(space_path.relative_to(ROOT)),
        "search_space_sha256": canonical_sha256(space),
        "search_space_file_sha256": sha256_file(space_path),
        "candidate_manifests": manifest_entries,
        "phase11_manifest_path": str(phase11_path.relative_to(ROOT)),
        "phase11_manifest_sha256": sha256_file(phase11_path),
        "split_sha256": phase11["split_sha256"],
        "train_subject_set_sha256": _subject_hash(train),
        "validation_subject_set_sha256": _subject_hash(validation),
        "validation_row_keys_sha256": phase11["partitions"]["validation"]["row_keys_sha256"],
        "feature_schema_sha256": phase11["feature_schema_sha256"],
        "flat_feature_map_sha256": phase11["flat_feature_map_sha256"],
        "preprocessor_sha256": phase11["preprocessor_sha256"],
        "target_contract_sha256": phase11["target_contract_sha256"],
        "recovery_target_scaler_sha256": phase11["recovery_target_scaler_sha256"],
        "support_class_weight_sha256": phase11["support_class_weight_sha256"],
        "environment_path": str(environment_path.relative_to(ROOT)),
        "environment_sha256": sha256_file(environment_path),
        "validation_objectives": {
            "recovery": "MIN_STAY_BALANCED_MAE24_ORIGINAL_DELTA_SOFA_UNITS",
            "icu_time": "MIN_STAY_BALANCED_WEIGHTED_MEDIAN_ABSOLUTE_ERROR_HOURS",
            "organ_support": "MAX_STAY_BALANCED_AUPRC_RAW_UNCALIBRATED",
        },
        "tie_break_policies": {
            "recovery": ["mae24_asc", "mae48_asc", "rmse24_asc", "candidate_id_lexical"],
            "icu_time": ["median_absolute_error_hours_asc", "mae_hours_asc", "rmse_hours_asc", "candidate_id_lexical"],
            "organ_support": ["auprc_desc", "brier_asc", "auroc_desc", "candidate_id_lexical"],
        },
        "early_stopping": {
            "rounds": 50,
            "evaluation_partition": "validation",
            "internal_monitor_weighting": "ordinary_row_weighted",
            "internal_monitors": space["internal_early_stopping_monitors"],
            "authoritative_ranking": "external_existing_stay_balanced_evaluators",
        },
        "max_attempts": MAX_ATTEMPTS,
        "test_access_policy": "DEVELOPMENT_TEST_ACCESS_FORBIDDEN",
        "implementation_commit": implementation_commit,
        "selected_models_v1_presearch": {
            "path": "artifacts/models/selected_models_v1.json",
            "present": (ROOT / "artifacts/models/selected_models_v1.json").is_file(),
            "sha256": sha256_file(ROOT / "artifacts/models/selected_models_v1.json") if (ROOT / "artifacts/models/selected_models_v1.json").is_file() else None,
        },
        "test_gate_presearch": {
            "phase10_test_opened": phase11["test_partition_requested"],
            "phase10_test_matrix_constructed": phase11["test_matrix_constructed"],
            "g3_freeze_present": (ROOT / "artifacts/governance/g3_freeze.json").is_file(),
        },
        "execution_commit": None,
        "started_at_utc": None,
        "completed_at_utc": None,
        "terminal_candidate_counts": None,
        "test_accessed": False,
        "calibration_performed": False,
        "threshold_selected": False,
        "family_selection_performed": False,
    }
    master["immutable_contract_sha256"] = canonical_sha256(immutable_contract(master))
    master_path = MANIFEST_ROOT / "phase12_search_manifest_v1.json"
    _write_new(master_path, master)
    print(json.dumps({
        "status": master["status"],
        "master_manifest": str(master_path.relative_to(ROOT)),
        "search_space_sha256": master["search_space_sha256"],
        "candidate_list_hashes": {task: item["candidate_list_hash"] for task, item in manifest_entries.items()},
        "task_sampler_seeds": master["task_sampler_seeds"],
        "scientific_candidates": 90,
        "test_accessed": False,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
