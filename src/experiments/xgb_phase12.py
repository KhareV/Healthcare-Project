"""Governed deterministic Phase-12 XGBoost scientific-search contracts."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from data.synthetic.config import canonical_json_bytes
from data.synthetic.provenance import sha256_file
from experiments.search_governance import GovernanceError, SEARCH_BUDGET


MASTER_SEARCH_SEED = 20260921
SEARCH_VERSION = "synthetic_xgb_phase12_validation_search_v1"
SAMPLER_VERSION = "numpy_pcg64_latin_hypercube_v1"
TASKS = ("recovery", "icu_time", "organ_support")
TASK_SLUGS = {
    "recovery": "recovery",
    "icu_time": "icu-time",
    "organ_support": "support",
}
MAX_ATTEMPTS = 3
IMMUTABLE_MASTER_FIELDS = (
    "search_version", "family", "tasks", "master_search_seed",
    "sampler", "task_sampler_seeds", "search_space_sha256",
    "candidate_manifests", "split_sha256", "feature_schema_sha256",
    "flat_feature_map_sha256", "preprocessor_sha256",
    "target_contract_sha256", "recovery_target_scaler_sha256",
    "support_class_weight_sha256", "validation_objectives",
    "tie_break_policies", "early_stopping", "max_attempts",
    "test_access_policy", "implementation_commit", "environment_sha256",
    "train_subject_set_sha256", "validation_subject_set_sha256",
    "validation_row_keys_sha256",
    "selected_models_v1_presearch", "test_gate_presearch",
)


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def derived_seed(task: str, purpose: str = "sampler", candidate_id: str = "") -> int:
    if task not in TASKS:
        raise GovernanceError("unsupported Phase-12 task")
    if purpose == "sampler":
        message = f"{MASTER_SEARCH_SEED}|phase12|xgb|{task}"
    elif purpose == "model" and candidate_id:
        message = f"{MASTER_SEARCH_SEED}|phase12|xgb|{task}|{candidate_id}|model"
    else:
        raise GovernanceError("unsupported deterministic seed purpose")
    return int.from_bytes(hashlib.sha256(message.encode("utf-8")).digest()[:4], "big")


def _load(path: Path) -> Mapping[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_search_space(space: Mapping[str, object]) -> None:
    expected = {
        "max_depth": ("integer", 3, 8, "uniform"),
        "learning_rate": ("continuous", 0.01, 0.20, "log_uniform"),
        "subsample": ("continuous", 0.60, 1.00, "uniform"),
        "colsample_bytree": ("continuous", 0.60, 1.00, "uniform"),
        "min_child_weight": ("integer", 1, 10, "uniform"),
        "reg_alpha": ("continuous", 0.0, 1.0, "uniform"),
        "reg_lambda": ("continuous", 0.1, 10.0, "log_uniform"),
    }
    if (
        space.get("search_space_version") != "xgb_phase12_search_space_v1"
        or space.get("family") != "xgboost"
        or space.get("budget_per_task") != SEARCH_BUDGET
    ):
        raise GovernanceError("Phase-12 XGBoost search identity/budget mismatch")
    parameters = space.get("parameters")
    if not isinstance(parameters, Mapping) or set(parameters) != set(expected):
        raise GovernanceError("Phase-12 searched dimensions changed")
    for name, (kind, low, high, distribution) in expected.items():
        item = parameters[name]
        if (
            item.get("type") != kind
            or item.get("minimum") != low
            or item.get("maximum") != high
            or item.get("distribution") != distribution
        ):
            raise GovernanceError("Phase-12 search bound/transform changed: " + name)
    fixed = space.get("fixed_execution", {})
    required_fixed = {
        "n_estimators": 1000,
        "tree_method": "hist",
        "device": "cpu",
        "n_jobs": 1,
        "early_stopping_rounds": 50,
        "serialization": "native_json",
        "numeric_dtype": "float32",
    }
    if any(fixed.get(key) != value for key, value in required_fixed.items()):
        raise GovernanceError("Phase-12 fixed XGBoost execution policy changed")


def _transform(spec: Mapping[str, object], u: float):
    low, high = spec["minimum"], spec["maximum"]
    if spec["type"] == "integer":
        return min(int(low) + math.floor(u * (int(high) - int(low) + 1)), int(high))
    if spec["distribution"] == "log_uniform":
        return math.exp(math.log(float(low)) + u * (math.log(float(high)) - math.log(float(low))))
    return float(low) + u * (float(high) - float(low))


def materialize_candidates(space: Mapping[str, object], task: str) -> tuple[Mapping[str, object], ...]:
    validate_search_space(space)
    if task not in TASKS:
        raise GovernanceError("unsupported Phase-12 task")
    names = tuple(space["parameters"])
    rng = np.random.Generator(np.random.PCG64(derived_seed(task)))
    coordinates = np.empty((SEARCH_BUDGET, len(names)), dtype=np.float64)
    for column in range(len(names)):
        permutation = rng.permutation(SEARCH_BUDGET)
        jitter = rng.random(SEARCH_BUDGET)
        coordinates[:, column] = (permutation + jitter) / SEARCH_BUDGET
    candidates = []
    for index in range(SEARCH_BUDGET):
        config = {
            name: _transform(space["parameters"][name], float(coordinates[index, column]))
            for column, name in enumerate(names)
        }
        config["n_estimators"] = 1000
        candidate_id = f"xgb-{TASK_SLUGS[task]}-{index + 1:03d}"
        candidates.append({
            "candidate_id": candidate_id,
            "config": config,
            "config_hash": canonical_sha256(config),
            "model_seed": derived_seed(task, "model", candidate_id),
        })
    validate_candidate_list(candidates, task, space)
    return tuple(candidates)


def validate_candidate_list(
    candidates: Sequence[Mapping[str, object]],
    task: str,
    space: Mapping[str, object],
) -> None:
    validate_search_space(space)
    if len(candidates) != SEARCH_BUDGET:
        raise GovernanceError("candidate list must contain exactly 30 configurations")
    expected_ids = [f"xgb-{TASK_SLUGS[task]}-{index:03d}" for index in range(1, 31)]
    if [item.get("candidate_id") for item in candidates] != expected_ids:
        raise GovernanceError("candidate IDs must be exact, ordered, and capped at 30")
    hashes = []
    for candidate in candidates:
        config = candidate.get("config")
        if not isinstance(config, Mapping) or candidate.get("config_hash") != canonical_sha256(config):
            raise GovernanceError("candidate config hash mismatch")
        if candidate.get("model_seed") != derived_seed(task, "model", candidate["candidate_id"]):
            raise GovernanceError("candidate model seed mismatch")
        if config.get("n_estimators") != 1000:
            raise GovernanceError("candidate estimator cap changed")
        for name, spec in space["parameters"].items():
            value = config.get(name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise GovernanceError("candidate parameter is not numeric: " + name)
            if not float(spec["minimum"]) <= float(value) <= float(spec["maximum"]):
                raise GovernanceError("candidate parameter outside search bounds: " + name)
            if spec["type"] == "integer" and not isinstance(value, int):
                raise GovernanceError("integer candidate parameter is not integer: " + name)
        hashes.append(candidate["config_hash"])
    if len(set(hashes)) != SEARCH_BUDGET:
        raise GovernanceError("candidate configurations must have 30 unique hashes")


def candidate_manifest(space: Mapping[str, object], task: str, *, implementation_commit: str) -> Mapping[str, object]:
    candidates = materialize_candidates(space, task)
    return {
        "manifest_version": "xgb_phase12_candidate_manifest_v1",
        "search_version": SEARCH_VERSION,
        "status": "FROZEN_READY_TO_RUN",
        "decision_authority": "USER_DELEGATED_AI_PROJECT_DECISION",
        "family": "xgboost",
        "task": task,
        "run_type": "scientific",
        "sampler": SAMPLER_VERSION,
        "master_search_seed": MASTER_SEARCH_SEED,
        "task_seed_derivation": f"{MASTER_SEARCH_SEED}|phase12|xgb|{task}",
        "task_sampler_seed": derived_seed(task),
        "candidate_count": SEARCH_BUDGET,
        "collision_count": 0,
        "candidate_list_hash": canonical_sha256(candidates),
        "search_space_hash": canonical_sha256(space),
        "implementation_commit": implementation_commit,
        "candidates": list(candidates),
        "test_accessed": False,
    }


def validate_candidate_manifest(
    manifest: Mapping[str, object], space: Mapping[str, object]
) -> tuple[Mapping[str, object], ...]:
    task = manifest.get("task")
    if (
        task not in TASKS
        or manifest.get("manifest_version") != "xgb_phase12_candidate_manifest_v1"
        or manifest.get("search_version") != SEARCH_VERSION
        or manifest.get("run_type") != "scientific"
        or manifest.get("sampler") != SAMPLER_VERSION
        or manifest.get("master_search_seed") != MASTER_SEARCH_SEED
        or manifest.get("task_sampler_seed") != derived_seed(task)
        or manifest.get("candidate_count") != SEARCH_BUDGET
        or manifest.get("collision_count") != 0
        or manifest.get("test_accessed") is not False
    ):
        raise GovernanceError("scientific candidate manifest contract mismatch")
    candidates = tuple(manifest.get("candidates", ()))
    validate_candidate_list(candidates, task, space)
    if manifest.get("candidate_list_hash") != canonical_sha256(candidates):
        raise GovernanceError("scientific candidate-list hash mismatch")
    if manifest.get("search_space_hash") != canonical_sha256(space):
        raise GovernanceError("scientific search-space hash mismatch")
    return candidates


def immutable_contract(master: Mapping[str, object]) -> Mapping[str, object]:
    return {field: master.get(field) for field in IMMUTABLE_MASTER_FIELDS}


def validate_master_manifest(root: Path, master: Mapping[str, object]) -> None:
    if (
        master.get("search_version") != SEARCH_VERSION
        or master.get("family") != "xgboost"
        or tuple(master.get("tasks", ())) != TASKS
        or master.get("master_search_seed") != MASTER_SEARCH_SEED
        or master.get("sampler") != SAMPLER_VERSION
        or master.get("max_attempts") != MAX_ATTEMPTS
        or master.get("test_accessed") is not False
    ):
        raise GovernanceError("Phase-12 master search contract mismatch")
    if master.get("immutable_contract_sha256") != canonical_sha256(immutable_contract(master)):
        raise GovernanceError("Phase-12 immutable search contract changed")
    for task, item in master["candidate_manifests"].items():
        path = root / item["path"]
        if sha256_file(path) != item["sha256"]:
            raise GovernanceError("candidate manifest bytes changed: " + task)


def assert_lifecycle_transition(before: Mapping[str, object], after: Mapping[str, object]) -> None:
    allowed = {
        "FROZEN_READY_TO_RUN": {"RUNNING"},
        "RUNNING": {"RUNNING", "COMPLETE", "FAILED_WITH_RECORDED_CANDIDATES"},
        "COMPLETE": {"COMPLETE"},
        "FAILED_WITH_RECORDED_CANDIDATES": {"FAILED_WITH_RECORDED_CANDIDATES"},
    }
    if after.get("status") not in allowed.get(before.get("status"), set()):
        raise GovernanceError("invalid Phase-12 search lifecycle transition")
    if immutable_contract(before) != immutable_contract(after):
        raise GovernanceError("started Phase-12 search contract cannot mutate")


@dataclass(frozen=True)
class RankedCandidate:
    candidate_id: str
    status: str
    metrics: Mapping[str, object]
    probability_type: str = ""


def ranking_key(result: RankedCandidate, task: str):
    metrics = result.metrics
    if result.status != "COMPLETE":
        raise GovernanceError("only COMPLETE candidates may be ranked")
    if task == "recovery":
        values = (metrics["mae24"], metrics["mae48"], metrics["rmse24"])
    elif task == "icu_time":
        values = (
            metrics["median_absolute_error_hours"],
            metrics["mae_hours"],
            metrics["rmse_hours"],
        )
    elif task == "organ_support":
        if result.probability_type != "raw_uncalibrated":
            raise GovernanceError("support ranking requires raw uncalibrated probability")
        values = (-metrics["auprc"], metrics["brier"], -metrics["auroc"])
    else:
        raise GovernanceError("unsupported Phase-12 ranking task")
    if any(isinstance(value, bool) or not math.isfinite(float(value)) for value in values):
        raise GovernanceError("ranking metrics must be finite")
    return tuple(float(value) for value in values) + (result.candidate_id,)


def choose_best(results: Sequence[RankedCandidate], task: str) -> RankedCandidate:
    complete = [result for result in results if result.status == "COMPLETE"]
    if not complete:
        raise GovernanceError("task has zero COMPLETE XGBoost candidates")
    return min(complete, key=lambda result: ranking_key(result, task))
