"""Frozen candidate generation and ranking for final governed GRU search."""
from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from training.checkpoint import metadata_sidecar_path
from vedant_infra.hashing import sha256_file


SEARCH_VERSION = "vedant_final_gru_validation_search_v1"
SAMPLER_VERSION = "deterministic_latin_hypercube_v1"
MASTER_SEED = 20260921
BUDGET = 30
MAX_ATTEMPTS = 3
TASKS = ("recovery", "icu_time", "organ_support")
PREFIX = {"recovery": "gru-recovery", "icu_time": "gru-icu-time", "organ_support": "gru-support"}


class GRUSearchError(RuntimeError):
    pass


def canonical_sha256(value: object) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def derived_seed(task: str, purpose: str, candidate_id: str | None = None) -> int:
    if task not in TASKS:
        raise GRUSearchError("unsupported GRU task")
    if purpose == "sampler":
        text = f"{MASTER_SEED}|vedant_final_gru|{task}|sampler"
    elif purpose == "model" and candidate_id:
        text = f"{MASTER_SEED}|vedant_final_gru|{task}|{candidate_id}|model"
    else:
        raise GRUSearchError("invalid seed derivation request")
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:4], "big")


def validate_search_space(space: Mapping[str, object]) -> None:
    expected_parameters = {
        "hidden_dim": {"type": "categorical", "values": [64, 128, 192]},
        "num_layers": {"type": "categorical", "values": [1, 2]},
        "dropout": {"type": "continuous", "minimum": 0.1, "maximum": 0.4, "distribution": "uniform"},
        "learning_rate": {"type": "continuous", "minimum": 0.0001, "maximum": 0.003, "distribution": "log_uniform"},
        "batch_size": {"type": "categorical", "values": [64, 128]},
        "weight_decay": {"type": "continuous", "minimum": 0.000001, "maximum": 0.001, "distribution": "log_uniform"},
    }
    expected_fixed = {
        "optimizer": "adamw", "max_epochs": 60, "early_stopping_patience": 8,
        "huber_delta": 1.0, "unidirectional": True, "recurrent_dropout": 0.0,
        "include_observation_mask": True, "include_tslo": True, "restore_best_epoch": True,
    }
    if (space.get("search_space_version") != "vedant_final_gru_search_space_v1"
            or space.get("family") != "gru" or space.get("budget_per_task") != BUDGET
            or space.get("master_seed") != MASTER_SEED or space.get("sampler") != SAMPLER_VERSION
            or space.get("parameters") != expected_parameters or space.get("fixed") != expected_fixed
            or space.get("test_accessed") is not False):
        raise GRUSearchError("final GRU search space differs from frozen Stage-1 contract")


def _scale(spec, fraction: float):
    low, high = float(spec["minimum"]), float(spec["maximum"])
    if spec["distribution"] == "uniform":
        return low + fraction * (high - low)
    return math.exp(math.log(low) + fraction * (math.log(high) - math.log(low)))


def materialize_candidates(space: Mapping[str, object], task: str):
    validate_search_space(space)
    rng = random.Random(derived_seed(task, "sampler"))
    dimensions = ("hidden_dim", "num_layers", "dropout", "learning_rate", "batch_size", "weight_decay")
    fractions = {}
    for name in dimensions:
        strata = list(range(BUDGET)); rng.shuffle(strata)
        fractions[name] = [(index + rng.random()) / BUDGET for index in strata]
    candidates = []
    for index in range(BUDGET):
        config = {}
        for name in dimensions:
            spec, fraction = space["parameters"][name], fractions[name][index]
            if spec["type"] == "categorical":
                values = spec["values"]
                config[name] = values[min(int(fraction * len(values)), len(values) - 1)]
            else:
                config[name] = _scale(spec, fraction)
        candidate_id = f"{PREFIX[task]}-{index + 1:03d}"
        candidates.append({
            "candidate_id": candidate_id, "config": config,
            "config_hash": canonical_sha256(config),
            "model_seed": derived_seed(task, "model", candidate_id),
        })
    validate_candidates(candidates, task)
    return tuple(candidates)


def validate_candidates(candidates, task: str) -> None:
    if len(candidates) != BUDGET:
        raise GRUSearchError("exactly 30 GRU candidates are required")
    expected_ids = [f"{PREFIX[task]}-{index:03d}" for index in range(1, BUDGET + 1)]
    if [row.get("candidate_id") for row in candidates] != expected_ids:
        raise GRUSearchError("GRU candidate IDs are not the exact frozen sequence")
    hashes = [row.get("config_hash") for row in candidates]
    if len(set(hashes)) != BUDGET:
        raise GRUSearchError("GRU candidate configs must be unique")
    for row in candidates:
        if canonical_sha256(row["config"]) != row["config_hash"]:
            raise GRUSearchError("GRU candidate config hash mismatch")
        if row["model_seed"] != derived_seed(task, "model", row["candidate_id"]):
            raise GRUSearchError("GRU candidate model seed mismatch")


def candidate_manifest(space, task: str, implementation_commit: str):
    candidates = materialize_candidates(space, task)
    return {
        "manifest_version": "vedant_final_gru_candidate_manifest_v1",
        "search_version": SEARCH_VERSION, "run_type": "scientific", "family": "gru",
        "task": task, "status": "FROZEN_BEFORE_TRAINING", "candidate_count": BUDGET,
        "sampler": SAMPLER_VERSION, "master_seed": MASTER_SEED,
        "task_sampler_seed": derived_seed(task, "sampler"),
        "candidate_list_hash": canonical_sha256(candidates),
        "implementation_commit": implementation_commit, "candidates": list(candidates),
        "test_accessed": False,
    }


def ranking_key(task: str, row: Mapping[str, object]):
    if row.get("status") != "COMPLETE":
        raise GRUSearchError("only COMPLETE GRU candidates may be ranked")
    if task == "recovery":
        return (row["mae24"], row["mae48"], row["rmse24"], row["candidate_id"])
    if task == "icu_time":
        return (row["median_absolute_error_hours"], row["mae_hours"], row["rmse_hours"], row["candidate_id"])
    if task == "organ_support":
        if row.get("probability_type") != "raw_uncalibrated":
            raise GRUSearchError("support candidate must remain uncalibrated")
        return (-row["auprc"], row["brier"], -row["auroc"], row["candidate_id"])
    raise GRUSearchError("unsupported GRU task")


def choose_best(task: str, rows: Sequence[Mapping[str, object]]):
    complete = [row for row in rows if row.get("status") == "COMPLETE"]
    if not complete:
        raise GRUSearchError("at least one COMPLETE GRU candidate is required")
    return min(complete, key=lambda row: ranking_key(task, row))


def validate_checkpoint_lineage(checkpoint_path: Path, expected: Mapping[str, object]):
    sidecar = json.loads(metadata_sidecar_path(checkpoint_path).read_text(encoding="utf-8"))
    if sha256_file(checkpoint_path) != sidecar.get("checkpoint_sha256"):
        raise GRUSearchError("GRU checkpoint content hash mismatch")
    changed = [key for key, value in expected.items() if sidecar.get(key) != value]
    if changed:
        raise GRUSearchError("GRU checkpoint lineage mismatch: " + ", ".join(changed))
    return sidecar
