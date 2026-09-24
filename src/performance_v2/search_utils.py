"""Shared deterministic search utilities for Performance-v2 Phase 2.

DEVELOPMENT_VALIDATION_ONLY.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FOLD_CSV = ROOT / "artifacts/performance_v2/phase2/train_cv_fold_assignment_v1.csv"


def load_fold_assignment() -> Mapping[str, Mapping[str, int]]:
    with FOLD_CSV.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {
        row["subject_id"]: {"xgb_fold_5": int(row["xgb_fold_5"]), "gru_fold_3": int(row["gru_fold_3"])}
        for row in rows
    }


def fold_of(row: Mapping[str, object], assignment, key: str) -> int:
    return assignment[row["subject_id"]][key]


def sample_configs(space: Mapping[str, Tuple[float, float, str]], n: int, seed: int, extra_axes: Mapping[str, Sequence[object]] = None):
    """Deterministic Sobol-like sample via a seeded numpy Generator (log or
    linear scale per axis, declared in `space` as (low, high, "log"/"linear")).
    `extra_axes` are small discrete axes (e.g. feature variant, objective)
    cycled deterministically alongside the continuous sample."""

    rng = np.random.default_rng(seed)
    configs = []
    extra_axes = extra_axes or {}
    extra_names = list(extra_axes.keys())
    extra_lens = [len(extra_axes[name]) for name in extra_names]
    for index in range(n):
        config = {}
        for name, (low, high, scale) in space.items():
            if scale == "log":
                config[name] = float(np.exp(rng.uniform(np.log(low), np.log(high))))
            elif scale == "int":
                config[name] = int(rng.integers(low, high + 1))
            else:
                config[name] = float(rng.uniform(low, high))
        for axis_index, name in enumerate(extra_names):
            config[name] = extra_axes[name][index % extra_lens[axis_index]]
        configs.append(config)
    return configs


def config_hash(config: Mapping[str, object]) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:16]
