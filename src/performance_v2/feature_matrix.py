"""Build a dense diagnostic feature matrix (X, feature_names, groups) for a
list of TRAIN/VALIDATION rows, using performance_v2.diagnostic_features.
DEVELOPMENT_DIAGNOSTIC_ONLY.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np

from performance_v2.diagnostic_features import build_all_groups, feature_group_of

CACHE_PATH = "artifacts/performance_v2/phase1/_cache/sofa_trend_cache_v1.json"


def load_sofa_trend_cache(root: Path) -> Dict[Tuple[str, str], Mapping[str, float]]:
    raw = json.loads((root / CACHE_PATH).read_text(encoding="utf-8"))
    return {tuple(key.split("||", 1)): value for key, value in raw.items()}


def build_feature_matrix(rows: Sequence[Mapping[str, object]], sofa_trends: Mapping[Tuple[str, str], Mapping[str, float]]):
    per_row_dicts = []
    for row in rows:
        trend = sofa_trends.get((row["stay_id"], row["prediction_time"]), {})
        features = build_all_groups(row, baseline_sofa=row.get("baseline_sofa"), sofa_trend=trend)
        per_row_dicts.append(features)
    names = sorted({name for features in per_row_dicts for name in features})
    matrix = np.full((len(rows), len(names)), np.nan, dtype=np.float32)
    for row_index, features in enumerate(per_row_dicts):
        for name, value in features.items():
            matrix[row_index, names.index(name)] = value
    return matrix, names


def build_feature_matrix_fast(rows: Sequence[Mapping[str, object]], sofa_trends: Mapping[Tuple[str, str], Mapping[str, float]]):
    """Same as build_feature_matrix but avoids O(n^2) name-indexing."""

    per_row_dicts = []
    name_set = set()
    for row in rows:
        trend = sofa_trends.get((row["stay_id"], row["prediction_time"]), {})
        features = build_all_groups(row, baseline_sofa=row.get("baseline_sofa"), sofa_trend=trend)
        per_row_dicts.append(features)
        name_set.update(features.keys())
    names = sorted(name_set)
    index_of = {name: i for i, name in enumerate(names)}
    # One-hot categorical dummies ("==" in the name) are 0/1 encodings: a
    # row not carrying a particular category means that dummy is 0, not
    # "unknown" -- so those columns default to 0.0, not NaN, before values
    # are filled in.
    onehot_columns = np.asarray(["==" in name for name in names], dtype=bool)
    matrix = np.full((len(rows), len(names)), np.nan, dtype=np.float32)
    matrix[:, onehot_columns] = 0.0
    for row_index, features in enumerate(per_row_dicts):
        for name, value in features.items():
            matrix[row_index, index_of[name]] = value
    return matrix, names


def columns_for_groups(names: Sequence[str], groups: Sequence[str]) -> np.ndarray:
    allowed = set(groups)
    return np.asarray([feature_group_of(name) in allowed for name in names], dtype=bool)
