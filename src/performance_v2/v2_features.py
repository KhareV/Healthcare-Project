"""Phase 2: build the XGBoost-ready V2 flat matrix (Group A raw canonical
features + a train-only-normalized context variant) and the GRU-ready
tensors (temporal sequence/mask/tslo + statics-with-context).

DEVELOPMENT_VALIDATION_ONLY.

V0's representation is Phase 1's Group A: the same per-bin/per-channel raw
value/observation-mask/TSLO fields plus raw statics that
src/data/xgb_canonical.py's frozen flat feature map is built from (same
source fields, same information content) -- just not passed through the
official z-score/one-hot preprocessor, which XGBoost does not need (trees
are scale-invariant). This adds no information beyond the frozen v1
representation and mutates nothing in src/data/xgb_canonical.py or
configs/synthetic/feature_schema_v2.json.
"""

from __future__ import annotations

from typing import Mapping, Sequence, Tuple

import numpy as np

from performance_v2.context_normalization import apply_normalization, fit_normalization
from performance_v2.diagnostic_features import build_group_a
from performance_v2.feature_matrix import build_feature_matrix_fast


def group_a_feature_names(rows: Sequence[Mapping[str, object]]) -> Tuple[str, ...]:
    per_row = [build_group_a(row) for row in rows]
    return tuple(sorted({name for features in per_row for name in features}))


def build_group_a_matrix(rows: Sequence[Mapping[str, object]], names: Tuple[str, ...] = None) -> Tuple[np.ndarray, Tuple[str, ...]]:
    """`names` fixes the column universe (e.g. computed over train+val
    together) so a row subset (a CV fold, say) never silently drops a
    one-hot category column just because it happens not to appear in that
    subset -- every caller must use the SAME name set across all subsets of
    the same logical dataset."""

    per_row = [build_group_a(row) for row in rows]
    if names is None:
        names = tuple(sorted({name for features in per_row for name in features}))
    onehot = np.asarray(["==" in name for name in names], dtype=bool)
    matrix = np.full((len(rows), len(names)), np.nan, dtype=np.float32)
    matrix[:, onehot] = 0.0
    index_of = {name: i for i, name in enumerate(names)}
    for row_index, features in enumerate(per_row):
        for name, value in features.items():
            if name in index_of:
                matrix[row_index, index_of[name]] = value
    return matrix, names


def build_xgb_matrix_single(rows: Sequence[Mapping[str, object]], variant: str, *, names: Tuple[str, ...] = None):
    """One combined Group-A + normalized-context matrix for a single row
    set (e.g. all TRAIN rows, to be subset by fold afterward). Normalization
    is fit on `rows` itself -- callers doing CV over a fold split of `rows`
    accept full-TRAIN (not fold-local) normalization statistics as a
    documented, low-risk simplification: these are unsupervised, label-free
    linear rescaling parameters over a handful of context features, not a
    label-derived estimator (contrast: the ICU conditional prior, which
    MUST be fold-local and is implemented that way)."""

    matrix_a, names_a = build_group_a_matrix(rows, names=names)
    normalization = fit_normalization(rows, variant)
    context = apply_normalization(rows, normalization)
    matrix = np.concatenate([matrix_a, context], axis=1)
    names_out = names_a + tuple(normalization["feature_names"])
    return matrix, names_out, normalization


def build_xgb_matrices(train_rows, val_rows, variant: str):
    """Returns (train_x, val_x, feature_names, normalization) for one
    variant, group-A base + normalized context, normalization fit on
    train_rows only, using a shared Group-A column universe across both."""

    names_a = group_a_feature_names(list(train_rows) + list(val_rows))
    train_a, _ = build_group_a_matrix(train_rows, names=names_a)
    val_a, _ = build_group_a_matrix(val_rows, names=names_a)
    normalization = fit_normalization(train_rows, variant)
    train_ctx = apply_normalization(train_rows, normalization)
    val_ctx = apply_normalization(val_rows, normalization)
    train_x = np.concatenate([train_a, train_ctx], axis=1)
    val_x = np.concatenate([val_a, val_ctx], axis=1)
    names = names_a + tuple(normalization["feature_names"])
    return train_x, val_x, names, normalization
