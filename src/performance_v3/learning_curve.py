"""Learning curve of the FROZEN V2 recipe (no new search) on a fixed development-evaluation cohort.

The recipe (feature variant, objective, weight mode, hyperparameters, tree count) is read from
artifacts/performance_v2/phase3/selected_models_v2.json and trained exactly as
scripts/performance_v2_phase3_part9_10_final_refit.py trains the final V2 models (xgboost sklearn
API, random_state set, NaN-native, stay-balanced weights only when the recipe says so, no weights
and plain binary:logistic for support). Evaluation uses stay-balanced task metrics on LCE rows.

Development evidence only: the LCE holdout is reused across every point, so this is not final validation.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from performance_v2.diagnostic_features import feature_group_of
from performance_v2.context_v2 import VARIANT_KEYS
from performance_v3 import features as v3feat
from performance_v3 import splits
from performance_v3.models import postprocess_icu
from performance_v3.task_metrics import (
    fast_weighted_auprc, fast_weighted_auroc, icu_metrics, recovery_metrics, stay_weights, weighted_log_loss,
)

LEARNING_CURVE_VERSION = "performance_v3_learning_curve_v1"
SALT = "performance_v3_learning_curve_v1"
SIZES = (425, 850, 1700, 3800, 5900, 8000)
V1_DEV_SIZE = 1700
MODEL_SEEDS = (11, 23, 47)
BOOTSTRAP_REPLICATES = 500
BOOTSTRAP_SEED = 20261011
TASKS = ("recovery24", "recovery48", "icu_stay_time", "organ_support")
SELECTED_V2_REL = "artifacts/performance_v2/phase3/selected_models_v2.json"
TREE_PARAMS = ("max_depth", "min_child_weight", "learning_rate", "subsample", "colsample_bytree", "reg_alpha", "reg_lambda", "gamma")
PRIMARY_NAME = {"recovery24": "mae", "recovery48": "mae", "icu_stay_time": "weighted_median_ae_hours", "organ_support": "auprc"}
HIGHER_IS_BETTER = {"recovery24": False, "recovery48": False, "icu_stay_time": False, "organ_support": True}


class LearningCurveError(RuntimeError):
    pass


# ---------------------------------------------------------------- recipes

def load_v2_recipes(root: Path) -> Dict[str, Dict[str, object]]:
    selected = json.loads((Path(root) / SELECTED_V2_REL).read_text(encoding="utf-8"))
    recipes: Dict[str, Dict[str, object]] = {}
    for task in TASKS:
        entry = selected["tasks"][task]
        hp = entry["hyperparameters"]
        if entry["family"] != "xgboost":
            raise LearningCurveError(f"{task}: V2 family is {entry['family']!r}, recipe expects xgboost")
        recipes[task] = {
            "task": task,
            "feature_task": v3feat.canonical_task(task),
            "variant": entry["feature_variant"],
            "kind": "classification" if task == "organ_support" else "regression",
            "objective": hp.get("objective", "binary:logistic" if task == "organ_support" else None),
            "weight_mode": hp.get("weight_mode", "uniform"),
            "params": {k: hp[k] for k in TREE_PARAMS},
            "n_estimators": int(entry["final_training_budget"]),
            "recipe_ref": SELECTED_V2_REL,
        }
    return recipes


# ---------------------------------------------------------------- subsets

def training_subsets(v1_ids: Sequence[str], pool_ids: Sequence[str], repeat: int, *, sizes: Sequence[int] = SIZES, salt: str = SALT) -> Dict[int, Tuple[str, ...]]:
    """Nested within a repeat; independent across repeats. Sizes <= 1700 come from the V1 dev subjects only,
    larger sizes from the whole development pool (protocol: learning_curve.nesting)."""

    repeat_salt = f"{salt}||repeat={int(repeat)}"
    small = [s for s in sizes if s <= len(v1_ids)]
    large = [s for s in sizes if s > len(v1_ids)]
    out: Dict[int, Tuple[str, ...]] = {}
    if small:
        out.update(splits.nested_subject_subsets(v1_ids, small, salt=repeat_salt))
    if large:
        out.update(splits.nested_subject_subsets(pool_ids, large, salt=repeat_salt))
    return out


def assert_disjoint(train_ids: Iterable[str], eval_ids: Iterable[str]) -> None:
    overlap = set(train_ids) & set(eval_ids)
    if overlap:
        raise LearningCurveError(f"{len(overlap)} subjects appear in both the training subset and the evaluation cohort")


# ---------------------------------------------------------------- design matrix

def group_a_columns(names: Sequence[str]) -> np.ndarray:
    return np.asarray([i for i, n in enumerate(names) if feature_group_of(n) == "A"], dtype=np.int64)


def design_matrix(cache: v3feat.FeatureCache, row_idx: np.ndarray, variant: str, normalization: Mapping[str, object]) -> np.ndarray:
    """[Group A raw columns | train-only normalised context] exactly like performance_v2.v2_features."""

    a_cols = group_a_columns(cache.names)
    base = cache.matrix[np.asarray(row_idx)][:, a_cols]
    ctx = v3feat.apply_context_normalization(cache, normalization, row_mask=_mask(cache.n_rows, row_idx))
    # apply_context_normalization returns rows in index order of the boolean mask (ascending)
    order = np.argsort(np.argsort(np.asarray(row_idx)))
    return np.concatenate([base, ctx[order]], axis=1)


def _mask(n: int, idx: np.ndarray) -> np.ndarray:
    mask = np.zeros(n, dtype=bool)
    mask[np.asarray(idx)] = True
    return mask


# ---------------------------------------------------------------- training / evaluation

def train_v2_recipe(recipe: Mapping[str, object], X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray], *, seed: int, n_jobs: int = 8):
    import xgboost as xgb

    common = dict(random_state=int(seed), missing=np.nan, n_estimators=int(recipe["n_estimators"]), n_jobs=n_jobs, **recipe["params"])
    if recipe["kind"] == "classification":
        model = xgb.XGBClassifier(objective="binary:logistic", **common)
        model.fit(X, y)
    else:
        model = xgb.XGBRegressor(objective=recipe["objective"], **common)
        model.fit(X, y, sample_weight=sample_weight)
    return model


def predict(model, X: np.ndarray, kind: str) -> np.ndarray:
    return model.predict_proba(X)[:, 1] if kind == "classification" else model.predict(X)


def evaluate(task: str, y: np.ndarray, pred: np.ndarray, stay_ids: Sequence[str], times: Sequence[str]) -> Dict[str, float]:
    if task in ("recovery24", "recovery48"):
        m = recovery_metrics(y, pred, stay_ids, times, horizon="24h" if task == "recovery24" else "48h")
        return {k: m[k] for k in ("mae", "rmse", "median_ae")} | {"n_examples": m["n_examples"], "n_stays": m["n_stays"]}
    if task == "icu_stay_time":
        hours = np.expm1(y)
        m = icu_metrics(hours, pred, stay_ids, times)
        return {k: m[k] for k in ("weighted_median_ae_hours", "mae", "rmse")} | {"n_examples": m["n_examples"], "n_stays": m["n_stays"]}
    w = stay_weights(stay_ids)
    return {
        "auprc": fast_weighted_auprc(y, pred, w), "auroc": fast_weighted_auroc(y, pred, w),
        "brier": float(np.sum(w * (pred - y) ** 2) / np.sum(w)), "log_loss": weighted_log_loss(y, pred, w),
        "n_examples": int(len(y)), "n_stays": int(len(set(stay_ids))),
        "prevalence_rows": float(np.mean(y)), "prevalence_stay_balanced": float(np.sum(w * y) / np.sum(w)),
    }


def eligible_index(cache: v3feat.FeatureCache, task: str, subject_ids: Optional[Iterable[str]] = None) -> np.ndarray:
    mask = v3feat.task_mask(cache.meta, recipe_feature_task(task))
    if subject_ids is not None:
        keep = np.isin(cache.meta["subject_id"], np.asarray(sorted(set(subject_ids)), dtype=str))
        mask = mask & keep
    return np.flatnonzero(mask)


def recipe_feature_task(task: str) -> str:
    return v3feat.canonical_task(task)


def target_vector(cache: v3feat.FeatureCache, task: str, idx: np.ndarray) -> np.ndarray:
    col = v3feat.TASK_COLUMNS[recipe_feature_task(task)][1]
    return cache.meta[col][idx].astype(np.float64)


def fit_and_score(recipe: Mapping[str, object], train: v3feat.FeatureCache, train_idx: np.ndarray, ev: v3feat.FeatureCache, ev_idx: np.ndarray, *, seed: int, n_jobs: int = 8) -> Tuple[Dict[str, object], np.ndarray]:
    task = str(recipe["task"])
    norm = v3feat.fit_context_normalization(train, str(recipe["variant"]), row_mask=_mask(train.n_rows, train_idx))
    Xtr = design_matrix(train, train_idx, str(recipe["variant"]), norm)
    ytr = target_vector(train, task, train_idx)
    w = None
    if recipe["weight_mode"] == "stay_balanced" and recipe["kind"] == "regression":
        sid = [str(s) for s in train.meta["stay_id"][train_idx]]
        w = stay_weights(sid)
    t0 = time.time()
    model = train_v2_recipe(recipe, Xtr, ytr, w, seed=seed, n_jobs=n_jobs)
    train_seconds = time.time() - t0
    Xev = design_matrix(ev, ev_idx, str(recipe["variant"]), norm)
    t0 = time.time()
    pred = predict(model, Xev, str(recipe["kind"]))
    predict_seconds = time.time() - t0
    yev = target_vector(ev, task, ev_idx)
    metrics = evaluate(task, yev, pred, [str(s) for s in ev.meta["stay_id"][ev_idx]], [str(t) for t in ev.meta["prediction_time"][ev_idx]])
    metrics.update(
        {
            "train_seconds": round(train_seconds, 3),
            "predict_seconds_per_1000_rows": round(1000.0 * predict_seconds / max(1, len(ev_idx)), 4),
            "n_train_rows": int(len(train_idx)), "n_train_stays": int(len(set(train.meta["stay_id"][train_idx]))),
        }
    )
    return metrics, pred


# ---------------------------------------------------------------- trend classification

def relative_improvement(task: str, small: float, large: float) -> float:
    """Positive = larger training set is better."""

    if HIGHER_IS_BETTER[task]:
        return (large - small) / abs(small)
    return (small - large) / abs(small)


def label_for(rel_per_doubling: float) -> str:
    if rel_per_doubling >= 0.03:
        return "strong"
    if rel_per_doubling >= 0.005:
        return "marginal"
    if rel_per_doubling > -0.005:
        return "plateau"
    return "degrading"


def classify_trend(task: str, sizes: Sequence[int], values: Sequence[float], *, from_size: int = V1_DEV_SIZE) -> Dict[str, object]:
    pairs = sorted(zip(sizes, values))
    segments = []
    for (s0, v0), (s1, v1) in zip(pairs[:-1], pairs[1:]):
        doublings = math.log2(s1 / s0)
        rel = relative_improvement(task, v0, v1) / doublings
        segments.append({"from": s0, "to": s1, "relative_improvement_per_doubling": rel, "label": label_for(rel)})
    pts = [(s, v) for s, v in pairs if s >= from_size]
    overall = None
    if len(pts) >= 2:
        d = math.log2(pts[-1][0] / pts[0][0])
        rel = relative_improvement(task, pts[0][1], pts[-1][1]) / d
        overall = {"from": pts[0][0], "to": pts[-1][0], "relative_improvement_per_doubling": rel, "label": label_for(rel)}
    return {"segments": segments, "overall_from_1700": overall}
