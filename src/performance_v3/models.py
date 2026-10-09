"""Uniform tabular fit/predict adapter for the V3 candidate families.

Families: xgboost, lightgbm, catboost (tree, NaN-native), ridge / logistic
(median-impute + standardise fitted on the supplied training matrix only).
Sequence models are NOT handled here (see performance_v2.gru_training).

DEVELOPMENT_VALIDATION_ONLY. Nothing in this module reads data, labels or
any artifact; callers supply matrices.
"""

from __future__ import annotations

import math
import resource
import sys
import time
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from performance_v2.search_utils import config_hash, sample_configs

FAMILIES = ("xgboost", "lightgbm", "catboost", "ridge", "logistic")
TASKS = ("recovery24", "recovery48", "icu_stay_time", "organ_support")
REGRESSION_OBJECTIVES = ("squarederror", "pseudohubererror", "absoluteerror")
WEIGHT_MODES = ("none", "stay_balanced")
CLASS_WEIGHTS = ("none", "scale_pos_weight")
N_ESTIMATORS_CAP = 1500
DEFAULT_EARLY_STOPPING_ROUNDS = 50
MODELS_VERSION = "performance_v3_models_v1"

# (low, high, scale) with scale in {"log", "linear", "int"} as sample_configs expects.
HYPERPARAM_SPACES: Mapping[str, Mapping[str, Tuple[float, float, str]]] = {
    "xgboost": {
        "max_depth": (3, 8, "int"),
        "learning_rate": (0.01, 0.2, "log"),
        "subsample": (0.6, 1.0, "linear"),
        "colsample_bytree": (0.6, 1.0, "linear"),
        "min_child_weight": (1, 10, "int"),
        "reg_alpha": (1e-6, 1.0, "log"),
        "reg_lambda": (0.1, 20.0, "log"),
        "gamma": (0.0, 5.0, "linear"),
    },
    "lightgbm": {
        "num_leaves": (8, 64, "int"),
        "learning_rate": (0.01, 0.2, "log"),
        "subsample": (0.6, 1.0, "linear"),
        "colsample_bytree": (0.6, 1.0, "linear"),
        "min_data_in_leaf": (10, 200, "int"),
        "min_child_weight": (1e-3, 10.0, "log"),
        "reg_alpha": (1e-6, 1.0, "log"),
        "reg_lambda": (0.1, 20.0, "log"),
        "gamma": (0.0, 1.0, "linear"),
    },
    "catboost": {
        "depth": (4, 8, "int"),
        "learning_rate": (0.01, 0.2, "log"),
        "l2_leaf_reg": (1.0, 20.0, "log"),
        "subsample": (0.6, 1.0, "linear"),
        "colsample_bytree": (0.6, 1.0, "linear"),
    },
    "ridge": {"alpha": (1e-2, 1e3, "log")},
    "logistic": {"C": (1e-3, 1e2, "log")},
}


class ModelError(ValueError):
    """Raised for invalid family/config/task combinations."""


def is_classification(task: str) -> bool:
    if task not in TASKS:
        raise ModelError("unknown task: " + repr(task))
    return task == "organ_support"


def postprocess_icu(raw_log1p) -> np.ndarray:
    """Frozen ICU output domain: expm1(max(raw, 0))."""

    out = np.expm1(np.clip(np.asarray(raw_log1p, dtype=np.float64), 0.0, None))
    if not np.all(np.isfinite(out)):
        raise ModelError("postprocessed remaining ICU hours must be finite")
    return out


def sample_family_configs(family: str, n: int, seed: int, task: str) -> List[Dict[str, object]]:
    """Deterministic configs: continuous axes via sample_configs plus cycled
    discrete axes (objective for regression, weight_mode, class_weight for
    support). `n_estimators` is the early-stopping cap, a constant."""

    if family not in FAMILIES:
        raise ModelError("unknown family: " + repr(family))
    classification = is_classification(task)
    axes: Dict[str, Sequence[object]] = {"weight_mode": WEIGHT_MODES}
    if classification:
        axes["objective"] = ("binary",)
        axes["class_weight"] = CLASS_WEIGHTS
    elif family in ("ridge",):
        axes["objective"] = ("squarederror",)
    else:
        axes["objective"] = REGRESSION_OBJECTIVES
    configs = sample_configs(HYPERPARAM_SPACES[family], n, seed, extra_axes=axes)
    out = []
    for config in configs:
        config = dict(config)
        if family in ("xgboost", "lightgbm", "catboost"):
            config["n_estimators"] = N_ESTIMATORS_CAP
            config["early_stopping_rounds"] = DEFAULT_EARLY_STOPPING_ROUNDS
        config["family"] = family
        config["task"] = task
        config["config_hash"] = config_hash({k: v for k, v in config.items() if k != "config_hash"})
        out.append(config)
    return out


def _peak_rss_mb() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return float(peak) / (1024.0 * 1024.0) if sys.platform == "darwin" else float(peak) / 1024.0


def _effective_weights(config: Mapping[str, object], w) -> Optional[np.ndarray]:
    mode = config.get("weight_mode", "none")
    if mode not in WEIGHT_MODES:
        raise ModelError("unknown weight_mode: " + repr(mode))
    if mode == "none" or w is None:
        return None
    w = np.asarray(w, dtype=np.float64)
    if not np.all(np.isfinite(w)) or np.any(w < 0):
        raise ModelError("training weights must be finite and nonnegative")
    return w


def _scale_pos_weight(config: Mapping[str, object], y) -> Optional[float]:
    if config.get("class_weight", "none") != "scale_pos_weight":
        return None
    y = np.asarray(y)
    pos = float(np.sum(y == 1))
    neg = float(np.sum(y == 0))
    return neg / pos if pos > 0 and neg > 0 else 1.0


def _objective(config: Mapping[str, object], classification: bool) -> str:
    if classification:
        return "binary"
    objective = config.get("objective", "squarederror")
    if objective not in REGRESSION_OBJECTIVES:
        raise ModelError("unknown regression objective: " + repr(objective))
    return objective


def _es_parts(es):
    if es is None:
        return None
    x_es, y_es, w_es = es
    return np.asarray(x_es), np.asarray(y_es, dtype=np.float64), (None if w_es is None else np.asarray(w_es, dtype=np.float64))


def _fit_xgboost(config, X_tr, y_tr, w, X_ev, classification, es, nthread, seed):
    import xgboost as xgb

    objective = _objective(config, classification)
    params = {
        "objective": "binary:logistic" if classification else {
            "squarederror": "reg:squarederror",
            "pseudohubererror": "reg:pseudohubererror",
            "absoluteerror": "reg:absoluteerror",
        }[objective],
        "eval_metric": "logloss" if classification else "mae",
        "tree_method": "hist",
        "max_depth": int(config.get("max_depth", 6)),
        "eta": float(config.get("learning_rate", 0.05)),
        "subsample": float(config.get("subsample", 1.0)),
        "colsample_bytree": float(config.get("colsample_bytree", 1.0)),
        "min_child_weight": float(config.get("min_child_weight", 1)),
        "reg_alpha": float(config.get("reg_alpha", 0.0)),
        "reg_lambda": float(config.get("reg_lambda", 1.0)),
        "gamma": float(config.get("gamma", 0.0)),
        "nthread": int(nthread),
        "seed": int(seed),
        "verbosity": 0,
    }
    spw = _scale_pos_weight(config, y_tr) if classification else None
    if spw is not None:
        params["scale_pos_weight"] = spw
    n_estimators = int(min(config.get("n_estimators", N_ESTIMATORS_CAP), N_ESTIMATORS_CAP))
    dtrain = xgb.DMatrix(np.asarray(X_tr, dtype=np.float32), label=np.asarray(y_tr, dtype=np.float64), weight=w, nthread=nthread)
    deval = xgb.DMatrix(np.asarray(X_ev, dtype=np.float32), nthread=nthread)
    kwargs = {}
    parts = _es_parts(es)
    if parts is not None:
        x_es, y_es, w_es = parts
        des = xgb.DMatrix(np.asarray(x_es, dtype=np.float32), label=y_es, weight=w_es if config.get("weight_mode", "none") == "stay_balanced" else None, nthread=nthread)
        kwargs = {"evals": [(des, "es")], "early_stopping_rounds": int(config.get("early_stopping_rounds", DEFAULT_EARLY_STOPPING_ROUNDS)), "verbose_eval": False}
    booster = xgb.train(params, dtrain, num_boost_round=n_estimators, **kwargs)
    if parts is not None:
        best = int(booster.best_iteration) + 1
    else:
        best = n_estimators
    return booster.predict(deval, iteration_range=(0, best)), best


def _fit_lightgbm(config, X_tr, y_tr, w, X_ev, classification, es, nthread, seed):
    import lightgbm as lgb

    objective = _objective(config, classification)
    params = {
        "objective": "binary" if classification else {"squarederror": "regression", "pseudohubererror": "huber", "absoluteerror": "regression_l1"}[objective],
        "learning_rate": float(config.get("learning_rate", 0.05)),
        "num_leaves": int(config.get("num_leaves", 31)),
        "max_depth": int(config.get("max_depth", -1)),
        "min_data_in_leaf": int(config.get("min_data_in_leaf", 20)),
        "min_sum_hessian_in_leaf": float(config.get("min_child_weight", 1e-3)),
        "feature_fraction": float(config.get("colsample_bytree", 1.0)),
        "bagging_fraction": float(config.get("subsample", 1.0)),
        "bagging_freq": 1 if float(config.get("subsample", 1.0)) < 1.0 else 0,
        "lambda_l1": float(config.get("reg_alpha", 0.0)),
        "lambda_l2": float(config.get("reg_lambda", 0.0)),
        "min_gain_to_split": float(config.get("gamma", 0.0)),
        "num_threads": int(nthread),
        "seed": int(seed),
        "deterministic": True,
        "force_row_wise": True,
        "verbose": -1,
    }
    spw = _scale_pos_weight(config, y_tr) if classification else None
    if spw is not None:
        params["scale_pos_weight"] = spw
    n_estimators = int(min(config.get("n_estimators", N_ESTIMATORS_CAP), N_ESTIMATORS_CAP))
    dtrain = lgb.Dataset(np.asarray(X_tr, dtype=np.float64), label=np.asarray(y_tr, dtype=np.float64), weight=w, free_raw_data=False)
    parts = _es_parts(es)
    valid_sets, callbacks = None, None
    if parts is not None:
        x_es, y_es, w_es = parts
        valid_sets = [lgb.Dataset(np.asarray(x_es, dtype=np.float64), label=y_es, weight=w_es if config.get("weight_mode", "none") == "stay_balanced" else None, reference=dtrain)]
        callbacks = [lgb.early_stopping(int(config.get("early_stopping_rounds", DEFAULT_EARLY_STOPPING_ROUNDS)), verbose=False)]
    booster = lgb.train(params, dtrain, num_boost_round=n_estimators, valid_sets=valid_sets, callbacks=callbacks)
    best = int(booster.best_iteration) if parts is not None and booster.best_iteration else n_estimators
    return booster.predict(np.asarray(X_ev, dtype=np.float64), num_iteration=best), best


def _fit_catboost(config, X_tr, y_tr, w, X_ev, classification, es, nthread, seed):
    from catboost import CatBoostClassifier, CatBoostRegressor, Pool

    objective = _objective(config, classification)
    subsample = float(config.get("subsample", 1.0))
    common = {
        "iterations": int(min(config.get("n_estimators", config.get("iterations", N_ESTIMATORS_CAP)), N_ESTIMATORS_CAP)),
        "learning_rate": float(config.get("learning_rate", 0.05)),
        "depth": int(config.get("depth", 6)),
        "l2_leaf_reg": float(config.get("l2_leaf_reg", 3.0)),
        "rsm": float(config.get("colsample_bytree", 1.0)),
        "thread_count": int(nthread),
        "random_seed": int(seed),
        "verbose": 0,
        "allow_writing_files": False,
        "bootstrap_type": "Bernoulli",
        "subsample": subsample,
    }
    if classification:
        spw = _scale_pos_weight(config, y_tr)
        if spw is not None:
            common["scale_pos_weight"] = spw
        model = CatBoostClassifier(loss_function="Logloss", **common)
    else:
        loss = {"squarederror": "RMSE", "pseudohubererror": "Huber:delta=1.0", "absoluteerror": "MAE"}[objective]
        model = CatBoostRegressor(loss_function=loss, **common)
    train_pool = Pool(np.asarray(X_tr, dtype=np.float64), label=np.asarray(y_tr, dtype=np.float64), weight=w)
    parts = _es_parts(es)
    fit_kwargs = {}
    if parts is not None:
        x_es, y_es, w_es = parts
        fit_kwargs = {
            "eval_set": Pool(np.asarray(x_es, dtype=np.float64), label=y_es, weight=w_es if config.get("weight_mode", "none") == "stay_balanced" else None),
            "early_stopping_rounds": int(config.get("early_stopping_rounds", DEFAULT_EARLY_STOPPING_ROUNDS)),
            "use_best_model": True,
        }
    model.fit(train_pool, **fit_kwargs)
    best = int(model.get_best_iteration() + 1) if parts is not None and model.get_best_iteration() is not None else common["iterations"]
    x_ev = np.asarray(X_ev, dtype=np.float64)
    pred = model.predict_proba(x_ev)[:, 1] if classification else model.predict(x_ev)
    return np.asarray(pred, dtype=np.float64), best


def _fit_linear(family, config, X_tr, y_tr, w, X_ev, classification, seed):
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.preprocessing import StandardScaler

    X_tr = np.asarray(X_tr, dtype=np.float64)
    X_ev = np.asarray(X_ev, dtype=np.float64)
    imputer = SimpleImputer(strategy="median", keep_empty_features=True).fit(X_tr)
    scaler = StandardScaler().fit(imputer.transform(X_tr))
    a = np.nan_to_num(scaler.transform(imputer.transform(X_tr)))
    b = np.nan_to_num(scaler.transform(imputer.transform(X_ev)))
    y = np.asarray(y_tr, dtype=np.float64)
    if classification:
        class_weight = "balanced" if config.get("class_weight", "none") == "scale_pos_weight" else None
        model = LogisticRegression(C=float(config.get("C", 1.0)), class_weight=class_weight, max_iter=2000, solver="lbfgs", random_state=int(seed))
        model.fit(a, y.astype(int), sample_weight=w)
        return model.predict_proba(b)[:, 1], 0
    model = Ridge(alpha=float(config.get("alpha", 1.0)))
    model.fit(a, y, sample_weight=w)
    return model.predict(b), 0


def fit_predict(
    family: str,
    config: Mapping[str, object],
    X_tr,
    y_tr,
    w_tr,
    X_ev,
    *,
    task: str,
    es: Optional[Tuple[object, object, object]] = None,
    nthread: int = 8,
    seed: int = 0,
):
    """Fit on (X_tr, y_tr) and predict X_ev. Returns (pred, info).

    `config["weight_mode"]` decides whether `w_tr` is used ("stay_balanced")
    or ignored ("none"). `es=(X_es, y_es, w_es)` is the inner early-stopping
    split drawn from TRAINING subjects only. Regression returns the raw
    target-domain prediction (log1p for ICU: apply postprocess_icu);
    organ_support returns P(y=1).
    """

    if family not in FAMILIES:
        raise ModelError("unknown family: " + repr(family))
    classification = is_classification(task)
    if classification and family == "ridge":
        raise ModelError("ridge is a regression family; use logistic for organ_support")
    if not classification and family == "logistic":
        raise ModelError("logistic is a classification family; use ridge for regression tasks")
    y_arr = np.asarray(y_tr, dtype=np.float64)
    if len(y_arr) != len(X_tr) or len(y_arr) == 0:
        raise ModelError("X_tr and y_tr must have equal nonzero length")
    if classification and not set(np.unique(y_arr)).issubset({0.0, 1.0}):
        raise ModelError("organ_support labels must be binary")
    w = _effective_weights(config, w_tr)
    start = time.perf_counter()
    if family == "xgboost":
        pred, best = _fit_xgboost(config, X_tr, y_arr, w, X_ev, classification, es, nthread, seed)
    elif family == "lightgbm":
        pred, best = _fit_lightgbm(config, X_tr, y_arr, w, X_ev, classification, es, nthread, seed)
    elif family == "catboost":
        pred, best = _fit_catboost(config, X_tr, y_arr, w, X_ev, classification, es, nthread, seed)
    else:
        pred, best = _fit_linear(family, config, X_tr, y_arr, w, X_ev, classification, seed)
    pred = np.asarray(pred, dtype=np.float64)
    if not np.all(np.isfinite(pred)):
        raise ModelError("model produced non-finite predictions")
    info = {
        "best_iteration": int(best),
        "fit_seconds": float(time.perf_counter() - start),
        "peak_rss_mb": _peak_rss_mb(),
        "family": family,
        "task": task,
        "models_version": MODELS_VERSION,
    }
    return pred, info
