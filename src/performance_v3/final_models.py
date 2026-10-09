"""Persistent fitted-model API for the V3 candidate families.

`fit_final` re-uses the exact training code of `performance_v3.models`
(`_fit_xgboost`, `_fit_lightgbm`, `_fit_catboost`) by capturing the model object
those helpers train, so a persisted model is byte-for-byte the model the search
evaluated (no second copy of the parameter mapping). Linear families are fitted
with the same sklearn calls and stored as plain numpy parameters, so their
`predict` is pure numpy and reload-exact.

A saved model carries its ordered `feature_columns`; `predict` refuses any input
whose columns differ in name or order (fail closed). Every file is SHA-256 bound
in `model_meta.json`; `load_fitted_model` verifies all of them.

Prediction domain (same as `models.fit_predict`): regression returns the raw
target-domain value (ICU: log1p hours -> apply `models.postprocess_icu`);
organ_support returns P(y=1).

DEVELOPMENT/FINAL-REFIT code only: nothing here reads data, labels or any
sealed cohort.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np

from performance_v3 import models as M

FINAL_MODELS_VERSION = "performance_v3_final_models_v1"
META_NAME = "model_meta.json"
TREE_FAMILIES = ("xgboost", "lightgbm", "catboost")
N_ESTIMATORS_MULTIPLIER = 1.10


class FinalModelError(ValueError):
    pass


class ColumnMismatchError(FinalModelError):
    pass


class HashMismatchError(FinalModelError):
    pass


def n_estimators_rule(best_iterations: Sequence[int]) -> int:
    """Protocol final_refit rule: ceil(1.10 * median best iteration across CV folds)."""

    values = [int(v) for v in best_iterations]
    if not values or any(v <= 0 for v in values):
        raise FinalModelError("best_iterations must be a non-empty list of positive integers")
    return int(math.ceil(round(N_ESTIMATORS_MULTIPLIER * float(np.median(values)), 9)))


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def names_sha256(names: Sequence[str]) -> str:
    return _sha256_bytes(json.dumps(list(names), separators=(",", ":"), ensure_ascii=True).encode("utf-8"))


def _canonical_json(value) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode("utf-8")


@contextlib.contextmanager
def _capture(family: str):
    """Capture the model object the models.py helper trains (restored on exit)."""

    store: Dict[str, object] = {}
    restore = []
    if family == "xgboost":
        import xgboost as xgb

        original = xgb.train

        def wrapped(*args, **kwargs):
            booster = original(*args, **kwargs)
            store["model"] = booster
            return booster

        xgb.train = wrapped
        restore.append(lambda: setattr(xgb, "train", original))
    elif family == "lightgbm":
        import lightgbm as lgb

        original = lgb.train

        def wrapped(*args, **kwargs):
            booster = original(*args, **kwargs)
            store["model"] = booster
            return booster

        lgb.train = wrapped
        restore.append(lambda: setattr(lgb, "train", original))
    elif family == "catboost":
        import catboost

        for cls in (catboost.CatBoostRegressor, catboost.CatBoostClassifier):
            original = cls.fit

            def make(original):
                def wrapped(self, *args, **kwargs):
                    out = original(self, *args, **kwargs)
                    store["model"] = self
                    return out

                return wrapped

            cls.fit = make(original)
            restore.append(lambda cls=cls, original=original: setattr(cls, "fit", original))
    try:
        yield store
    finally:
        for undo in restore:
            undo()


class FittedModel:
    """A fitted V3 model with immutable ordered feature columns."""

    def __init__(self, *, family: str, task: str, config: Mapping[str, object], feature_columns: Sequence[str],
                 n_estimators: int, model: object = None, linear: Optional[Mapping[str, object]] = None):
        self.family = family
        self.task = task
        self.config = dict(config)
        self.feature_columns = tuple(str(c) for c in feature_columns)
        self.n_estimators = int(n_estimators)
        self._model = model
        self._linear = None if linear is None else {k: (np.asarray(v, dtype=np.float64) if isinstance(v, (list, tuple)) else v) for k, v in linear.items()}
        self.classification = M.is_classification(task)

    # ---- prediction ----------------------------------------------------
    def predict(self, X, feature_names: Optional[Sequence[str]] = None) -> np.ndarray:
        if feature_names is not None and tuple(str(n) for n in feature_names) != self.feature_columns:
            raise ColumnMismatchError("input feature names/order differ from the model's stored feature_columns")
        X = np.asarray(X)
        if X.ndim != 2 or X.shape[1] != len(self.feature_columns):
            raise ColumnMismatchError(f"expected {len(self.feature_columns)} columns, got shape {X.shape}")
        if self.family == "xgboost":
            import xgboost as xgb

            pred = self._model.predict(xgb.DMatrix(np.asarray(X, dtype=np.float32)), iteration_range=(0, self.n_estimators))
        elif self.family == "lightgbm":
            pred = self._model.predict(np.asarray(X, dtype=np.float64), num_iteration=self.n_estimators)
        elif self.family == "catboost":
            x = np.asarray(X, dtype=np.float64)
            pred = self._model.predict_proba(x)[:, 1] if self.classification else self._model.predict(x)
        else:
            pred = self._predict_linear(np.asarray(X, dtype=np.float64))
        pred = np.asarray(pred, dtype=np.float64)
        if not np.all(np.isfinite(pred)):
            raise FinalModelError("model produced non-finite predictions")
        return pred

    def _predict_linear(self, X: np.ndarray) -> np.ndarray:
        p = self._linear
        filled = np.where(np.isnan(X), p["median"], X)
        z = np.nan_to_num((filled - p["mean"]) / p["scale"])
        raw = z @ p["coef"] + float(p["intercept"])
        if self.classification:
            return 1.0 / (1.0 + np.exp(-raw))
        return raw

    # ---- persistence ---------------------------------------------------
    def save(self, directory) -> Dict[str, object]:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        files: List[Dict[str, str]] = []
        if self.family == "xgboost":
            path = directory / "model.json"
            self._model.save_model(str(path))
        elif self.family == "lightgbm":
            path = directory / "model.txt"
            self._model.save_model(str(path), num_iteration=self.n_estimators)
        elif self.family == "catboost":
            path = directory / "model.cbm"
            self._model.save_model(str(path))
        else:
            path = directory / "linear_params.json"
            payload = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in self._linear.items()}
            path.write_bytes(_canonical_json(payload))
        file_hash = _sha256_file(path)
        files.append({"path": path.name, "sha256": file_hash})
        meta = {
            "artifact_version": FINAL_MODELS_VERSION,
            "family": self.family,
            "task": self.task,
            "config": _jsonable(self.config),
            "n_estimators": self.n_estimators,
            "feature_columns": list(self.feature_columns),
            "names_sha256": names_sha256(self.feature_columns),
            "files": files,
            "model_sha256": file_hash,
        }
        (directory / META_NAME).write_bytes(_canonical_json(meta))
        return meta


def _jsonable(value):
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def fit_final(family: str, config: Mapping[str, object], X, y, w, *, task: str, feature_columns: Sequence[str],
              n_estimators: Optional[int] = None, nthread: int = 8, seed: int = 0) -> FittedModel:
    """Fit on all supplied rows with a FIXED tree budget (no early stopping)."""

    if family not in M.FAMILIES:
        raise FinalModelError("unknown family: " + repr(family))
    classification = M.is_classification(task)
    if classification and family == "ridge":
        raise FinalModelError("ridge is a regression family; use logistic for organ_support")
    if not classification and family == "logistic":
        raise FinalModelError("logistic is a classification family; use ridge for regression tasks")
    X = np.asarray(X)
    y_arr = np.asarray(y, dtype=np.float64)
    if X.ndim != 2 or len(X) != len(y_arr) or len(y_arr) == 0:
        raise FinalModelError("X and y must be aligned and non-empty")
    if X.shape[1] != len(feature_columns):
        raise FinalModelError("feature_columns length must equal X.shape[1]")
    if classification and not set(np.unique(y_arr)).issubset({0.0, 1.0}):
        raise FinalModelError("organ_support labels must be binary")
    cfg = dict(config)
    weights = M._effective_weights(cfg, w)
    if family in TREE_FAMILIES:
        n = int(n_estimators if n_estimators is not None else cfg.get("n_estimators", M.N_ESTIMATORS_CAP))
        n = int(min(n, M.N_ESTIMATORS_CAP))
        cfg["n_estimators"] = n
        helper = {"xgboost": M._fit_xgboost, "lightgbm": M._fit_lightgbm, "catboost": M._fit_catboost}[family]
        with _capture(family) as store:
            helper(cfg, X, y_arr, weights, X[:1], classification, None, nthread, seed)
        if "model" not in store:
            raise FinalModelError("failed to capture the fitted model object")
        return FittedModel(family=family, task=task, config=cfg, feature_columns=feature_columns, n_estimators=n, model=store["model"])
    params = _fit_linear_params(family, cfg, X, y_arr, weights, classification, seed)
    return FittedModel(family=family, task=task, config=cfg, feature_columns=feature_columns, n_estimators=0, linear=params)


def _fit_linear_params(family, config, X, y, w, classification, seed) -> Dict[str, object]:
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.preprocessing import StandardScaler

    X = np.asarray(X, dtype=np.float64)
    imputer = SimpleImputer(strategy="median", keep_empty_features=True).fit(X)
    scaler = StandardScaler().fit(imputer.transform(X))
    a = np.nan_to_num(scaler.transform(imputer.transform(X)))
    if classification:
        class_weight = "balanced" if config.get("class_weight", "none") == "scale_pos_weight" else None
        model = LogisticRegression(C=float(config.get("C", 1.0)), class_weight=class_weight, max_iter=2000, solver="lbfgs", random_state=int(seed))
        model.fit(a, y.astype(int), sample_weight=w)
        coef = model.coef_.reshape(-1)
        intercept = float(model.intercept_.reshape(-1)[0])
    else:
        model = Ridge(alpha=float(config.get("alpha", 1.0)))
        model.fit(a, y, sample_weight=w)
        coef = np.asarray(model.coef_, dtype=np.float64).reshape(-1)
        intercept = float(np.asarray(model.intercept_).reshape(-1)[0])
    median = np.nan_to_num(np.asarray(imputer.statistics_, dtype=np.float64), nan=0.0)
    return {"median": median, "mean": np.asarray(scaler.mean_, dtype=np.float64), "scale": np.asarray(scaler.scale_, dtype=np.float64),
            "coef": coef, "intercept": intercept}


def load_fitted_model(meta_path) -> FittedModel:
    meta_path = Path(meta_path)
    if meta_path.is_dir():
        meta_path = meta_path / META_NAME
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    base = meta_path.parent
    for entry in meta["files"]:
        actual = _sha256_file(base / entry["path"])
        if actual != entry["sha256"]:
            raise HashMismatchError(f"{entry['path']}: sha256 mismatch (file modified after save)")
    if meta["names_sha256"] != names_sha256(meta["feature_columns"]):
        raise HashMismatchError("feature_columns do not match the recorded names_sha256")
    if meta["model_sha256"] != meta["files"][0]["sha256"]:
        raise HashMismatchError("model_sha256 does not match the model file entry")
    family, task = meta["family"], meta["task"]
    n = int(meta["n_estimators"])
    common = dict(family=family, task=task, config=meta["config"], feature_columns=meta["feature_columns"], n_estimators=n)
    path = base / meta["files"][0]["path"]
    if family == "xgboost":
        import xgboost as xgb

        booster = xgb.Booster()
        booster.load_model(str(path))
        return FittedModel(model=booster, **common)
    if family == "lightgbm":
        import lightgbm as lgb

        return FittedModel(model=lgb.Booster(model_file=str(path)), **common)
    if family == "catboost":
        from catboost import CatBoostClassifier, CatBoostRegressor

        model = CatBoostClassifier() if M.is_classification(task) else CatBoostRegressor()
        model.load_model(str(path))
        return FittedModel(model=model, **common)
    payload = json.loads(path.read_text(encoding="utf-8"))
    return FittedModel(linear=payload, **common)
