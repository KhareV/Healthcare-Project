"""Governed cross-fitted candidate search for V3 (development evidence only).

Everything is built from the V3 libraries: feature cache + variant columns
(features.py), uniform model adapter (models.py), stay-balanced metrics
(task_metrics.py) and subject-level folds (splits.py). Every trial is appended
to a JSONL ledger, including failures. Nothing here can read a sealed cohort:
callers supply a FeatureCache built from development rows only.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from performance_v3 import task_metrics as tm
from performance_v3.features import (
    TASK_COLUMNS, FeatureCache, canonical_task, task_mask, variant_columns, weights_stay_balanced,
)
from performance_v3.models import fit_predict, is_classification, postprocess_icu
from performance_v3.splits import V3_CROSSFIT_VERSION, clone_safe_folds, crossfit_fold_for_subject

SEARCH_VERSION = "performance_v3_search_v1"
TASKS = ("recovery24", "recovery48", "icu_stay_time", "organ_support")
LOWER_IS_BETTER = {"recovery24": True, "recovery48": True, "icu_stay_time": True, "organ_support": False}
PRIMARY_NAME = {"recovery24": "mae", "recovery48": "mae", "icu_stay_time": "weighted_median_ae_hours", "organ_support": "auprc"}
ES_FRACTION_MOD = 10  # inner early-stopping split = 1/10 of TRAINING subjects


@dataclass
class TaskData:
    task: str
    variant: str
    X: np.ndarray            # float32 [n, d] (eligible rows, variant columns)
    names: Tuple[str, ...]
    y: np.ndarray            # training target (ICU: log1p hours)
    y_eval: np.ndarray       # evaluation target (ICU: hours)
    w: np.ndarray            # stay-balanced 1/n_i weights
    stay_id: np.ndarray
    subject_id: np.ndarray
    prediction_time: np.ndarray
    fold: np.ndarray         # int outer fold per row
    n_folds: int

    @property
    def n_rows(self) -> int:
        return int(self.X.shape[0])

    @property
    def n_stays(self) -> int:
        return int(len(np.unique(self.stay_id)))


def assign_folds(subject_ids: Iterable[str], *, n_folds: int = 5, version: str = V3_CROSSFIT_VERSION,
                 clone_groups: Optional[Mapping[str, object]] = None) -> Dict[str, int]:
    ids = sorted(set(str(s) for s in subject_ids))
    if clone_groups:
        return clone_safe_folds(ids, clone_groups, n_folds, version)
    return {s: crossfit_fold_for_subject(s, n_folds, version) for s in ids}


def prepare_task_data(cache: FeatureCache, task: str, variant: str, *, include_s: bool = False,
                      n_folds: int = 5, version: str = V3_CROSSFIT_VERSION,
                      clone_groups: Optional[Mapping[str, object]] = None) -> TaskData:
    if task not in TASKS:
        raise ValueError("unknown task " + repr(task))
    feature_task = canonical_task(task)
    mask = task_mask(cache.meta, feature_task)
    weights = weights_stay_balanced(cache.meta, feature_task)[mask]
    columns = variant_columns(cache.names, variant, feature_task, include_s=include_s)
    X = np.ascontiguousarray(cache.matrix[mask][:, columns], dtype=np.float32)
    names = tuple(n for n, keep in zip(cache.names, columns) if keep)
    target_column = TASK_COLUMNS[feature_task][1]
    y = np.asarray(cache.meta[target_column][mask], dtype=np.float64)
    y_eval = np.expm1(y) if task == "icu_stay_time" else y
    subjects = np.asarray([str(s) for s in cache.meta["subject_id"][mask]])
    folds_by_subject = assign_folds(subjects, n_folds=n_folds, version=version, clone_groups=clone_groups)
    fold = np.asarray([folds_by_subject[s] for s in subjects], dtype=np.int32)
    return TaskData(
        task=task, variant=variant, X=X, names=names, y=y, y_eval=y_eval, w=weights,
        stay_id=np.asarray([str(s) for s in cache.meta["stay_id"][mask]]), subject_id=subjects,
        prediction_time=np.asarray([str(s) for s in cache.meta["prediction_time"][mask]]), fold=fold, n_folds=n_folds,
    )


def fold_metrics(task: str, y_eval, raw_pred, stay_id, prediction_time) -> Dict[str, float]:
    """Stay-balanced fold metrics. Support uses the fast O(n log n) AUPRC/AUROC."""

    if task in ("recovery24", "recovery48"):
        out = tm.recovery_metrics(y_eval, raw_pred, stay_id, prediction_time, horizon="24h" if task == "recovery24" else "48h")
        return {k: float(out[k]) for k in ("mae", "rmse", "median_ae")}
    if task == "icu_stay_time":
        out = tm.icu_metrics(y_eval, raw_pred, stay_id, prediction_time)
        return {k: float(out[k]) for k in ("weighted_median_ae_hours", "mae", "rmse")}
    w = tm.stay_weights(stay_id)
    y = np.asarray(y_eval)
    p = np.clip(np.asarray(raw_pred, dtype=np.float64), 1e-12, 1 - 1e-12)
    return {
        "auprc": tm.fast_weighted_auprc(y, p, w),
        "auroc": tm.fast_weighted_auroc(y, p, w),
        "brier": float(np.sum(w * (p - y) ** 2) / np.sum(w)),
        "log_loss": tm.weighted_log_loss(y, p, w),
    }


def _es_mask(subjects: np.ndarray, outer_fold: int, version: str) -> np.ndarray:
    def pick(s: str) -> bool:
        digest = hashlib.sha256(f"{version}|early_stop|{outer_fold}|{s}".encode()).hexdigest()
        return int(digest, 16) % ES_FRACTION_MOD == 0
    unique = {s: pick(s) for s in np.unique(subjects)}
    return np.asarray([unique[s] for s in subjects], dtype=bool)


def run_cv_trial(td: TaskData, family: str, config: Mapping[str, object], *, folds: Sequence[int], seed: int,
                 nthread: int = 8, version: str = V3_CROSSFIT_VERSION, keep_oof: bool = False) -> Dict[str, object]:
    """Cross-fitted trial over the requested outer folds; returns metrics + optional OOF raw predictions."""

    per_fold: List[Dict[str, float]] = []
    best_iters: List[int] = []
    fit_seconds = 0.0
    oof = np.full(td.n_rows, np.nan) if keep_oof else None
    for k in folds:
        ev = td.fold == k
        tr = ~ev
        if not ev.any() or not tr.any():
            raise ValueError(f"fold {k} empty")
        es = None
        tr_fit = tr
        if family in ("xgboost", "lightgbm", "catboost") and config.get("use_early_stopping", True):
            es_rows = tr & _es_mask(td.subject_id, k, version)
            tr_fit = tr & ~es_rows
            if es_rows.any() and tr_fit.any():
                es = (td.X[es_rows], td.y[es_rows], td.w[es_rows])
            else:
                tr_fit = tr
        pred, info = fit_predict(family, config, td.X[tr_fit], td.y[tr_fit], td.w[tr_fit], td.X[ev], task=td.task, es=es, nthread=nthread, seed=seed)
        fit_seconds += float(info["fit_seconds"])
        best_iters.append(int(info["best_iteration"]))
        per_fold.append(fold_metrics(td.task, td.y_eval[ev], pred, td.stay_id[ev], td.prediction_time[ev]))
        if keep_oof:
            oof[ev] = pred
    primary = PRIMARY_NAME[td.task]
    values = np.asarray([m[primary] for m in per_fold], dtype=np.float64)
    secondary = {k: float(np.mean([m[k] for m in per_fold])) for k in per_fold[0]}
    result = {
        "primary_metric": primary,
        "primary_mean": float(values.mean()),
        "primary_sd": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        "per_fold_primary": [float(v) for v in values],
        "secondary_means": secondary,
        "best_iterations": best_iters,
        "fit_seconds": fit_seconds,
        "folds": [int(k) for k in folds],
    }
    if keep_oof:
        result["oof"] = oof
    return result


def better(task: str, a: float, b: float) -> bool:
    return a < b if LOWER_IS_BETTER[task] else a > b


class TrialLedger:
    """Append-only JSONL trial ledger with resume support (failed trials are kept)."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._done: Dict[str, Mapping[str, object]] = {}
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    record = json.loads(line)
                    self._done[record["trial_id"]] = record

    def has(self, trial_id: str) -> bool:
        return trial_id in self._done

    def get(self, trial_id: str) -> Optional[Mapping[str, object]]:
        return self._done.get(trial_id)

    def append(self, record: Mapping[str, object]) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True, default=_json_default) + "\n")
        self._done[str(record["trial_id"])] = record

    def records(self) -> List[Mapping[str, object]]:
        return list(self._done.values())


def _json_default(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value))


def trial_id_for(stage: str, task: str, family: str, variant: str, config_hash: str, seed: int, folds: Sequence[int]) -> str:
    return f"{stage}|{task}|{family}|{variant}|{config_hash}|s{seed}|f{''.join(str(k) for k in folds)}"


def run_trial_logged(ledger: TrialLedger, td: TaskData, stage: str, family: str, config: Mapping[str, object], *,
                     folds: Sequence[int], seed: int, nthread: int = 8) -> Mapping[str, object]:
    chash = str(config.get("config_hash") or hashlib.sha256(json.dumps(config, sort_keys=True, default=str).encode()).hexdigest()[:16])
    tid = trial_id_for(stage, td.task, family, td.variant, chash, seed, folds)
    existing = ledger.get(tid)
    if existing is not None and existing.get("status") == "ok":
        return existing
    started = time.time()
    record: Dict[str, object] = {
        "trial_id": tid, "stage": stage, "task": td.task, "family": family, "variant": td.variant,
        "config": dict(config), "config_hash": chash, "seed": seed, "n_rows": td.n_rows, "n_stays": td.n_stays,
        "n_features": int(td.X.shape[1]), "search_version": SEARCH_VERSION,
    }
    try:
        result = run_cv_trial(td, family, config, folds=folds, seed=seed, nthread=nthread)
        record.update(result)
        record["status"] = "ok"
    except Exception as error:  # failures are data: keep them
        record["status"] = "failed"
        record["error"] = f"{type(error).__name__}: {error}"
    record["wall_seconds"] = round(time.time() - started, 2)
    ledger.append(record)
    return record


def best_trial(records: Iterable[Mapping[str, object]], task: str, *, stage: Optional[str] = None,
               family: Optional[str] = None, variant: Optional[str] = None) -> Optional[Mapping[str, object]]:
    pool = [r for r in records if r.get("status") == "ok" and r["task"] == task
            and (stage is None or r["stage"] == stage) and (family is None or r["family"] == family)
            and (variant is None or r["variant"] == variant)]
    if not pool:
        return None
    sign = 1.0 if LOWER_IS_BETTER[task] else -1.0
    return min(pool, key=lambda r: (sign * float(r["primary_mean"]), float(r["primary_sd"])))
