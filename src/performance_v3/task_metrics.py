"""Stay-balanced task metrics for V3 (thin wrappers over evaluation.metrics).

All metrics use 1/n_i stay-balanced weights from
evaluation.weights.compute_stay_weights. Fast vectorised AUROC/AUPRC are
provided for bootstrap loops; they are parity-tested against the frozen
Phase-9 estimators in evaluation.metrics.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, Mapping, Optional, Sequence

import numpy as np

from evaluation.metrics import (
    PredictionRecord,
    evaluate_icu_time,
    evaluate_organ_support,
    evaluate_recovery_horizon,
)
from evaluation.weighted_stats import weighted_median
from evaluation.weights import compute_stay_weights
from performance_v3.models import postprocess_icu

TASK_METRICS_VERSION = "performance_v3_task_metrics_v1"
PRIMARY_METRIC_NAME = {
    "recovery24": "mae",
    "recovery48": "mae",
    "icu_stay_time": "weighted_median_ae_hours",
    "organ_support": "auprc",
}
HIGHER_IS_BETTER = {"recovery24": False, "recovery48": False, "icu_stay_time": False, "organ_support": True}
_EPS = 1e-12


def make_records(y, pred, stay_id, prediction_time=None, eligible=None):
    """PredictionRecords; synthesises a unique prediction_time per stay row if absent."""

    y = np.asarray(y, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    n = len(y)
    if not (len(pred) == len(stay_id) == n):
        raise ValueError("y, pred and stay_id must have equal length")
    if prediction_time is None:
        counter: Dict[object, int] = {}
        times = []
        for sid in stay_id:
            counter[sid] = counter.get(sid, 0) + 1
            times.append("row-%06d" % counter[sid])
    else:
        times = list(prediction_time)
    elig = [True] * n if eligible is None else [bool(e) for e in eligible]
    return tuple(
        PredictionRecord(stay_id=s, prediction_time=str(t), target=float(a), prediction=float(b), eligible=e)
        for s, t, a, b, e in zip(stay_id, times, y, pred, elig)
    )


def stay_weights(stay_id) -> np.ndarray:
    result = compute_stay_weights(list(stay_id), [True] * len(stay_id))
    return np.asarray(result.weights, dtype=np.float64)


def fast_weighted_auroc(y, score, w) -> float:
    """Weighted AUROC with half credit for ties (same convention as
    evaluation.metrics._weighted_auroc) in O(n log n)."""

    y = np.asarray(y); score = np.asarray(score, dtype=np.float64); w = np.asarray(w, dtype=np.float64)
    pos_w = float(np.sum(w[y == 1])); neg_w = float(np.sum(w[y == 0]))
    if pos_w <= 0 or neg_w <= 0:
        return math.nan
    _, inverse = np.unique(score, return_inverse=True)
    k = inverse.max() + 1
    wp = np.bincount(inverse, weights=w * (y == 1), minlength=k)
    wn = np.bincount(inverse, weights=w * (y == 0), minlength=k)
    neg_below = np.cumsum(wn) - wn
    return float(np.sum(wp * (neg_below + 0.5 * wn)) / (pos_w * neg_w))


def fast_weighted_auprc(y, score, w) -> float:
    """Weighted average precision, step integral (same convention as
    evaluation.metrics._weighted_average_precision)."""

    y = np.asarray(y); score = np.asarray(score, dtype=np.float64); w = np.asarray(w, dtype=np.float64)
    pos_w = float(np.sum(w[y == 1]))
    if pos_w <= 0:
        return math.nan
    _, inverse = np.unique(-score, return_inverse=True)  # descending score groups
    k = inverse.max() + 1
    gp = np.bincount(inverse, weights=w * (y == 1), minlength=k)
    gt = np.bincount(inverse, weights=w, minlength=k)
    cum_p = np.cumsum(gp)
    cum_t = np.cumsum(gt)
    recall = cum_p / pos_w
    precision = cum_p / cum_t
    prev = np.concatenate([[0.0], recall[:-1]])
    return float(np.sum((recall - prev) * precision))


def weighted_log_loss(y, p, w) -> float:
    y = np.asarray(y, dtype=np.float64); p = np.clip(np.asarray(p, dtype=np.float64), _EPS, 1 - _EPS); w = np.asarray(w, dtype=np.float64)
    return float(np.sum(w * -(y * np.log(p) + (1 - y) * np.log(1 - p))) / np.sum(w))


def recovery_metrics(y, pred, stay_id, prediction_time=None, horizon: str = "24h") -> Dict[str, float]:
    records = make_records(y, pred, stay_id, prediction_time)
    result = evaluate_recovery_horizon(records, horizon=horizon)
    w = stay_weights(stay_id)
    abs_err = np.abs(np.asarray(pred, dtype=np.float64) - np.asarray(y, dtype=np.float64))
    return {
        "mae": float(result.metrics["mae"]),
        "rmse": float(result.metrics["rmse"]),
        "median_ae": float(weighted_median(tuple(abs_err), tuple(w))),
        "n_examples": int(result.counts["N_examples"]),
        "n_stays": int(result.counts["N_ICU_stays"]),
    }


def icu_metrics(y_hours, pred_raw_log1p, stay_id, prediction_time=None) -> Dict[str, float]:
    """`pred_raw_log1p` is the raw model output; frozen postprocess
    expm1(max(.,0)) is applied here, evaluation is in hours."""

    hours = postprocess_icu(pred_raw_log1p)
    records = make_records(y_hours, hours, stay_id, prediction_time)
    result = evaluate_icu_time(records)
    return {
        "weighted_median_ae_hours": float(result.metrics["median_absolute_error"]),
        "mae": float(result.metrics["mae"]),
        "rmse": float(result.metrics["rmse"]),
        "n_examples": int(result.counts["N_examples"]),
        "n_stays": int(result.counts["N_ICU_stays"]),
    }


def support_metrics(y, score, stay_id, prediction_time=None, *, threshold: Optional[float] = None, probability_type: str = "raw") -> Dict[str, object]:
    records = make_records(y, score, stay_id, prediction_time)
    kwargs = {}
    if threshold is not None:
        kwargs = {"fixed_threshold": float(threshold), "threshold_identifier": "v3_supplied_threshold"}
    result = evaluate_organ_support(records, probability_type=probability_type, **kwargs)
    w = stay_weights(stay_id)
    out: Dict[str, object] = {
        "auprc": float(result.metrics["auprc"]),
        "auroc": float(result.metrics["auroc"]),
        "brier": float(result.metrics["brier"]),
        "log_loss": weighted_log_loss(y, score, w),
        "n_examples": int(result.counts["N_examples"]),
        "n_stays": int(result.counts["N_ICU_stays"]),
        "n_positive_examples": int(result.counts["N_positive_examples"]),
        "n_positive_stays": int(result.counts["N_positive_stays"]),
    }
    if threshold is not None:
        m = result.metrics
        out.update(
            {
                "threshold": float(threshold),
                "precision": float(m["precision"]),
                "recall": float(m["sensitivity"]),
                "specificity": float(m["specificity"]),
                "f1": float(m["f1"]),
                "confusion": {k: float(m["weighted_" + k]) for k in ("TP", "TN", "FP", "FN")},
            }
        )
    return out


def primary_metric(task: str, y, pred, stay_id, prediction_time=None) -> float:
    """The task's development primary metric (support: raw-score AUPRC)."""

    if task in ("recovery24", "recovery48"):
        horizon = "24h" if task == "recovery24" else "48h"
        return recovery_metrics(y, pred, stay_id, prediction_time, horizon=horizon)["mae"]
    if task == "icu_stay_time":
        return icu_metrics(y, pred, stay_id, prediction_time)["weighted_median_ae_hours"]
    if task == "organ_support":
        return support_metrics(y, pred, stay_id, prediction_time)["auprc"]
    raise ValueError("unknown task: " + repr(task))


def is_better(task: str, a: float, b: float) -> bool:
    """True if metric value `a` is strictly better than `b` for `task`."""

    if task not in HIGHER_IS_BETTER:
        raise ValueError("unknown task: " + repr(task))
    return a > b if HIGHER_IS_BETTER[task] else a < b


def record_metric_fn(task: str, name: Optional[str] = None, *, threshold: Optional[float] = None) -> Callable[[Sequence[PredictionRecord]], Optional[float]]:
    """Fast bootstrap-compatible estimator over PredictionRecords.

    Recovery: mae|rmse|median_ae (raw deltas). ICU: records must carry
    HOURS: weighted_median_ae_hours|mae|rmse. Support: auprc|auroc|brier|
    log_loss|f1|precision|recall|specificity (threshold needed for the last
    four). Uses stay weights recomputed per replicate (each bootstrap
    cluster id is its own stay)."""

    name = name or PRIMARY_METRIC_NAME[task]

    def fn(records):
        eligible = [r for r in records if r.eligible]
        if not eligible:
            return None
        y = np.asarray([r.target for r in eligible], dtype=np.float64)
        p = np.asarray([r.prediction for r in eligible], dtype=np.float64)
        w = stay_weights([r.stay_id for r in eligible])
        if task in ("recovery24", "recovery48", "icu_stay_time"):
            err = np.abs(p - y)
            if name == "mae":
                return float(np.sum(w * err) / np.sum(w))
            if name == "rmse":
                return float(math.sqrt(np.sum(w * (p - y) ** 2) / np.sum(w)))
            if name in ("median_ae", "weighted_median_ae_hours"):
                return float(weighted_median(tuple(err), tuple(w)))
            raise ValueError("unknown metric: " + name)
        if name == "auprc":
            return fast_weighted_auprc(y, p, w)
        if name == "auroc":
            return fast_weighted_auroc(y, p, w)
        if name == "brier":
            return float(np.sum(w * (p - y) ** 2) / np.sum(w))
        if name == "log_loss":
            return weighted_log_loss(y, p, w)
        if name in ("f1", "precision", "recall", "specificity"):
            if threshold is None:
                raise ValueError(name + " requires a threshold")
            pred_pos = p >= threshold
            tp = float(np.sum(w[(y == 1) & pred_pos])); fp = float(np.sum(w[(y == 0) & pred_pos]))
            fn_ = float(np.sum(w[(y == 1) & ~pred_pos])); tn = float(np.sum(w[(y == 0) & ~pred_pos]))
            if name == "precision":
                return tp / (tp + fp) if tp + fp > 0 else math.nan
            if name == "recall":
                return tp / (tp + fn_) if tp + fn_ > 0 else math.nan
            if name == "specificity":
                return tn / (tn + fp) if tn + fp > 0 else math.nan
            pr = tp / (tp + fp) if tp + fp > 0 else math.nan
            rc = tp / (tp + fn_) if tp + fn_ > 0 else math.nan
            return 2 * pr * rc / (pr + rc) if math.isfinite(pr) and math.isfinite(rc) and pr + rc > 0 else math.nan
        raise ValueError("unknown metric: " + name)

    return fn
