"""Support-probability calibrators, nested cross-fit comparison, selection
and threshold rules for V3 (development only).

Methods: none, platt, isotonic, beta. Frozen calibrators are plain
JSON-serialisable dicts. Isotonic application reuses
performance_v2.final_evaluation.isotonic_apply (np.interp over knots).
"""

from __future__ import annotations

import math
from typing import Dict, Mapping, Optional, Sequence

import numpy as np

from performance_v2.final_evaluation import isotonic_apply
from performance_v3.task_metrics import fast_weighted_auprc, fast_weighted_auroc, weighted_log_loss

CALIBRATION_VERSION = "performance_v3_calibration_v1"
METHODS = ("none", "platt", "isotonic", "beta")
DEFAULT_ORDER = ("none", "platt", "beta", "isotonic")
EPS = 1e-6


class CalibrationError(ValueError):
    """Raised for invalid calibration inputs or leakage-prone fold structure."""


def _logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), EPS, 1 - EPS)
    return np.log(p) - np.log1p(-p)


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))


def _weights(w, n):
    if w is None:
        return np.ones(n, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    if len(w) != n or np.any(w < 0) or not np.all(np.isfinite(w)):
        raise CalibrationError("weights must be finite, nonnegative and aligned")
    return w


def _logit_fit(features: np.ndarray, y: np.ndarray, w: np.ndarray):
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
    model.fit(features, y.astype(int), sample_weight=w)
    return model.coef_.ravel().astype(float), float(model.intercept_[0])


def fit_calibrator(method: str, scores, y, w=None) -> Dict[str, object]:
    """Fit on (score, label, weight) -> frozen JSON-serialisable dict."""

    if method not in METHODS:
        raise CalibrationError("unknown calibration method: " + repr(method))
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(s) != len(y) or len(s) == 0:
        raise CalibrationError("scores and labels must be equal nonzero length")
    if not set(np.unique(y)).issubset({0.0, 1.0}):
        raise CalibrationError("labels must be binary")
    w = _weights(w, len(s))
    frozen: Dict[str, object] = {"method": method, "version": CALIBRATION_VERSION}
    if method == "none":
        return frozen
    if len(np.unique(y)) < 2:
        raise CalibrationError("calibration needs both classes")
    if method == "platt":
        coef, intercept = _logit_fit(_logit(s).reshape(-1, 1), y, w)
        frozen.update({"slope": coef[0], "intercept": intercept})
        if frozen["slope"] < 0:  # keep monotone non-decreasing
            frozen.update({"slope": 0.0, "intercept": float(math.log(max(np.average(y, weights=w), EPS) / max(1 - np.average(y, weights=w), EPS)))})
    elif method == "beta":
        p = np.clip(s, EPS, 1 - EPS)
        features = np.column_stack([np.log(p), -np.log1p(-p)])
        coef, intercept = _logit_fit(features, y, w)
        a, b = float(coef[0]), float(coef[1])
        if a < 0 or b < 0:  # Kull et al.: constrain a,b >= 0 for monotonicity
            keep = [i for i, c in enumerate((a, b)) if c >= 0]
            if keep:
                sub, intercept = _logit_fit(features[:, keep], y, w)
                a = b = 0.0
                for column, value in zip(keep, sub):
                    if column == 0:
                        a = max(float(value), 0.0)
                    else:
                        b = max(float(value), 0.0)
            else:
                prior = float(np.average(y, weights=w))
                a = b = 0.0
                intercept = float(math.log(max(prior, EPS) / max(1 - prior, EPS)))
        frozen.update({"a": a, "b": b, "intercept": float(intercept)})
    else:  # isotonic
        from sklearn.isotonic import IsotonicRegression

        iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        iso.fit(s, y, sample_weight=w)
        frozen.update(
            {"x_thresholds": iso.X_thresholds_.astype(float).tolist(), "y_thresholds": iso.y_thresholds_.astype(float).tolist()}
        )
    return frozen


def apply_calibrator(frozen: Mapping[str, object], scores) -> np.ndarray:
    method = frozen["method"]
    s = np.asarray(scores, dtype=np.float64)
    if method == "none":
        return np.clip(s, 0.0, 1.0)
    if method == "platt":
        return _sigmoid(float(frozen["slope"]) * _logit(s) + float(frozen["intercept"]))
    if method == "beta":
        p = np.clip(s, EPS, 1 - EPS)
        return _sigmoid(float(frozen["a"]) * np.log(p) + float(frozen["b"]) * (-np.log1p(-p)) + float(frozen["intercept"]))
    if method == "isotonic":
        return isotonic_apply(s, frozen)
    raise CalibrationError("unknown calibration method: " + repr(method))


fit = fit_calibrator
apply = apply_calibrator


def expected_calibration_error(p, y, w, n_bins: int = 10) -> float:
    """ECE over equal-mass (weight-quantile) bins."""

    p = np.asarray(p, dtype=np.float64); y = np.asarray(y, dtype=np.float64); w = np.asarray(w, dtype=np.float64)
    order = np.argsort(p, kind="mergesort")
    p, y, w = p[order], y[order], w[order]
    total = float(np.sum(w))
    cum = np.cumsum(w) - 0.5 * w
    bins = np.minimum((cum / total * n_bins).astype(int), n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        mask = bins == b
        wb = float(np.sum(w[mask]))
        if wb <= 0:
            continue
        ece += wb / total * abs(float(np.average(y[mask], weights=w[mask])) - float(np.average(p[mask], weights=w[mask])))
    return float(ece)


def calibration_metrics(p, y, w=None) -> Dict[str, float]:
    p = np.asarray(p, dtype=np.float64); y = np.asarray(y, dtype=np.float64)
    w = _weights(w, len(p))
    metrics = {
        "brier": float(np.sum(w * (p - y) ** 2) / np.sum(w)),
        "log_loss": weighted_log_loss(y, p, w),
        "ece_10_equal_mass": expected_calibration_error(p, y, w),
        "auroc": fast_weighted_auroc(y, p, w),
        "auprc": fast_weighted_auprc(y, p, w),
        "tie_fraction": float(1.0 - len(np.unique(p)) / len(p)),
    }
    if len(np.unique(y)) == 2:
        coef, intercept = _logit_fit(_logit(p).reshape(-1, 1), y, w)
        metrics["calibration_slope"] = float(coef[0])
        metrics["calibration_intercept"] = float(intercept)
    else:
        metrics["calibration_slope"] = math.nan
        metrics["calibration_intercept"] = math.nan
    return metrics


def nested_crossfit_calibration(oof_scores, y, w, subject_ids, folds, methods: Sequence[str] = METHODS):
    """Held-out calibrated probabilities per method. For each fold k the
    calibrator is fitted on rows with fold != k and applied to fold k.
    Raises if any subject spans folds (a stay's rows must stay together).

    Returns {method: {"heldout_prob": ndarray, "metrics": {...}}} plus
    "_audit": per-fold fit/eval sizes and overlap counts (always 0).
    """

    s = np.asarray(oof_scores, dtype=np.float64); y = np.asarray(y, dtype=np.float64)
    folds = np.asarray(folds); subject_ids = np.asarray(subject_ids)
    n = len(s)
    if not (len(y) == len(folds) == len(subject_ids) == n):
        raise CalibrationError("inputs must be aligned")
    w = _weights(w, n)
    by_subject: Dict[object, set] = {}
    for sid, fold in zip(subject_ids.tolist(), folds.tolist()):
        by_subject.setdefault(sid, set()).add(fold)
    if any(len(v) > 1 for v in by_subject.values()):
        raise CalibrationError("a subject spans multiple folds; folds must be subject-grouped")
    unique_folds = sorted(set(folds.tolist()))
    if len(unique_folds) < 2:
        raise CalibrationError("nested cross-fitting needs at least two folds")
    out: Dict[str, object] = {}
    audit = []
    for fold in unique_folds:
        fit_mask = folds != fold
        eval_mask = ~fit_mask
        audit.append(
            {
                "fold": fold,
                "n_fit": int(fit_mask.sum()),
                "n_eval": int(eval_mask.sum()),
                "row_overlap": int(np.sum(fit_mask & eval_mask)),
                "subject_overlap": len(set(subject_ids[fit_mask].tolist()) & set(subject_ids[eval_mask].tolist())),
            }
        )
    for method in methods:
        held = np.full(n, np.nan)
        for fold in unique_folds:
            fit_mask = folds != fold
            frozen = fit_calibrator(method, s[fit_mask], y[fit_mask], w[fit_mask])
            held[~fit_mask] = apply_calibrator(frozen, s[~fit_mask])
        out[method] = {"heldout_prob": held, "metrics": calibration_metrics(held, y, w)}
    out["_audit"] = audit
    return out


def select_calibrator(results: Mapping[str, Mapping[str, object]], tie_tolerance: float = 0.002, order: Sequence[str] = DEFAULT_ORDER, metric: str = "log_loss") -> str:
    """Lowest `metric`; any method within `tie_tolerance` of the best is a
    tie, resolved by `order` (simpler first)."""

    scores = {m: float(results[m]["metrics"][metric]) for m in order if m in results}
    if not scores:
        raise CalibrationError("no candidate calibrators to select from")
    best = min(scores.values())
    for method in order:
        if method in scores and scores[method] <= best + tie_tolerance:
            return method
    raise CalibrationError("unreachable")


def _weighted_f1(labels, predictions, weights) -> float:
    """Same rule as scripts/performance_v2_phase3_part7_8_calibration_threshold.py."""

    labels = np.asarray(labels); predictions = np.asarray(predictions); weights = np.asarray(weights)
    tp = float(np.sum(weights[(labels == 1) & (predictions == 1)]))
    fp = float(np.sum(weights[(labels == 0) & (predictions == 1)]))
    fn = float(np.sum(weights[(labels == 1) & (predictions == 0)]))
    if tp + fp == 0 or tp + fn == 0:
        return 0.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def select_threshold_weighted_f1(prob, y, w=None) -> Dict[str, object]:
    """Maximise stay-balanced F1 over sorted unique probabilities plus
    {0, 1} (V2 candidate set); `p >= t` is positive. Float ties within 1e-15
    resolve to the LOWEST threshold (V3 protocol; V2 preferred the value
    closest to 0.5)."""

    p = np.asarray(prob, dtype=np.float64); y = np.asarray(y)
    w = _weights(w, len(p))
    candidates = np.unique(np.concatenate([p, [0.0, 1.0]]))
    uniq = np.unique(p)
    pos_w = np.bincount(np.searchsorted(uniq, p), weights=w * (y == 1), minlength=len(uniq))
    neg_w = np.bincount(np.searchsorted(uniq, p), weights=w * (y == 0), minlength=len(uniq))
    suffix_pos = np.concatenate([np.cumsum(pos_w[::-1])[::-1], [0.0]])
    suffix_neg = np.concatenate([np.cumsum(neg_w[::-1])[::-1], [0.0]])
    total_pos = float(pos_w.sum())
    best_t, best_f1 = None, -1.0
    for t in candidates:  # ascending => first max kept == lowest threshold
        idx = int(np.searchsorted(uniq, t, side="left"))
        tp, fp = float(suffix_pos[idx]), float(suffix_neg[idx])
        fn = total_pos - tp
        if tp + fp == 0 or tp + fn == 0 or tp == 0:
            f1 = 0.0
        else:
            precision, recall = tp / (tp + fp), tp / (tp + fn)
            f1 = 2 * precision * recall / (precision + recall)
        if f1 > best_f1 + 1e-15:
            best_t, best_f1 = float(t), f1
    pred = (p >= best_t).astype(int)
    return {
        "threshold": best_t,
        "f1": best_f1,
        "weighted_confusion": {
            "TP": float(np.sum(w[(y == 1) & (pred == 1)])),
            "FP": float(np.sum(w[(y == 0) & (pred == 1)])),
            "TN": float(np.sum(w[(y == 0) & (pred == 0)])),
            "FN": float(np.sum(w[(y == 1) & (pred == 0)])),
        },
        "comparator": ">=",
        "tie_break": "lowest threshold",
    }
