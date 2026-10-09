"""Paired stay-cluster bootstrap for V2-vs-V3 comparison.

Both models are scored on the SAME stay resamples: the draws made while
bootstrapping model A are passed as `forced_draws` to the bootstrap of model
B (evaluation.bootstrap.grouped_bootstrap). A replicate is used only if the
metric is finite for BOTH models (joint skip). difference = B - A.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np

from evaluation.bootstrap import BootstrapError, grouped_bootstrap, percentile
from evaluation.metrics import PredictionRecord

PAIRED_BOOTSTRAP_VERSION = "performance_v3_paired_stay_cluster_bootstrap_v1"


class PairedBootstrapError(BootstrapError):
    """Raised when paired inputs violate the identical-row-set contract."""


def _eligible_keyed(records: Sequence[PredictionRecord]):
    keyed = {}
    order = []
    for record in records:
        if not record.eligible:
            continue
        key = (record.stay_id, record.prediction_time)
        if key in keyed:
            raise PairedBootstrapError("duplicate (stay_id, prediction_time) in paired input: " + repr(key))
        keyed[key] = record
        order.append(key)
    return keyed, order


def align_paired_records(records_a: Sequence[PredictionRecord], records_b: Sequence[PredictionRecord]) -> Tuple[Tuple[PredictionRecord, ...], Tuple[PredictionRecord, ...]]:
    """Eligible rows of A and B re-expressed in A's order; raises unless the
    (stay_id, prediction_time) sets are identical."""

    keyed_a, order_a = _eligible_keyed(records_a)
    keyed_b, _ = _eligible_keyed(records_b)
    if set(keyed_a) != set(keyed_b):
        only_a = len(set(keyed_a) - set(keyed_b))
        only_b = len(set(keyed_b) - set(keyed_a))
        raise PairedBootstrapError(
            "paired eligible row sets differ: %d only in A, %d only in B (denominators must be identical)" % (only_a, only_b)
        )
    if not order_a:
        raise PairedBootstrapError("no eligible rows")
    return tuple(keyed_a[k] for k in order_a), tuple(keyed_b[k] for k in order_a)


def _capturing(fn: Callable, sink: list):
    def wrapped(records):
        value = fn(records)
        sink.append(None if value is None else float(value))
        return value

    return wrapped


def _level_key(level: float) -> str:
    return "%g" % level


def paired_stay_cluster_bootstrap(
    records_a: Sequence[PredictionRecord],
    records_b: Sequence[PredictionRecord],
    metric_fn: Callable[[Sequence[PredictionRecord]], Optional[float]],
    *,
    n: int = 2000,
    seed: int,
    ci_levels: Sequence[float] = (0.95, 0.9875),
    metric_name: str = "metric",
    task: str = "task",
) -> Dict[str, object]:
    rows_a, rows_b = align_paired_records(records_a, records_b)
    sink_a: list = []
    sink_b: list = []
    result_a = grouped_bootstrap(
        rows_a, metric_fn=_capturing(metric_fn, sink_a), metric_name=metric_name, task=task, n_bootstrap=n, seed=seed
    )
    result_b = grouped_bootstrap(
        rows_b,
        metric_fn=_capturing(metric_fn, sink_b),
        metric_name=metric_name,
        task=task,
        n_bootstrap=n,
        seed=seed,
        forced_draws=result_a.sampled_cluster_sequences,
    )
    # sink[0] is the full-sample point estimate, sink[1:] the replicates.
    point_a, point_b = sink_a[0], sink_b[0]
    rep_a = np.array([math.nan if v is None else v for v in sink_a[1:]], dtype=np.float64)
    rep_b = np.array([math.nan if v is None else v for v in sink_b[1:]], dtype=np.float64)
    if len(rep_a) != n or len(rep_b) != n:
        raise PairedBootstrapError("replicate bookkeeping mismatch")
    valid = np.isfinite(rep_a) & np.isfinite(rep_b)
    diff_all = rep_b - rep_a
    diff_valid = diff_all[valid]
    cis: Dict[str, Optional[Tuple[float, float]]] = {}
    for level in ci_levels:
        if not 0.0 < level < 1.0:
            raise PairedBootstrapError("ci level must be in (0,1)")
        if len(diff_valid) == 0:
            cis[_level_key(level)] = None
            continue
        alpha = (1.0 - level) / 2.0
        values = diff_valid.tolist()
        cis[_level_key(level)] = (percentile(values, alpha), percentile(values, 1.0 - alpha))
    diff_point = None if point_a is None or point_b is None else float(point_b - point_a)
    return {
        "version": PAIRED_BOOTSTRAP_VERSION,
        "metric_name": metric_name,
        "task": task,
        "difference_definition": "metric(B) - metric(A)",
        "seed": seed,
        "n_requested": n,
        "n_valid_joint": int(valid.sum()),
        "n_invalid": int(n - valid.sum()),
        "n_stays": result_a.n_original_stays,
        "n_rows": result_a.n_original_examples,
        "point_a": point_a,
        "point_b": point_b,
        "diff_point": diff_point,
        "ci": cis,
        "per_replicate_a": rep_a[valid],
        "per_replicate_b": rep_b[valid],
        "per_replicate_diff": diff_valid,
        "per_replicate_all_a": rep_a,
        "per_replicate_all_b": rep_b,
        "valid_mask": valid,
        "sampled_cluster_sequences": result_a.sampled_cluster_sequences,
    }


def ci_excludes_zero(result: Dict[str, object], level: float) -> bool:
    ci = result["ci"].get(_level_key(level))
    if ci is None:
        return False
    return ci[0] > 0.0 or ci[1] < 0.0
