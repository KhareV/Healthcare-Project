"""ICU conditional-prior (Phase-1-justified, fold-safe) TRAIN-only median
remaining-time table. DEVELOPMENT_VALIDATION_ONLY.

CRITICAL fold-safety contract (Part 7 / Part 14 of the Phase-2 spec): a
prior used to score a row must never have been fit using that row's own
label. `fit_conditional_prior_table` is fit on a Sequence of (row, hours)
pairs and returns a pure lookup table; the caller is responsible for
supplying only the rows/labels of the appropriate training subset (the
other folds during CV, or full TRAIN for the final VALIDATION fit) --
`apply_conditional_prior` never sees labels at all, so it cannot leak them.
"""

from __future__ import annotations

from typing import Dict, Mapping, Sequence, Tuple

import numpy as np

from evaluation.weighted_stats import weighted_median
from evaluation.weights import compute_stay_weights

PRIOR_VERSION = "performance_v2_icu_conditional_prior_v1"
MIN_CELL = 20
ELAPSED_EDGES = (24, 42, 60, 78, 96, 1e9)


def _elapsed_band(elapsed: float) -> int:
    return next(i for i in range(len(ELAPSED_EDGES) - 1) if ELAPSED_EDGES[i] <= elapsed < ELAPSED_EDGES[i + 1])


def _severity_band(sofa) -> str:
    if sofa is None:
        return "unknown"
    return "low" if sofa <= 3 else ("mid" if sofa <= 7 else "high")


def _coarse_key(row: Mapping[str, object]) -> int:
    return _elapsed_band(row["elapsed_episode_hours_at_t"])


def _fine_key(row: Mapping[str, object]) -> Tuple[int, int, str, str]:
    return (_elapsed_band(row["elapsed_episode_hours_at_t"]), row["cutoff_index"], row["cardiac_condition_group"], _severity_band(row.get("baseline_sofa")))


def fit_conditional_prior_table(fit_rows: Sequence[Mapping[str, object]], fit_hours: Sequence[float]) -> Mapping[str, object]:
    """TRAIN-only (or TRAIN-fold-only) fit. Never touches labels outside
    `fit_hours`, which the caller must restrict to the appropriate subset."""

    def weighted_median_by(key_fn):
        buckets_hours: Dict[object, list] = {}
        buckets_stays: Dict[object, list] = {}
        counts: Dict[object, int] = {}
        for row, hours in zip(fit_rows, fit_hours):
            key = key_fn(row)
            buckets_hours.setdefault(key, []).append(hours)
            buckets_stays.setdefault(key, []).append(row["stay_id"])
            counts[key] = counts.get(key, 0) + 1
        medians = {}
        for key, hours_list in buckets_hours.items():
            weights = compute_stay_weights(buckets_stays[key], [True] * len(hours_list)).weights
            medians[key] = float(weighted_median(hours_list, weights))
        return medians, counts

    coarse_medians, _coarse_counts = weighted_median_by(_coarse_key)
    fine_medians, fine_counts = weighted_median_by(_fine_key)
    global_weights = compute_stay_weights([row["stay_id"] for row in fit_rows], [True] * len(fit_rows)).weights
    global_median = float(weighted_median(list(fit_hours), global_weights))
    return {
        "prior_version": PRIOR_VERSION, "min_cell_count": MIN_CELL,
        "coarse_medians": {str(k): v for k, v in coarse_medians.items()},
        "fine_medians": {str(k): v for k, v in fine_medians.items()},
        "fine_counts": {str(k): v for k, v in fine_counts.items()},
        "global_median": global_median,
        "n_fit_rows": len(fit_rows),
    }


def apply_conditional_prior(rows: Sequence[Mapping[str, object]], table: Mapping[str, object]) -> np.ndarray:
    coarse = table["coarse_medians"]
    fine = table["fine_medians"]
    fine_counts = table["fine_counts"]
    global_median = table["global_median"]
    predictions = []
    for row in rows:
        fine_key = str(_fine_key(row))
        if fine_counts.get(fine_key, 0) >= table["min_cell_count"] and fine_key in fine:
            predictions.append(fine[fine_key])
        else:
            coarse_key = str(_coarse_key(row))
            predictions.append(coarse.get(coarse_key, global_median))
    return np.asarray(predictions, dtype=np.float64)
