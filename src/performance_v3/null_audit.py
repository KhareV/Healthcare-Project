"""Availability-linked missingness audit (protocol amendment v1.1, defect D1).

For every feature column: NaN-rate gap between rows whose 24h horizon is NOT available
(stay ends within 24h of t) and rows where it is, matched on elapsed episode time so the
legitimate time-dependence of observation processes is removed. A feature whose missingness
encodes horizon availability leaks future information.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

ELAPSED_BIN_EDGES = (30, 42, 54, 66, 78, 90, 102)
THRESHOLD = 0.05


def nan_gap_by_column(matrix: np.ndarray, names: Sequence[str], elapsed_hours: np.ndarray,
                      horizon_unavailable: np.ndarray, min_cell: int = 50) -> List[Tuple[str, float, float]]:
    bins = np.digitize(elapsed_hours, ELAPSED_BIN_EDGES)
    out = []
    for j, name in enumerate(names):
        nan = np.isnan(matrix[:, j])
        rate = float(nan.mean())
        diffs, weights = [], []
        for b in np.unique(bins):
            in_bin = bins == b
            a = in_bin & horizon_unavailable
            e = in_bin & ~horizon_unavailable
            if a.sum() >= min_cell and e.sum() >= min_cell:
                diffs.append(float(nan[a].mean() - nan[e].mean()))
                weights.append(float(min(a.sum(), e.sum())))
        gap = float(np.average(diffs, weights=weights)) if weights else 0.0
        out.append((name, gap, rate))
    return out


def audit_cache(cache, threshold: float = THRESHOLD) -> Dict[str, object]:
    names = list(cache.names)
    elapsed = cache.matrix[:, names.index("B__elapsed_episode_hours_at_t")]
    unavailable = ~np.asarray(cache.meta["recovery24_eligible"], dtype=bool)
    gaps = sorted(nan_gap_by_column(cache.matrix, names, elapsed, unavailable), key=lambda t: -abs(t[1]))
    flagged = [g for g in gaps if abs(g[1]) > threshold]
    return {
        "rows": int(cache.matrix.shape[0]), "columns": len(names), "threshold": threshold,
        "flagged_columns": [{"name": n, "gap": g, "nan_rate": r} for n, g, r in flagged],
        "max_abs_gap": abs(gaps[0][1]) if gaps else 0.0,
        "top_gaps": [{"name": n, "gap": g, "nan_rate": r} for n, g, r in gaps[:10]],
        "current_sofa_nan_fraction": float(np.isnan(cache.matrix[:, names.index("B__current_sofa")]).mean()) if "B__current_sofa" in names else None,
        "verdict": "PASS" if not flagged else "FAIL",
    }
