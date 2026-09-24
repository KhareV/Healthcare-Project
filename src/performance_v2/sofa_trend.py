"""Past-cutoff-only SOFA trend for Performance-v2 Phase 1 diagnostic features.

DEVELOPMENT_DIAGNOSTIC_ONLY. Reuses experiments.stage5_final_test's frozen,
unmodified SOFA-scoring context/support-provider machinery
(data.synthetic.sofa.sofa_at) exactly as Stage 5 did for its
complete-component sensitivity analysis. Every cutoff passed to sofa_at()
here is at-or-before the row's own prediction cutoff t (t, t-6h, t-12h,
t-24h, t-48h, clamped to the stay's intime) -- never a future cutoff, and
sofa_at() itself only reads history at-or-before whatever cutoff it is
given, so this cannot see anything beyond the requested past cutoff.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Dict, Mapping, Sequence, Tuple

from data.synthetic.sofa import sofa_at
from experiments.stage5_final_test import _stay_support_provider, _z, build_sofa_scoring_context

_LAGS_HOURS = {"t": 0, "t_minus_6h": 6, "t_minus_12h": 12, "t_minus_24h": 24, "t_minus_48h": 48}


def compute_sofa_trends(root, rows: Sequence[Mapping[str, object]]) -> Dict[Tuple[str, str], Dict[str, float]]:
    context = build_sofa_scoring_context(root)
    provider = context.provider
    stay_support_cache: dict = {}

    def support_provider_for(stay_id):
        if stay_id not in stay_support_cache:
            stay_support_cache[stay_id] = _stay_support_provider(context, stay_id)
        return stay_support_cache[stay_id]

    trends: Dict[Tuple[str, str], Dict[str, float]] = {}
    for row in rows:
        stay_id = row["stay_id"]
        stay_record = provider.stays.get(stay_id)
        if stay_record is None:
            continue
        stay_history = context.history_by_stay.get(stay_id, ())
        stay_support = support_provider_for(stay_id)
        intime = datetime.fromisoformat(str(stay_record["intime"]).replace("Z", "+00:00")).astimezone(timezone.utc)
        t = datetime.fromisoformat(str(row["prediction_time"]).replace("Z", "+00:00")).astimezone(timezone.utc)
        trend: Dict[str, float] = {}
        for label, lag in _LAGS_HOURS.items():
            cutoff = t - timedelta(hours=lag)
            if cutoff < intime:
                continue
            try:
                result = sofa_at(
                    stay_history, stay_record, _z(cutoff),
                    spec=provider.spec, bindings=provider.bindings, support_provider=stay_support,
                )
            except Exception:
                continue
            trend[label] = float(result.total_score)
        trends[(stay_id, row["prediction_time"])] = trend
    return trends
