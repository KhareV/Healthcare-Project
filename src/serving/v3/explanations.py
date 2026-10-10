"""TreeSHAP explanations for the four selected V3 heads (xgboost x3, catboost x1).

Reuses performance_v3.serving.tree_shap_values (shap.TreeExplainer on the RAW margin with an additivity
check). This module only adds readable labels, output-domain statements, and group/temporal-bin
aggregation. Attribution is statistical ("contributed to the prediction"), never causal.
"""

from __future__ import annotations

import re
from typing import Dict, List, Mapping, Sequence

import numpy as np

from performance_v3.serving import tree_shap_values
from serving.v2.explanations import _readable_label as _v2_label

_BIN = re.compile(r"^A__bin(\d)__(.+?)__(value|mask|tslo)$")
_D = re.compile(r"^D__(.+?)__(?:latest__)?(mean|min|max|std)_(12|24|48)h$")
_E = re.compile(r"^E__(.+?)__(?:latest__)?observed_(count|fraction)_8bins$")

DOMAINS = {
    "recovery24": "Predicted change in SOFA over 24 h (SOFA units): the tree-ensemble output itself, before the display-only 0-24 clip of absolute SOFA.",
    "recovery48": "Predicted change in SOFA over 48 h (SOFA units): independent model, never chained from the 24 h output; before the display-only 0-24 clip.",
    "icu_stay_time": "Raw model output = log1p(remaining ICU-like hours), before the non-negative clamp and expm1 to hours.",
    "organ_support": "Raw model margin (log-odds) before Platt calibration. The calibrated probability is a monotone transform of this margin; SHAP values are not defined on the calibrated probability.",
}


def _clean(base: str) -> str:
    return re.sub(r"__(latest|sum)$", "", base).replace("_", " ").strip()


def readable_label(name: str) -> str:
    m = _D.match(name)
    if m:
        base, stat, hours = m.groups()
        return f"{_clean(base)} — {stat} over last {hours}h"
    m = _E.match(name)
    if m:
        base, kind = m.groups()
        return f"{_clean(base)} — observed " + ("count (of 8 bins)" if kind == "count" else "fraction (of 8 bins)")
    if name == "B__current_sofa":
        return "Current SOFA at cutoff"
    return _v2_label(name)


def feature_group(name: str) -> str:
    m = _BIN.match(name)
    if m:
        return {"value": "Observed values", "mask": "Observation indicators", "tslo": "Time since last observation"}[m.group(3)]
    if name.startswith("A__static__"):
        return "Static / demographic"
    return {"B": "Episode context & current SOFA", "D": "Windowed statistics (12/24/48 h)", "E": "Observation density", "F": "Support-state history"}.get(name.split("__", 1)[0], "Other")


def temporal_bin(name: str):
    m = _BIN.match(name)
    return int(m.group(1)) if m else None


def explain_task(model, task: str, matrix: np.ndarray, feature_names: Sequence[str], *, stay_id: str, prediction_time: str, top_k: int = 8) -> Dict[str, object]:
    values, base, diag = tree_shap_values(model, np.asarray(matrix).reshape(1, -1))
    v = values[0]
    if len(v) != len(feature_names):
        raise ValueError("SHAP vector length differs from feature names")
    ranked = sorted(zip(feature_names, v.tolist()), key=lambda p: p[1], reverse=True)
    pos = [p for p in ranked if p[1] > 0][:top_k]
    neg = sorted([p for p in ranked if p[1] < 0], key=lambda p: p[1])[:top_k]
    items = lambda pairs: [{"feature_name": n, "label": readable_label(n), "attribution": float(a), "group": feature_group(n)} for n, a in pairs]
    groups: Dict[str, Dict[str, float]] = {}
    bins: Dict[int, Dict[str, float]] = {}
    for n, a in zip(feature_names, v.tolist()):
        g = groups.setdefault(feature_group(n), {"signed_sum": 0.0, "abs_sum": 0.0})
        g["signed_sum"] += a; g["abs_sum"] += abs(a)
        b = temporal_bin(n)
        if b is not None:
            e = bins.setdefault(b, {"signed_sum": 0.0, "abs_sum": 0.0})
            e["signed_sum"] += a; e["abs_sum"] += abs(a)
    return {
        "top_positive_contributors": items(pos),
        "top_negative_contributors": items(neg),
        "method": "tree_shap",
        "diagnostics": {
            "additivity_check_passed": bool(diag["additivity_check_passed"]), "base_value": float(base),
            "raw_margin_prediction": float(base + float(v.sum())), "method": diag["method"],
            "explains": DOMAINS[task], "output_domain_code": diag["output_domain"],
            "model_family": model.family, "causal_claim": False,
            "wording": "Features that contributed to this model's output; statistical attribution, not causal effect.",
        },
        "by_feature_group": sorted(({"group": g, **s} for g, s in groups.items()), key=lambda r: -r["abs_sum"]),
        "by_temporal_bin": [{"bin": b, "hours_ago_start": 48 - 6 * b, "hours_ago_end": 42 - 6 * b, **bins[b]} for b in sorted(bins)],
        "identity": {"stay_id": stay_id, "prediction_time": prediction_time, "task": task},
    }
