"""V2 TreeSHAP explanations for the four frozen XGBoost models.

Uses shap.TreeExplainer directly against the exact matrix
serving.v2.runtime built for the prediction (never a re-derived one),
verifies the booster's serialized bytes are unchanged before/after (SHAP
must never mutate the frozen model), and checks the additivity identity
(sum of SHAP values + base value == raw model margin) before returning
anything. Support is explained on its RAW (pre-calibration) margin output,
exactly as isotonic calibration is not SHAP-additive.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Mapping, Sequence, Tuple

import numpy as np

_BIN_RE = re.compile(r"^A__bin(\d)__(.+)__(value|mask|tslo)$")
_STATIC_RE = re.compile(r"^A__static__(.+?)(==(.+))?$")


class TreeShapError(RuntimeError):
    pass


@dataclass(frozen=True)
class ExplanationItem:
    feature_name: str
    readable_label: str
    attribution: float


def _readable_label(flat_name: str) -> str:
    match = _BIN_RE.match(flat_name)
    if match:
        bin_index, base, channel = match.groups()
        bin_index = int(bin_index)
        start = -48 + 6 * bin_index
        end = start + 6
        window = f"{abs(start)}-{abs(end)}h ago" if end != 0 else "most recent 6h"
        clean_base = re.sub(r"__(latest|sum)$", "", base).replace("_", " ").strip()
        suffix = {"value": "", "mask": " (observed?)", "tslo": " (time since last observed)"}[channel]
        return f"{clean_base} — {window}{suffix}"
    match = _STATIC_RE.match(flat_name)
    if match:
        base, _, value = match.groups()
        label = base.replace("_", " ")
        return f"{label} = {value}" if value else label
    if flat_name.startswith("B__"):
        return {
            "B__elapsed_episode_hours_at_t": "Elapsed ICU time at cutoff",
            "B__cutoff_index": "Cutoff index",
            "B__hours_since_first_eligible_cutoff": "Hours since first eligible cutoff",
            "B__current_sofa": "Current SOFA",
        }.get(flat_name, flat_name)
    if flat_name.startswith("F__"):
        parts = flat_name.split("__")
        support = parts[1].replace("_", " ") if len(parts) > 1 else flat_name
        metric = parts[2].replace("_", " ") if len(parts) > 2 else ""
        return f"{support} — {metric}"
    return flat_name


def explain(model, matrix: np.ndarray, feature_names: Sequence[str], *, top_k: int = 8) -> Tuple[List[ExplanationItem], List[ExplanationItem], Mapping[str, object]]:
    """Return (top_positive, top_negative, diagnostics) for one [1,D] row.

    diagnostics includes the additivity check result and a note on whether
    this explains the raw or calibrated output (raw, always, for XGBoost
    margin -- isotonic calibration is applied strictly after and is not
    SHAP-additive).
    """

    try:
        import shap
    except ImportError as error:  # pragma: no cover - dependency guard
        raise TreeShapError("shap is not installed") from error

    booster = model.get_booster()
    before = booster.save_raw()
    explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent", model_output="raw")
    shap_values = explainer.shap_values(matrix)
    after = booster.save_raw()
    if before != after:
        raise TreeShapError("TreeExplainer mutated the frozen booster; refusing to return an explanation")

    values = np.asarray(shap_values).reshape(-1)
    base_value = float(np.asarray(explainer.expected_value).reshape(-1)[0])
    margin = float(model.get_booster().predict(_as_dmatrix(matrix), output_margin=True)[0])
    reconstructed = float(values.sum() + base_value)
    additivity_ok = bool(np.isclose(reconstructed, margin, atol=1e-3))

    ranked = sorted(zip(feature_names, values.tolist()), key=lambda item: item[1], reverse=True)
    positive = [item for item in ranked if item[1] > 0][:top_k]
    negative = sorted([item for item in ranked if item[1] < 0], key=lambda item: item[1])[:top_k]

    def _items(pairs):
        return [ExplanationItem(feature_name=name, readable_label=_readable_label(name), attribution=float(value)) for name, value in pairs]

    diagnostics = {
        "additivity_check_passed": additivity_ok,
        "base_value": base_value,
        "raw_margin_prediction": margin,
        "explains": "raw XGBoost margin output (pre-calibration where applicable)",
        "method": "TreeSHAP (shap.TreeExplainer, tree_path_dependent, model_output=raw)",
    }
    return _items(positive), _items(negative), diagnostics


def _as_dmatrix(matrix: np.ndarray):
    import xgboost as xgb

    return xgb.DMatrix(matrix, missing=np.nan)
