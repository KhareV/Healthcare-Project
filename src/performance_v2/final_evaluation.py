"""Performance-v2 Phase 4: shared, frozen inference/evaluation helpers for
the ONE-TIME fresh-test run. Every transform here is APPLY-ONLY -- nothing
in this module fits/refits anything on any row set. Group-A column universes
and context normalization are reconstructed/loaded exactly as frozen by
Phase 3's final DEV refit; this module never mutates them.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Mapping, Sequence, Tuple

import numpy as np
import xgboost as xgb

from performance_v2.context_normalization import apply_normalization
from performance_v2.data_loading import load_dev_rows
from performance_v2.targets import icu_hours_target, icu_rows, recovery_targets, support_rows
from performance_v2.v2_features import build_group_a_matrix, group_a_feature_names

TASK_VARIANT = {
    "recovery24": "B_MIN",
    "recovery48": "B_MIN",
    "icu_stay_time": "B_PLUS_F",
    "organ_support": "B_FULL",
}

TASK_MODEL_FILE = {
    "recovery24": "artifacts/performance_v2/phase3/models/recovery24_xgb_v2.json",
    "recovery48": "artifacts/performance_v2/phase3/models/recovery48_xgb_v2.json",
    "icu_stay_time": "artifacts/performance_v2/phase3/models/icu_time_xgb_v2.json",
    "organ_support": "artifacts/performance_v2/phase3/models/organ_support_xgb_v2.json",
}


def dev_task_rows(root: Path, task: str) -> Tuple[Mapping[str, object], ...]:
    """The exact DEV row population Phase-3's final refit used for `task`."""

    train_rows = load_dev_rows(root, splits=("train",))
    val_rows = load_dev_rows(root, splits=("validation",))
    dev_rows = train_rows + val_rows
    if task in ("recovery24", "recovery48"):
        horizon = task[-2:]
        return recovery_targets(dev_rows, horizon)
    if task == "icu_stay_time":
        return icu_rows(dev_rows)
    if task == "organ_support":
        return support_rows(dev_rows)
    raise ValueError("unknown task: " + task)


def reconstruct_group_a_names(root: Path, task: str) -> Tuple[str, ...]:
    """Deterministically reproduce the exact Group-A column universe Phase-3's
    final refit used for `task` -- a pure function of the frozen DEV row
    population, which has not changed since Phase 3."""

    rows = dev_task_rows(root, task)
    return group_a_feature_names(rows)


def load_frozen_normalization(root: Path, variant: str) -> Mapping[str, object]:
    path = root / f"artifacts/performance_v2/phase3/preprocessing/context_normalization_v2_{variant}.json"
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def build_inference_matrix(
    rows: Sequence[Mapping[str, object]],
    *,
    group_a_names: Tuple[str, ...],
    normalization: Mapping[str, object],
) -> np.ndarray:
    """Apply-only: Group-A columns fixed to `group_a_names`, context columns
    fixed to the frozen `normalization` mean/std. Fits nothing."""

    matrix_a, _ = build_group_a_matrix(rows, names=group_a_names)
    context = apply_normalization(rows, normalization)
    return np.concatenate([matrix_a, context], axis=1)


def load_xgb_model(root: Path, task: str):
    path = root / TASK_MODEL_FILE[task]
    if task == "organ_support":
        model = xgb.XGBClassifier()
    else:
        model = xgb.XGBRegressor()
    model.load_model(str(path))
    return model


def isotonic_apply(raw_probabilities: Sequence[float], calibration_knots: Mapping[str, object]) -> np.ndarray:
    """Piecewise-linear apply of the frozen isotonic_support_v2 knots
    (out_of_bounds='clip', matching the fitted sklearn IsotonicRegression)."""

    x = np.asarray(calibration_knots["x_thresholds"], dtype=np.float64)
    y = np.asarray(calibration_knots["y_thresholds"], dtype=np.float64)
    raw = np.asarray(raw_probabilities, dtype=np.float64)
    return np.interp(raw, x, y)


def postprocess_icu_hours(raw_log_prediction: np.ndarray) -> np.ndarray:
    clamped = np.clip(raw_log_prediction, 0.0, None)
    hours = np.expm1(clamped)
    if not np.all(np.isfinite(hours)):
        raise ValueError("postprocessed remaining ICU hours must be finite")
    return hours


def verify_feature_dimension(root: Path, task: str) -> Mapping[str, object]:
    """Preflight-only check: reconstructed Group-A + frozen context feature
    count must equal the frozen model's own num_feature. Reads DEV rows only
    (never the fresh cohort)."""

    variant = TASK_VARIANT[task]
    group_a_names = reconstruct_group_a_names(root, task)
    normalization = load_frozen_normalization(root, variant)
    expected_total = len(group_a_names) + len(normalization["feature_names"])
    import json

    model_json = json.loads((root / TASK_MODEL_FILE[task]).read_text(encoding="utf-8"))
    num_feature = int(model_json["learner"]["learner_model_param"]["num_feature"])
    if expected_total != num_feature:
        raise ValueError(
            f"{task}: reconstructed feature count {expected_total} != model num_feature {num_feature}"
        )
    return {
        "task": task,
        "variant": variant,
        "group_a_count": len(group_a_names),
        "context_count": len(normalization["feature_names"]),
        "total": expected_total,
        "model_num_feature": num_feature,
    }
