"""Performance-v2 Phase 1, Part 12: TRAIN vs VALIDATION generalization gap.

DEVELOPMENT_DIAGNOSTIC_ONLY. Refits ONLY the already-chosen (Part 7)
observable-oracle configuration per task (no new search) and evaluates it
on both TRAIN and VALIDATION with the same stay-balanced estimator, to
check whether Part 7's oracle gain is a real, generalizing signal or
overfitting to validation. Never touches v1 TEST.

For the CURRENT SELECTED v1 models (xgb-recovery-014, gru-icu-time-026,
xgb-support-024), a from-scratch frozen-bundle TRAIN-side inference run was
not attempted in Phase 1's time budget (GRU requires building the exact
torch structured-input pipeline; XGBoost's early-stopped best_iteration
also is not directly comparable without re-running inference). This is a
documented scope limitation, not a silent omission: the observable-oracle
train/validation gap below is used as the primary generalization evidence,
since it directly tests the load-bearing Phase-1 finding (that structural
features close most of the gap) for overfitting risk.
"""

import json
import sys
from pathlib import Path

import numpy as np
from xgboost import XGBClassifier, XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.synthetic.config import canonical_json_bytes  # noqa: E402
from evaluation.metrics import evaluate_icu_time, evaluate_organ_support, evaluate_recovery_horizon  # noqa: E402
from performance_v2.data_loading import load_dev_rows  # noqa: E402
from performance_v2.feature_matrix import build_feature_matrix_fast, load_sofa_trend_cache  # noqa: E402
from performance_v2.targets import icu_hours_target, icu_rows, make_records, recovery_targets, support_rows  # noqa: E402
from vedant_infra.hashing import sha256_file  # noqa: E402

OUT_DIR = ROOT / "artifacts/performance_v2/phase1"
SEED = 2026_09_24


def _gap(train_metric, val_metric):
    absolute_gap = val_metric - train_metric
    relative_gap = absolute_gap / train_metric if train_metric else float("nan")
    return {"train": train_metric, "validation": val_metric, "absolute_gap": absolute_gap, "relative_gap": relative_gap}


def main() -> None:
    oracle_summary = json.loads((ROOT / "artifacts/performance_v2/phase1/observable_oracle/observable_oracle_summary_v1.json").read_text())
    train_rows = load_dev_rows(ROOT, splits=("train",))
    validation_rows = load_dev_rows(ROOT, splits=("validation",))
    sofa_trends = load_sofa_trend_cache(ROOT)

    result = {
        "status": "DEVELOPMENT_DIAGNOSTIC_ONLY",
        "artifact_version": "performance_v2_phase1_train_validation_gap_v1",
        "scope_limitation": (
            "Computed for the OBSERVABLE_ORACLE only (refit at its already-"
            "chosen Part-7 configuration, no new search). The frozen v1 "
            "selected models' own TRAIN-side metrics were not recomputed in "
            "Phase 1 (would require re-running frozen-bundle inference on "
            "TRAIN, including the GRU's torch structured-input pipeline); "
            "this is a documented scope limitation."
        ),
    }

    for horizon in ("24", "48"):
        train_h = recovery_targets(train_rows, horizon)
        val_h = recovery_targets(validation_rows, horizon)
        matrix, names = build_feature_matrix_fast(train_h + val_h, sofa_trends)
        n_train = len(train_h)
        train_x, val_x = matrix[:n_train], matrix[n_train:]
        train_y = np.asarray([float(r["delta_sofa_" + horizon]) for r in train_h])
        val_y = np.asarray([float(r["delta_sofa_" + horizon]) for r in val_h])
        config = oracle_summary["recovery"]["recovery" + horizon]["chosen_config"]
        model = XGBRegressor(objective="reg:absoluteerror", random_state=SEED, missing=np.nan, subsample=0.8, colsample_bytree=0.8, **config)
        model.fit(train_x, train_y)
        train_pred = model.predict(train_x)
        val_pred = model.predict(val_x)
        train_eval = evaluate_recovery_horizon(make_records(train_h, train_y.tolist(), train_pred.tolist()), horizon=horizon + "h")
        val_eval = evaluate_recovery_horizon(make_records(val_h, val_y.tolist(), val_pred.tolist()), horizon=horizon + "h")
        result["recovery" + horizon] = {"mae": _gap(train_eval.metrics["mae"], val_eval.metrics["mae"])}

    train_icu = icu_rows(train_rows)
    val_icu = icu_rows(validation_rows)
    matrix, names = build_feature_matrix_fast(train_icu + val_icu, sofa_trends)
    n_train = len(train_icu)
    train_x, val_x = matrix[:n_train], matrix[n_train:]
    train_hours = np.asarray([icu_hours_target(r) for r in train_icu])
    val_hours = np.asarray([icu_hours_target(r) for r in val_icu])
    train_log1p = np.log1p(train_hours)
    config = oracle_summary["icu_stay_time"]["chosen_config"]
    model = XGBRegressor(objective="reg:absoluteerror", random_state=SEED, missing=np.nan, subsample=0.8, colsample_bytree=0.8, **config)
    model.fit(train_x, train_log1p)
    train_pred = np.expm1(np.clip(model.predict(train_x), 0, None))
    val_pred = np.expm1(np.clip(model.predict(val_x), 0, None))
    train_eval = evaluate_icu_time(make_records(train_icu, train_hours.tolist(), train_pred.tolist()))
    val_eval = evaluate_icu_time(make_records(val_icu, val_hours.tolist(), val_pred.tolist()))
    result["icu_stay_time"] = {
        "median_absolute_error": _gap(train_eval.metrics["median_absolute_error"], val_eval.metrics["median_absolute_error"]),
        "mae": _gap(train_eval.metrics["mae"], val_eval.metrics["mae"]),
    }

    train_sup = support_rows(train_rows)
    val_sup = support_rows(validation_rows)
    matrix, names = build_feature_matrix_fast(train_sup + val_sup, sofa_trends)
    n_train = len(train_sup)
    train_x, val_x = matrix[:n_train], matrix[n_train:]
    train_y = [float(r["organ_support_label"]) for r in train_sup]
    val_y = [float(r["organ_support_label"]) for r in val_sup]
    config = oracle_summary["organ_support"]["chosen_config"]
    model = XGBClassifier(objective="binary:logistic", random_state=SEED, missing=np.nan, subsample=0.8, colsample_bytree=0.8, eval_metric="aucpr", **config)
    model.fit(train_x, train_y)
    train_pred = model.predict_proba(train_x)[:, 1]
    val_pred = model.predict_proba(val_x)[:, 1]
    train_eval = evaluate_organ_support(make_records(train_sup, train_y, train_pred.tolist()))
    val_eval = evaluate_organ_support(make_records(val_sup, val_y, val_pred.tolist()))
    result["organ_support"] = {"auprc": _gap(train_eval.metrics["auprc"], val_eval.metrics["auprc"])}

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "train_validation_gap_v1.json"
    out_path.write_bytes(canonical_json_bytes(result))
    print("wrote", out_path, sha256_file(out_path))


if __name__ == "__main__":
    main()
