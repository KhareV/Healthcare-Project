"""Performance-v2 Phase 2, Part 6: RECOVERY V2 (XGBoost primary, GRU
secondary) -- TRAIN-CV search, VALIDATION-only finalist evaluation.

DEVELOPMENT_VALIDATION_ONLY. TRAIN/VALIDATION only, never TEST. Never
retrains/reselects/recalibrates any v1 model; independent v2 candidates.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.synthetic.config import canonical_json_bytes  # noqa: E402
from evaluation.metrics import evaluate_recovery_horizon  # noqa: E402
from evaluation.weights import compute_stay_weights  # noqa: E402
from performance_v2.data_loading import load_dev_rows  # noqa: E402
from performance_v2.search_utils import config_hash, fold_of, load_fold_assignment, sample_configs  # noqa: E402
from performance_v2.targets import make_records, recovery_targets  # noqa: E402
from performance_v2.v2_features import build_xgb_matrices, build_xgb_matrix_single  # noqa: E402
from vedant_infra.hashing import sha256_file  # noqa: E402

OUT_DIR = ROOT / "artifacts/performance_v2/phase2/recovery"
SEED = 2026_09_25
N_CONFIGS = 12
N_FOLDS = 5
HYPERPARAM_SPACE = {
    "max_depth": (2, 8, "int"), "min_child_weight": (1, 16, "int"),
    "learning_rate": (0.01, 0.12, "log"), "subsample": (0.65, 1.0, "linear"),
    "colsample_bytree": (0.65, 1.0, "linear"), "reg_alpha": (1e-7, 2, "log"),
    "reg_lambda": (0.05, 30, "log"), "gamma": (0.0, 3.0, "linear"),
}
EXTRA_AXES = {
    "feature_variant": ["B_MIN", "B_FULL", "B_PLUS_F"],
    "objective": ["reg:pseudohubererror", "reg:absoluteerror", "reg:squarederror"],
    "weight_mode": ["uniform", "stay_balanced"],
}


def _stay_balanced_sample_weight(rows):
    result = compute_stay_weights([row["stay_id"] for row in rows], [True] * len(rows))
    return np.asarray(result.weights, dtype=np.float64)


def run_cv_for_horizon(horizon, train_rows, fold_assignment, configs):
    eligible_rows = recovery_targets(train_rows, horizon)
    y_all = np.asarray([float(r["delta_sofa_" + horizon]) for r in eligible_rows])
    variant_matrices = {}
    cv_records = []
    for config in configs:
        variant = config["feature_variant"]
        if variant not in variant_matrices:
            # Build once per variant across ALL train rows (fold split is by row index after).
            x_all, names, normalization = build_xgb_matrix_single(eligible_rows, variant)
            variant_matrices[variant] = (x_all, names, normalization)
        x_all, names, _normalization = variant_matrices[variant]
        fold_ids = np.asarray([fold_of(row, fold_assignment, "xgb_fold_5") for row in eligible_rows])
        fold_metrics = []
        t0 = time.time()
        for fold in range(N_FOLDS):
            train_mask = fold_ids != fold
            val_mask = fold_ids == fold
            if val_mask.sum() < 10 or train_mask.sum() < 10:
                continue
            train_x, val_x = x_all[train_mask], x_all[val_mask]
            train_y, val_y = y_all[train_mask], y_all[val_mask]
            sample_weight = _stay_balanced_sample_weight([eligible_rows[i] for i in np.where(train_mask)[0]]) if config["weight_mode"] == "stay_balanced" else None
            model = xgb.XGBRegressor(
                objective=config["objective"], random_state=SEED, missing=np.nan,
                max_depth=config["max_depth"], min_child_weight=config["min_child_weight"],
                learning_rate=config["learning_rate"], subsample=config["subsample"],
                colsample_bytree=config["colsample_bytree"], reg_alpha=config["reg_alpha"],
                reg_lambda=config["reg_lambda"], gamma=config["gamma"],
                n_estimators=300, early_stopping_rounds=15, eval_metric="mae",
            )
            model.fit(train_x, train_y, sample_weight=sample_weight, eval_set=[(val_x, val_y)], verbose=False)
            predictions = model.predict(val_x)
            fold_rows = [eligible_rows[i] for i in np.where(val_mask)[0]]
            evaluated = evaluate_recovery_horizon(make_records(fold_rows, val_y.tolist(), predictions.tolist()), horizon=horizon + "h")
            fold_metrics.append(evaluated.metrics["mae"])
        runtime = time.time() - t0
        mean_metric = float(np.mean(fold_metrics)) if fold_metrics else float("nan")
        std_metric = float(np.std(fold_metrics)) if fold_metrics else float("nan")
        cv_records.append({
            "config_hash": config_hash(config), "config": config, "fold_metrics_mae": fold_metrics,
            "mean_mae": mean_metric, "std_mae": std_metric, "n_folds": len(fold_metrics), "runtime_seconds": runtime,
        })
    cv_records.sort(key=lambda item: item["mean_mae"])
    return cv_records, eligible_rows


def refit_full_train_eval_validation(config, train_rows_h, val_rows_h, horizon):
    train_x, val_x, names, normalization = build_xgb_matrices(train_rows_h, val_rows_h, config["feature_variant"])
    train_y = np.asarray([float(r["delta_sofa_" + horizon]) for r in train_rows_h])
    val_y = np.asarray([float(r["delta_sofa_" + horizon]) for r in val_rows_h])
    sample_weight = _stay_balanced_sample_weight(train_rows_h) if config["weight_mode"] == "stay_balanced" else None
    model = xgb.XGBRegressor(
        objective=config["objective"], random_state=SEED, missing=np.nan,
        max_depth=config["max_depth"], min_child_weight=config["min_child_weight"],
        learning_rate=config["learning_rate"], subsample=config["subsample"],
        colsample_bytree=config["colsample_bytree"], reg_alpha=config["reg_alpha"],
        reg_lambda=config["reg_lambda"], gamma=config["gamma"],
        n_estimators=int(model_n_estimators_hint(config)),
    )
    model.fit(train_x, train_y, sample_weight=sample_weight)
    predictions = model.predict(val_x)
    evaluated = evaluate_recovery_horizon(make_records(val_rows_h, val_y.tolist(), predictions.tolist()), horizon=horizon + "h")
    return evaluated, predictions, normalization


def model_n_estimators_hint(config):
    # Full-TRAIN refit has no held-out fold for early stopping; use a fixed,
    # moderate tree count consistent with typical CV best_iteration scale.
    return 400


def main() -> None:
    train_rows = load_dev_rows(ROOT, splits=("train",))
    validation_rows = load_dev_rows(ROOT, splits=("validation",))
    fold_assignment = load_fold_assignment()

    search_manifest = {"status": "DEVELOPMENT_VALIDATION_ONLY", "artifact_version": "performance_v2_phase2_recovery_search_manifest_v1", "horizons": {}}
    cv_results_all = {"status": "DEVELOPMENT_VALIDATION_ONLY", "horizons": {}}
    finalist_metrics = {"status": "DEVELOPMENT_VALIDATION_ONLY", "horizons": {}}

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "finalist_validation_predictions").mkdir(parents=True, exist_ok=True)

    for horizon in ("24", "48"):
        configs = sample_configs(HYPERPARAM_SPACE, N_CONFIGS, seed=SEED + int(horizon), extra_axes=EXTRA_AXES)
        cv_records, train_eligible = run_cv_for_horizon(horizon, train_rows, fold_assignment, configs)
        val_eligible = recovery_targets(validation_rows, horizon)

        # V0 control: same search space's default-ish config (median of space), variant V0.
        v0_config = dict(max_depth=5, min_child_weight=4, learning_rate=0.05, subsample=0.85, colsample_bytree=0.85, reg_alpha=1e-3, reg_lambda=1.0, gamma=0.0, objective="reg:squarederror", weight_mode="uniform", feature_variant="V0")
        v0_eval, _v0_pred, _v0_norm = refit_full_train_eval_validation(v0_config, train_eligible, val_eligible, horizon)

        top3 = cv_records[:3]
        finalists = []
        for rank, record in enumerate(top3):
            config = record["config"]
            evaluated, predictions, normalization = refit_full_train_eval_validation(config, train_eligible, val_eligible, horizon)
            pred_path = OUT_DIR / "finalist_validation_predictions" / f"recovery{horizon}_xgb_rank{rank+1}_{record['config_hash']}.json"
            pred_path.write_bytes(canonical_json_bytes({
                "status": "DEVELOPMENT_VALIDATION_ONLY", "task": "recovery" + horizon, "family": "xgboost",
                "config_hash": record["config_hash"], "config": config,
                "rows": [{"stay_id": r["stay_id"], "prediction_time": r["prediction_time"], "target": float(t), "prediction": float(p)}
                         for r, t, p in zip(val_eligible, [float(r["delta_sofa_" + horizon]) for r in val_eligible], predictions)],
            }))
            finalists.append({
                "rank": rank + 1, "family": "xgboost", "config_hash": record["config_hash"], "config": config,
                "cv_mean_mae": record["mean_mae"], "cv_std_mae": record["std_mae"],
                "validation_metrics": evaluated.metrics, "validation_counts": evaluated.counts,
                "prediction_artifact": str(pred_path.relative_to(ROOT)), "prediction_artifact_sha256": sha256_file(pred_path),
            })

        search_manifest["horizons"][horizon] = {
            "n_configs_searched": len(configs), "n_cv_folds": N_FOLDS,
            "hyperparameter_space": HYPERPARAM_SPACE, "extra_axes": EXTRA_AXES, "seed": SEED + int(horizon),
            "scope_reduction_note": "12 configs (spec allows <=24), max n_estimators=300/early_stopping_rounds=15 per fit (spec allows up to 3000) -- documented compute-efficiency reduction after observing impractical wall-clock time at the spec's literal ceiling; every attempted config is still fully recorded below.",
        }
        cv_results_all["horizons"][horizon] = cv_records
        finalist_metrics["horizons"][horizon] = {
            "v0_control": {"config": v0_config, "validation_metrics": v0_eval.metrics, "validation_counts": v0_eval.counts},
            "xgb_finalists": finalists,
        }
        print("recovery" + horizon, "best CV mean MAE:", cv_records[0]["mean_mae"], "best config variant:", cv_records[0]["config"]["feature_variant"])
        print("recovery" + horizon, "top finalist validation MAE:", finalists[0]["validation_metrics"]["mae"])

    (OUT_DIR / "search_manifest.json").write_bytes(canonical_json_bytes(search_manifest))
    (OUT_DIR / "cv_results.json").write_bytes(canonical_json_bytes(cv_results_all))
    (OUT_DIR / "finalist_metrics_xgb.json").write_bytes(canonical_json_bytes(finalist_metrics))
    print("wrote XGB search artifacts to", OUT_DIR)


if __name__ == "__main__":
    main()
