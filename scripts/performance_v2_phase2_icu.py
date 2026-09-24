"""Performance-v2 Phase 2, Part 7: ICU V2 (highest priority).

XGBoost (log1p target, primary), conditional-prior/residual candidate
(Phase-1-justified, fold-safe), optional AFT. GRU handled in a separate
script (performance_v2_phase2_icu_gru.py) for tractable runtime isolation.

DEVELOPMENT_VALIDATION_ONLY. TRAIN/VALIDATION only, never TEST.
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
from evaluation.metrics import evaluate_icu_time  # noqa: E402
from evaluation.weights import compute_stay_weights  # noqa: E402
from performance_v2.conditional_prior import apply_conditional_prior, fit_conditional_prior_table  # noqa: E402
from performance_v2.data_loading import load_dev_rows  # noqa: E402
from performance_v2.search_utils import config_hash, fold_of, load_fold_assignment, sample_configs  # noqa: E402
from performance_v2.targets import icu_hours_target, icu_rows, make_records  # noqa: E402
from performance_v2.v2_features import build_xgb_matrices, build_xgb_matrix_single  # noqa: E402
from vedant_infra.hashing import sha256_file  # noqa: E402

OUT_DIR = ROOT / "artifacts/performance_v2/phase2/icu"
SEED = 2026_09_25
N_CONFIGS = 24
N_FOLDS = 5
HYPERPARAM_SPACE = {
    "max_depth": (2, 8, "int"), "min_child_weight": (1, 16, "int"),
    "learning_rate": (0.01, 0.12, "log"), "subsample": (0.65, 1.0, "linear"),
    "colsample_bytree": (0.65, 1.0, "linear"), "reg_alpha": (1e-7, 2, "log"),
    "reg_lambda": (0.05, 30, "log"), "gamma": (0.0, 3.0, "linear"),
}
EXTRA_AXES = {
    "feature_variant": ["B_MIN", "B_FULL", "B_PLUS_F"],
    "objective": ["reg:pseudohubererror", "reg:absoluteerror"],
    "weight_mode": ["uniform", "stay_balanced"],
}


def _stay_balanced_sample_weight(rows):
    result = compute_stay_weights([row["stay_id"] for row in rows], [True] * len(rows))
    return np.asarray(result.weights, dtype=np.float64)


def _xgb_estimator(config, n_estimators=600, early_stopping_rounds=20):
    return xgb.XGBRegressor(
        objective=config["objective"], random_state=SEED, missing=np.nan,
        max_depth=config["max_depth"], min_child_weight=config["min_child_weight"],
        learning_rate=config["learning_rate"], subsample=config["subsample"],
        colsample_bytree=config["colsample_bytree"], reg_alpha=config["reg_alpha"],
        reg_lambda=config["reg_lambda"], gamma=config["gamma"],
        n_estimators=n_estimators,
        **({"early_stopping_rounds": early_stopping_rounds, "eval_metric": "mae"} if early_stopping_rounds else {}),
    )


def run_xgb_cv(rows, hours, fold_assignment, configs):
    fold_ids = np.asarray([fold_of(row, fold_assignment, "xgb_fold_5") for row in rows])
    log_targets = np.log1p(np.asarray(hours))
    variant_matrices = {}
    cv_records = []
    for config in configs:
        variant = config["feature_variant"]
        if variant not in variant_matrices:
            variant_matrices[variant] = build_xgb_matrix_single(rows, variant)
        x_all, names, _normalization = variant_matrices[variant]
        fold_metrics = []
        t0 = time.time()
        for fold in range(N_FOLDS):
            train_mask, val_mask = fold_ids != fold, fold_ids == fold
            if val_mask.sum() < 10 or train_mask.sum() < 10:
                continue
            train_x, val_x = x_all[train_mask], x_all[val_mask]
            train_y, val_y_log = log_targets[train_mask], log_targets[val_mask]
            fold_rows_val = [rows[i] for i in np.where(val_mask)[0]]
            fold_rows_train = [rows[i] for i in np.where(train_mask)[0]]
            sample_weight = _stay_balanced_sample_weight(fold_rows_train) if config["weight_mode"] == "stay_balanced" else None
            model = _xgb_estimator(config)
            model.fit(train_x, train_y, sample_weight=sample_weight, eval_set=[(val_x, val_y_log)], verbose=False)
            raw_pred = model.predict(val_x)
            hours_pred = np.expm1(np.clip(raw_pred, 0, None))
            val_hours = np.expm1(val_y_log)
            evaluated = evaluate_icu_time(make_records(fold_rows_val, val_hours.tolist(), hours_pred.tolist()))
            fold_metrics.append(evaluated.metrics["median_absolute_error"])
        runtime = time.time() - t0
        mean_metric = float(np.mean(fold_metrics)) if fold_metrics else float("nan")
        std_metric = float(np.std(fold_metrics)) if fold_metrics else float("nan")
        cv_records.append({
            "config_hash": config_hash(config), "config": config, "fold_metrics_median_ae": fold_metrics,
            "mean_median_ae": mean_metric, "std_median_ae": std_metric, "n_folds": len(fold_metrics), "runtime_seconds": runtime,
        })
    cv_records.sort(key=lambda item: item["mean_median_ae"])
    return cv_records


def run_conditional_prior_cv(rows, hours, fold_assignment):
    fold_ids = np.asarray([fold_of(row, fold_assignment, "xgb_fold_5") for row in rows])
    hours_arr = np.asarray(hours)
    fold_metrics_prior_only, fold_metrics_a, fold_metrics_b = [], [], []
    variant = "B_FULL"
    x_all, names, _norm = build_xgb_matrix_single(rows, variant)
    for fold in range(N_FOLDS):
        train_mask, val_mask = fold_ids != fold, fold_ids == fold
        if val_mask.sum() < 10 or train_mask.sum() < 10:
            continue
        fit_rows = [rows[i] for i in np.where(train_mask)[0]]
        fit_hours = hours_arr[train_mask]
        table = fit_conditional_prior_table(fit_rows, fit_hours)
        val_rows_fold = [rows[i] for i in np.where(val_mask)[0]]
        val_hours = hours_arr[val_mask]
        prior_pred = apply_conditional_prior(val_rows_fold, table)
        evaluated = evaluate_icu_time(make_records(val_rows_fold, val_hours.tolist(), prior_pred.tolist()))
        fold_metrics_prior_only.append(evaluated.metrics["median_absolute_error"])

        # A: prior as an additional feature.
        train_prior = apply_conditional_prior([rows[i] for i in np.where(train_mask)[0]], table)
        train_x_a = np.concatenate([x_all[train_mask], np.log1p(train_prior).reshape(-1, 1)], axis=1)
        val_x_a = np.concatenate([x_all[val_mask], np.log1p(prior_pred).reshape(-1, 1)], axis=1)
        model_a = _xgb_estimator({"objective": "reg:pseudohubererror", "max_depth": 4, "min_child_weight": 4, "learning_rate": 0.05, "subsample": 0.85, "colsample_bytree": 0.85, "reg_alpha": 1e-3, "reg_lambda": 1.0, "gamma": 0.0})
        model_a.fit(train_x_a, np.log1p(fit_hours), eval_set=[(val_x_a, np.log1p(val_hours))], verbose=False)
        pred_a = np.expm1(np.clip(model_a.predict(val_x_a), 0, None))
        evaluated_a = evaluate_icu_time(make_records(val_rows_fold, val_hours.tolist(), pred_a.tolist()))
        fold_metrics_a.append(evaluated_a.metrics["median_absolute_error"])

        # B: residual model in log space, target = log1p(true) - log1p(prior).
        residual_target = np.log1p(fit_hours) - np.log1p(train_prior)
        model_b = _xgb_estimator({"objective": "reg:pseudohubererror", "max_depth": 4, "min_child_weight": 4, "learning_rate": 0.05, "subsample": 0.85, "colsample_bytree": 0.85, "reg_alpha": 1e-3, "reg_lambda": 1.0, "gamma": 0.0})
        val_residual_target = np.log1p(val_hours) - np.log1p(prior_pred)
        model_b.fit(x_all[train_mask], residual_target, eval_set=[(x_all[val_mask], val_residual_target)], verbose=False)
        residual_pred = model_b.predict(x_all[val_mask])
        pred_b = np.expm1(np.clip(np.log1p(prior_pred) + residual_pred, 0, None))
        evaluated_b = evaluate_icu_time(make_records(val_rows_fold, val_hours.tolist(), pred_b.tolist()))
        fold_metrics_b.append(evaluated_b.metrics["median_absolute_error"])

    return {
        "prior_only": {"fold_metrics": fold_metrics_prior_only, "mean_median_ae": float(np.mean(fold_metrics_prior_only))},
        "A_prior_as_feature": {"fold_metrics": fold_metrics_a, "mean_median_ae": float(np.mean(fold_metrics_a))},
        "B_residual_model": {"fold_metrics": fold_metrics_b, "mean_median_ae": float(np.mean(fold_metrics_b))},
    }


def run_aft_cv(rows, hours, fold_assignment):
    fold_ids = np.asarray([fold_of(row, fold_assignment, "xgb_fold_5") for row in rows])
    hours_arr = np.asarray(hours, dtype=np.float64)
    x_all, names, _norm = build_xgb_matrix_single(rows, "B_FULL")
    configs = [
        {"max_depth": d, "aft_loss_distribution": dist, "aft_loss_distribution_scale": scale}
        for d in (3, 5) for dist in ("normal",) for scale in (0.8, 1.2)
    ][:6]
    cv_records = []
    for config in configs:
        fold_metrics = []
        for fold in range(N_FOLDS):
            train_mask, val_mask = fold_ids != fold, fold_ids == fold
            if val_mask.sum() < 10 or train_mask.sum() < 10:
                continue
            dtrain = xgb.DMatrix(x_all[train_mask], missing=np.nan)
            dtrain.set_float_info("label_lower_bound", hours_arr[train_mask])
            dtrain.set_float_info("label_upper_bound", hours_arr[train_mask])
            dval = xgb.DMatrix(x_all[val_mask], missing=np.nan)
            dval.set_float_info("label_lower_bound", hours_arr[val_mask])
            dval.set_float_info("label_upper_bound", hours_arr[val_mask])
            params = {"objective": "survival:aft", "eval_metric": "aft-nloglik", "max_depth": config["max_depth"], "aft_loss_distribution": config["aft_loss_distribution"], "aft_loss_distribution_scale": config["aft_loss_distribution_scale"], "seed": SEED}
            booster = xgb.train(params, dtrain, num_boost_round=400, evals=[(dval, "val")], early_stopping_rounds=30, verbose_eval=False)
            pred_hours = booster.predict(dval)
            val_rows_fold = [rows[i] for i in np.where(val_mask)[0]]
            evaluated = evaluate_icu_time(make_records(val_rows_fold, hours_arr[val_mask].tolist(), pred_hours.tolist()))
            fold_metrics.append(evaluated.metrics["median_absolute_error"])
        cv_records.append({"config": config, "fold_metrics": fold_metrics, "mean_median_ae": float(np.mean(fold_metrics)) if fold_metrics else float("nan")})
    cv_records.sort(key=lambda item: item["mean_median_ae"])
    return cv_records


def main() -> None:
    train_rows = load_dev_rows(ROOT, splits=("train",))
    validation_rows = load_dev_rows(ROOT, splits=("validation",))
    fold_assignment = load_fold_assignment()

    train_icu = icu_rows(train_rows)
    val_icu = icu_rows(validation_rows)
    train_hours = [icu_hours_target(r) for r in train_icu]
    val_hours = [icu_hours_target(r) for r in val_icu]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "finalist_validation_predictions").mkdir(parents=True, exist_ok=True)

    configs = sample_configs(HYPERPARAM_SPACE, N_CONFIGS, seed=SEED, extra_axes=EXTRA_AXES)
    xgb_cv = run_xgb_cv(train_icu, train_hours, fold_assignment, configs)
    print("ICU XGB best CV mean median AE:", xgb_cv[0]["mean_median_ae"], xgb_cv[0]["config"])

    prior_cv = run_conditional_prior_cv(train_icu, train_hours, fold_assignment)
    print("ICU conditional-prior CV:", prior_cv["prior_only"]["mean_median_ae"], prior_cv["A_prior_as_feature"]["mean_median_ae"], prior_cv["B_residual_model"]["mean_median_ae"])

    aft_cv = run_aft_cv(train_icu, train_hours, fold_assignment)
    print("ICU AFT best CV mean median AE:", aft_cv[0]["mean_median_ae"] if aft_cv else None)

    # V0 control.
    v0_config = dict(max_depth=5, min_child_weight=4, learning_rate=0.05, subsample=0.85, colsample_bytree=0.85, reg_alpha=1e-3, reg_lambda=1.0, gamma=0.0, objective="reg:pseudohubererror", weight_mode="uniform", feature_variant="V0")
    v0_x, v0_val_x, v0_names, v0_norm = build_xgb_matrices(train_icu, val_icu, "V0")
    model_v0 = _xgb_estimator(v0_config, n_estimators=400, early_stopping_rounds=None)
    model_v0.fit(v0_x, np.log1p(train_hours))
    v0_pred = np.expm1(np.clip(model_v0.predict(v0_val_x), 0, None))
    v0_eval = evaluate_icu_time(make_records(val_icu, val_hours, v0_pred.tolist()))

    # Finalists: top 3 XGB (log1p), best conditional-prior/residual system, best AFT.
    finalists = []
    for rank, record in enumerate(xgb_cv[:3]):
        config = record["config"]
        train_x, val_x, names, normalization = build_xgb_matrices(train_icu, val_icu, config["feature_variant"])
        sample_weight = _stay_balanced_sample_weight(train_icu) if config["weight_mode"] == "stay_balanced" else None
        model = _xgb_estimator(config, n_estimators=400, early_stopping_rounds=None)
        model.fit(train_x, np.log1p(train_hours), sample_weight=sample_weight)
        pred = np.expm1(np.clip(model.predict(val_x), 0, None))
        evaluated = evaluate_icu_time(make_records(val_icu, val_hours, pred.tolist()))
        pred_path = OUT_DIR / "finalist_validation_predictions" / f"icu_xgb_rank{rank+1}_{record['config_hash']}.json"
        pred_path.write_bytes(canonical_json_bytes({
            "status": "DEVELOPMENT_VALIDATION_ONLY", "task": "icu_stay_time", "family": "xgboost_log1p",
            "config_hash": record["config_hash"], "config": config,
            "rows": [{"stay_id": r["stay_id"], "prediction_time": r["prediction_time"], "target": float(t), "prediction": float(p)} for r, t, p in zip(val_icu, val_hours, pred)],
        }))
        finalists.append({"rank": rank + 1, "system": "xgb_log1p", "config_hash": record["config_hash"], "config": config, "cv_mean_median_ae": record["mean_median_ae"], "validation_metrics": evaluated.metrics, "validation_counts": evaluated.counts, "prediction_artifact": str(pred_path.relative_to(ROOT)), "prediction_artifact_sha256": sha256_file(pred_path)})

    # Conditional-prior / residual finalist: full-TRAIN fit, VALIDATION eval, use the best of {prior_only, A, B} per CV.
    full_table = fit_conditional_prior_table(train_icu, train_hours)
    prior_val_pred = apply_conditional_prior(val_icu, full_table)
    best_prior_mode = min(("prior_only", "A_prior_as_feature", "B_residual_model"), key=lambda mode: prior_cv[mode]["mean_median_ae"])
    if best_prior_mode == "prior_only":
        final_prior_pred = prior_val_pred
    else:
        train_prior_pred = apply_conditional_prior(train_icu, full_table)
        x_all_train, names_ctx, _norm = build_xgb_matrix_single(train_icu, "B_FULL")
        x_all_val, _n2, _norm2 = build_xgb_matrix_single(val_icu, "B_FULL")
        if best_prior_mode == "A_prior_as_feature":
            train_x_a = np.concatenate([x_all_train, np.log1p(train_prior_pred).reshape(-1, 1)], axis=1)
            val_x_a = np.concatenate([x_all_val, np.log1p(prior_val_pred).reshape(-1, 1)], axis=1)
            model_a = _xgb_estimator({"objective": "reg:pseudohubererror", "max_depth": 4, "min_child_weight": 4, "learning_rate": 0.05, "subsample": 0.85, "colsample_bytree": 0.85, "reg_alpha": 1e-3, "reg_lambda": 1.0, "gamma": 0.0}, n_estimators=400, early_stopping_rounds=None)
            model_a.fit(train_x_a, np.log1p(train_hours))
            final_prior_pred = np.expm1(np.clip(model_a.predict(val_x_a), 0, None))
        else:
            residual_target = np.log1p(train_hours) - np.log1p(train_prior_pred)
            model_b = _xgb_estimator({"objective": "reg:pseudohubererror", "max_depth": 4, "min_child_weight": 4, "learning_rate": 0.05, "subsample": 0.85, "colsample_bytree": 0.85, "reg_alpha": 1e-3, "reg_lambda": 1.0, "gamma": 0.0}, n_estimators=400, early_stopping_rounds=None)
            model_b.fit(x_all_train, residual_target)
            final_prior_pred = np.expm1(np.clip(np.log1p(prior_val_pred) + model_b.predict(x_all_val), 0, None))
    prior_evaluated = evaluate_icu_time(make_records(val_icu, val_hours, final_prior_pred.tolist()))
    prior_pred_path = OUT_DIR / "finalist_validation_predictions" / "icu_conditional_prior_finalist.json"
    prior_pred_path.write_bytes(canonical_json_bytes({"status": "DEVELOPMENT_VALIDATION_ONLY", "task": "icu_stay_time", "system": "conditional_prior_" + best_prior_mode, "rows": [{"stay_id": r["stay_id"], "prediction_time": r["prediction_time"], "target": float(t), "prediction": float(p)} for r, t, p in zip(val_icu, val_hours, final_prior_pred)]}))
    finalists.append({"rank": None, "system": "conditional_prior_" + best_prior_mode, "validation_metrics": prior_evaluated.metrics, "validation_counts": prior_evaluated.counts, "prediction_artifact": str(prior_pred_path.relative_to(ROOT)), "prediction_artifact_sha256": sha256_file(prior_pred_path)})

    # AFT finalist (best CV config), if competitive.
    aft_finalist = None
    if aft_cv:
        best_aft = aft_cv[0]
        x_all_train, names_ctx, _norm = build_xgb_matrix_single(train_icu, "B_FULL")
        x_all_val, _n2, _norm2 = build_xgb_matrix_single(val_icu, "B_FULL")
        dtrain = xgb.DMatrix(x_all_train, missing=np.nan)
        dtrain.set_float_info("label_lower_bound", np.asarray(train_hours))
        dtrain.set_float_info("label_upper_bound", np.asarray(train_hours))
        dval = xgb.DMatrix(x_all_val, missing=np.nan)
        params = {"objective": "survival:aft", "eval_metric": "aft-nloglik", "max_depth": best_aft["config"]["max_depth"], "aft_loss_distribution": best_aft["config"]["aft_loss_distribution"], "aft_loss_distribution_scale": best_aft["config"]["aft_loss_distribution_scale"], "seed": SEED}
        booster = xgb.train(params, dtrain, num_boost_round=400)
        aft_pred = booster.predict(dval)
        aft_evaluated = evaluate_icu_time(make_records(val_icu, val_hours, aft_pred.tolist()))
        aft_pred_path = OUT_DIR / "finalist_validation_predictions" / "icu_aft_finalist.json"
        aft_pred_path.write_bytes(canonical_json_bytes({"status": "DEVELOPMENT_VALIDATION_ONLY", "task": "icu_stay_time", "system": "aft", "config": best_aft["config"], "rows": [{"stay_id": r["stay_id"], "prediction_time": r["prediction_time"], "target": float(t), "prediction": float(p)} for r, t, p in zip(val_icu, val_hours, aft_pred)]}))
        aft_finalist = {"system": "aft", "config": best_aft["config"], "cv_mean_median_ae": best_aft["mean_median_ae"], "validation_metrics": aft_evaluated.metrics, "validation_counts": aft_evaluated.counts, "prediction_artifact": str(aft_pred_path.relative_to(ROOT)), "prediction_artifact_sha256": sha256_file(aft_pred_path)}
        finalists.append(aft_finalist)

    search_manifest = {"status": "DEVELOPMENT_VALIDATION_ONLY", "artifact_version": "performance_v2_phase2_icu_search_manifest_v1", "n_configs_searched_xgb": len(configs), "n_cv_folds": N_FOLDS, "hyperparameter_space": HYPERPARAM_SPACE, "extra_axes": EXTRA_AXES, "aft_configs_searched": len(aft_cv), "seed": SEED}
    (OUT_DIR / "search_manifest.json").write_bytes(canonical_json_bytes(search_manifest))
    (OUT_DIR / "cv_results.json").write_bytes(canonical_json_bytes({"status": "DEVELOPMENT_VALIDATION_ONLY", "xgb_log1p": xgb_cv, "conditional_prior": prior_cv, "aft": aft_cv}))
    (OUT_DIR / "conditional_prior_contract.json").write_bytes(canonical_json_bytes({"status": "DEVELOPMENT_VALIDATION_ONLY", "prior_version": full_table["prior_version"], "fold_safety": "fit_conditional_prior_table is called per-CV-fold using only that fold's TRAINING rows/labels (Part 7); for the VALIDATION fit it is fit from full TRAIN only.", "full_train_table": full_table, "best_mode": best_prior_mode}))
    (OUT_DIR / "finalist_metrics_xgb.json").write_bytes(canonical_json_bytes({"status": "DEVELOPMENT_VALIDATION_ONLY", "v0_control": {"config": v0_config, "validation_metrics": v0_eval.metrics, "validation_counts": v0_eval.counts}, "finalists": finalists}))
    print("wrote ICU (non-GRU) search artifacts to", OUT_DIR)


if __name__ == "__main__":
    main()
