"""Performance-v2 Phase 2, Part 8: SUPPORT V2. XGBoost only (no GRU/LSTM/
Transformer). Model selection only -- calibration/threshold deferred to
Phase 3 (requires_v2_recalibration=true is recorded, never frozen here).

DEVELOPMENT_VALIDATION_ONLY. TRAIN/VALIDATION only, never TEST. Preserves
support eligibility/label/class-weight semantics, event definition, horizon,
prevalence exactly as v1 defines them (this script only changes the
feature representation and hyperparameters, never the label/eligibility
logic).
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
from evaluation.metrics import evaluate_organ_support  # noqa: E402
from performance_v2.data_loading import load_dev_rows  # noqa: E402
from performance_v2.search_utils import config_hash, fold_of, load_fold_assignment, sample_configs  # noqa: E402
from performance_v2.targets import make_records, support_rows  # noqa: E402
from performance_v2.v2_features import build_xgb_matrices, build_xgb_matrix_single  # noqa: E402
from vedant_infra.hashing import sha256_file  # noqa: E402

OUT_DIR = ROOT / "artifacts/performance_v2/phase2/support"
SEED = 2026_09_25
N_CONFIGS = 20
N_FOLDS = 5
HYPERPARAM_SPACE = {
    "max_depth": (2, 8, "int"), "min_child_weight": (1, 16, "int"),
    "learning_rate": (0.01, 0.12, "log"), "subsample": (0.65, 1.0, "linear"),
    "colsample_bytree": (0.65, 1.0, "linear"), "reg_alpha": (1e-7, 2, "log"),
    "reg_lambda": (0.05, 30, "log"), "gamma": (0.0, 3.0, "linear"),
}
EXTRA_AXES = {"feature_variant": ["B_MIN", "B_FULL"]}


def _xgb_classifier(config, n_estimators=1200, early_stopping_rounds=25):
    return xgb.XGBClassifier(
        objective="binary:logistic", random_state=SEED, missing=np.nan, eval_metric="aucpr",
        max_depth=config["max_depth"], min_child_weight=config["min_child_weight"],
        learning_rate=config["learning_rate"], subsample=config["subsample"],
        colsample_bytree=config["colsample_bytree"], reg_alpha=config["reg_alpha"],
        reg_lambda=config["reg_lambda"], gamma=config["gamma"],
        n_estimators=n_estimators,
        **({"early_stopping_rounds": early_stopping_rounds} if early_stopping_rounds else {}),
    )


def run_cv(rows, labels, fold_assignment, configs):
    fold_ids = np.asarray([fold_of(row, fold_assignment, "xgb_fold_5") for row in rows])
    y_all = np.asarray(labels, dtype=np.float64)
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
            train_y, val_y = y_all[train_mask], y_all[val_mask]
            model = _xgb_classifier(config)
            model.fit(train_x, train_y, eval_set=[(val_x, val_y)], verbose=False)
            predictions = model.predict_proba(val_x)[:, 1]
            fold_rows_val = [rows[i] for i in np.where(val_mask)[0]]
            evaluated = evaluate_organ_support(make_records(fold_rows_val, val_y.tolist(), predictions.tolist()))
            fold_metrics.append(evaluated.metrics["auprc"])
        runtime = time.time() - t0
        mean_metric = float(np.mean(fold_metrics)) if fold_metrics else float("nan")
        std_metric = float(np.std(fold_metrics)) if fold_metrics else float("nan")
        cv_records.append({"config_hash": config_hash(config), "config": config, "fold_metrics_auprc": fold_metrics, "mean_auprc": mean_metric, "std_auprc": std_metric, "n_folds": len(fold_metrics), "runtime_seconds": runtime})
    cv_records.sort(key=lambda item: -item["mean_auprc"])
    return cv_records


def main() -> None:
    train_rows = load_dev_rows(ROOT, splits=("train",))
    validation_rows = load_dev_rows(ROOT, splits=("validation",))
    fold_assignment = load_fold_assignment()

    train_sup = support_rows(train_rows)
    val_sup = support_rows(validation_rows)
    train_y = [float(r["organ_support_label"]) for r in train_sup]
    val_y = [float(r["organ_support_label"]) for r in val_sup]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "finalist_validation_predictions").mkdir(parents=True, exist_ok=True)

    configs = sample_configs(HYPERPARAM_SPACE, N_CONFIGS, seed=SEED, extra_axes=EXTRA_AXES)
    cv_records = run_cv(train_sup, train_y, fold_assignment, configs)
    print("support best CV mean AUPRC (B_MIN/B_FULL):", cv_records[0]["mean_auprc"], cv_records[0]["config"]["feature_variant"])

    # Bounded check: is B_PLUS_F worth it? One quick CV pass at the winning hyperparams.
    best_hparams = {k: cv_records[0]["config"][k] for k in HYPERPARAM_SPACE}
    plus_f_config = dict(best_hparams, feature_variant="B_PLUS_F")
    plus_f_cv = run_cv(train_sup, train_y, fold_assignment, [plus_f_config])
    b_plus_f_gain = plus_f_cv[0]["mean_auprc"] - cv_records[0]["mean_auprc"]
    print("B_PLUS_F vs best B_MIN/B_FULL CV AUPRC delta:", b_plus_f_gain)
    variant_pool = list(cv_records)
    include_plus_f = b_plus_f_gain > 0.003
    if include_plus_f:
        variant_pool = sorted(variant_pool + plus_f_cv, key=lambda item: -item["mean_auprc"])

    # V0 control.
    v0_config = dict(max_depth=5, min_child_weight=4, learning_rate=0.05, subsample=0.85, colsample_bytree=0.85, reg_alpha=1e-3, reg_lambda=1.0, gamma=0.0, feature_variant="V0")
    v0_x, v0_val_x, v0_names, v0_norm = build_xgb_matrices(train_sup, val_sup, "V0")
    model_v0 = _xgb_classifier(v0_config, n_estimators=400, early_stopping_rounds=None)
    model_v0.fit(v0_x, train_y)
    v0_pred = model_v0.predict_proba(v0_val_x)[:, 1]
    v0_eval = evaluate_organ_support(make_records(val_sup, val_y, v0_pred.tolist()))

    finalists = []
    for rank, record in enumerate(variant_pool[:3]):
        config = record["config"]
        train_x, val_x, names, normalization = build_xgb_matrices(train_sup, val_sup, config["feature_variant"])
        model = _xgb_classifier(config, n_estimators=400, early_stopping_rounds=None)
        model.fit(train_x, train_y)
        predictions = model.predict_proba(val_x)[:, 1]
        evaluated = evaluate_organ_support(make_records(val_sup, val_y, predictions.tolist()))
        pred_path = OUT_DIR / "finalist_validation_predictions" / f"support_xgb_rank{rank+1}_{record['config_hash']}.json"
        pred_path.write_bytes(canonical_json_bytes({
            "status": "DEVELOPMENT_VALIDATION_ONLY", "task": "organ_support", "family": "xgboost",
            "config_hash": record["config_hash"], "config": config,
            "rows": [{"stay_id": r["stay_id"], "prediction_time": r["prediction_time"], "target": float(t), "prediction": float(p)} for r, t, p in zip(val_sup, val_y, predictions)],
        }))
        finalists.append({"rank": rank + 1, "config_hash": record["config_hash"], "config": config, "cv_mean_auprc": record["mean_auprc"], "validation_metrics": evaluated.metrics, "validation_counts": evaluated.counts, "prediction_artifact": str(pred_path.relative_to(ROOT)), "prediction_artifact_sha256": sha256_file(pred_path)})

    search_manifest = {"status": "DEVELOPMENT_VALIDATION_ONLY", "artifact_version": "performance_v2_phase2_support_search_manifest_v1", "n_configs_searched": len(configs), "n_cv_folds": N_FOLDS, "hyperparameter_space": HYPERPARAM_SPACE, "extra_axes": EXTRA_AXES, "b_plus_f_probe_delta_auprc": b_plus_f_gain, "b_plus_f_included": include_plus_f, "seed": SEED}
    (OUT_DIR / "search_manifest.json").write_bytes(canonical_json_bytes(search_manifest))
    (OUT_DIR / "cv_results.json").write_bytes(canonical_json_bytes({"status": "DEVELOPMENT_VALIDATION_ONLY", "b_min_b_full": cv_records, "b_plus_f_probe": plus_f_cv}))
    (OUT_DIR / "finalist_metrics.json").write_bytes(canonical_json_bytes({"status": "DEVELOPMENT_VALIDATION_ONLY", "v0_control": {"config": v0_config, "validation_metrics": v0_eval.metrics, "validation_counts": v0_eval.counts}, "finalists": finalists, "requires_v2_recalibration": True, "calibration_note": "Phase 2 is model selection only; final support calibrator/threshold are deferred to Phase 3's governed cross-fitted DEV calibration strategy."}))
    print("wrote support search artifacts to", OUT_DIR)


if __name__ == "__main__":
    main()
