"""Performance-v2 Phase 2, Part 7 (GRU primary deep-sequence candidate for
ICU). Ranked on postprocessed stay-balanced median AE hours, not raw
log-space loss (training loss is Huber/L1 in log1p space; ranking always
applies expm1(clamp_min(x,0)) first).

DEVELOPMENT_VALIDATION_ONLY. TRAIN/VALIDATION only, never TEST.

Same documented compute-efficiency scope reduction as the recovery GRU
script: 6 configs (spec allows <=12), 3-fold CV (as specified), max_epochs
25/patience 6 (spec allows 60/8).
"""

import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.synthetic.config import canonical_json_bytes  # noqa: E402
from evaluation.metrics import evaluate_icu_time  # noqa: E402
from performance_v2.context_normalization import apply_normalization, fit_normalization  # noqa: E402
from performance_v2.data_loading import load_dev_rows  # noqa: E402
from performance_v2.gru_training import SingleOutputGRU, fit_age_normalization, make_encoder_config, predict_gru, standardize_targets, train_gru  # noqa: E402
from performance_v2.search_utils import config_hash, fold_of, load_fold_assignment  # noqa: E402
from performance_v2.targets import icu_hours_target, icu_rows, make_records  # noqa: E402
from vedant_infra.hashing import sha256_file  # noqa: E402

OUT_DIR = ROOT / "artifacts/performance_v2/phase2/icu"
SEED = 2026_09_25
N_FOLDS = 3
MAX_EPOCHS = 15
PATIENCE = 4
DEVICE = torch.device("cpu")

CONFIG_GRID = [
    {"hidden_dim": h, "num_layers": l, "dropout": d, "lr": lr, "batch_size": b, "weight_decay": wd, "loss_name": loss, "feature_variant": variant}
    for h, l, d, lr, b, wd, loss, variant in [
        (64, 1, 0.15, 5e-4, 64, 1e-5, "huber", "B_FULL"),
        (128, 1, 0.20, 3e-4, 64, 1e-5, "huber", "B_FULL"),
        (64, 2, 0.25, 5e-4, 128, 1e-4, "l1", "B_FULL"),
        (128, 2, 0.20, 3e-4, 128, 1e-5, "huber", "B_MIN"),
        (192, 1, 0.15, 2e-4, 64, 1e-4, "huber", "B_PLUS_F"),
        (64, 1, 0.10, 1e-3, 128, 1e-6, "l1", "B_MIN"),
    ]
]


def run_cv(train_rows_icu, hours, fold_assignment):
    fold_ids = np.asarray([fold_of(row, fold_assignment, "gru_fold_3") for row in train_rows_icu])
    log_targets = np.log1p(np.asarray(hours))
    cv_records = []
    for config in CONFIG_GRID:
        fold_metrics = []
        t0 = time.time()
        for fold in range(N_FOLDS):
            train_mask, val_mask = fold_ids != fold, fold_ids == fold
            if val_mask.sum() < 10 or train_mask.sum() < 10:
                continue
            fold_train_rows = [train_rows_icu[i] for i in np.where(train_mask)[0]]
            fold_val_rows = [train_rows_icu[i] for i in np.where(val_mask)[0]]
            train_y_log, val_y_log = log_targets[train_mask], log_targets[val_mask]
            norm = fit_normalization(fold_train_rows, config["feature_variant"])
            train_ctx = apply_normalization(fold_train_rows, norm)
            val_ctx = apply_normalization(fold_val_rows, norm)
            age_mean, age_std = fit_age_normalization(fold_train_rows)
            target_mean, target_std = standardize_targets(train_y_log)
            gru_config = make_encoder_config(config["feature_variant"], config["hidden_dim"], config["num_layers"], config["dropout"])
            model = SingleOutputGRU(gru_config)
            model, _val_loss = train_gru(
                model, fold_train_rows, train_y_log, train_ctx, fold_val_rows, val_y_log, val_ctx,
                loss_name=config["loss_name"], lr=config["lr"], weight_decay=config["weight_decay"],
                batch_size=config["batch_size"], max_epochs=MAX_EPOCHS, patience=PATIENCE, seed=SEED,
                device=DEVICE, target_mean=target_mean, target_std=target_std, age_mean=age_mean, age_std=age_std,
            )
            pred_log = predict_gru(model, fold_val_rows, val_ctx, DEVICE, age_mean, age_std, target_mean, target_std)
            pred_hours = np.expm1(np.clip(pred_log, 0, None))
            val_hours = np.expm1(val_y_log)
            evaluated = evaluate_icu_time(make_records(fold_val_rows, val_hours.tolist(), pred_hours.tolist()))
            fold_metrics.append(evaluated.metrics["median_absolute_error"])
        runtime = time.time() - t0
        mean_metric = float(np.mean(fold_metrics)) if fold_metrics else float("nan")
        cv_records.append({"config_hash": config_hash(config), "config": config, "fold_metrics_median_ae": fold_metrics, "mean_median_ae": mean_metric, "std_median_ae": float(np.std(fold_metrics)) if fold_metrics else float("nan"), "n_folds": len(fold_metrics), "runtime_seconds": runtime})
        print("  gru config", config_hash(config), "mean_median_ae", mean_metric, "runtime", round(runtime, 1))
    cv_records.sort(key=lambda item: item["mean_median_ae"])
    return cv_records


def main() -> None:
    train_rows = load_dev_rows(ROOT, splits=("train",))
    validation_rows = load_dev_rows(ROOT, splits=("validation",))
    fold_assignment = load_fold_assignment()
    (OUT_DIR / "finalist_validation_predictions").mkdir(parents=True, exist_ok=True)

    train_icu = icu_rows(train_rows)
    val_icu = icu_rows(validation_rows)
    train_hours = [icu_hours_target(r) for r in train_icu]
    val_hours = np.asarray([icu_hours_target(r) for r in val_icu])

    cv_records = run_cv(train_icu, train_hours, fold_assignment)
    print("ICU GRU best CV mean median AE:", cv_records[0]["mean_median_ae"])

    top2 = cv_records[:2]
    finalists = []
    for rank, record in enumerate(top2):
        config = record["config"]
        train_y_log = np.log1p(np.asarray(train_hours))
        val_y_log = np.log1p(val_hours)
        norm = fit_normalization(train_icu, config["feature_variant"])
        train_ctx = apply_normalization(train_icu, norm)
        val_ctx = apply_normalization(val_icu, norm)
        age_mean, age_std = fit_age_normalization(train_icu)
        target_mean, target_std = standardize_targets(train_y_log)
        gru_config = make_encoder_config(config["feature_variant"], config["hidden_dim"], config["num_layers"], config["dropout"])
        model = SingleOutputGRU(gru_config)
        model, _val_loss = train_gru(
            model, train_icu, train_y_log, train_ctx, val_icu, val_y_log, val_ctx,
            loss_name=config["loss_name"], lr=config["lr"], weight_decay=config["weight_decay"],
            batch_size=config["batch_size"], max_epochs=MAX_EPOCHS, patience=PATIENCE, seed=SEED,
            device=DEVICE, target_mean=target_mean, target_std=target_std, age_mean=age_mean, age_std=age_std,
        )
        pred_log = predict_gru(model, val_icu, val_ctx, DEVICE, age_mean, age_std, target_mean, target_std)
        pred_hours = np.expm1(np.clip(pred_log, 0, None))
        evaluated = evaluate_icu_time(make_records(val_icu, val_hours.tolist(), pred_hours.tolist()))
        pred_path = OUT_DIR / "finalist_validation_predictions" / f"icu_gru_rank{rank+1}_{record['config_hash']}.json"
        pred_path.write_bytes(canonical_json_bytes({"status": "DEVELOPMENT_VALIDATION_ONLY", "task": "icu_stay_time", "family": "gru", "config_hash": record["config_hash"], "config": config, "rows": [{"stay_id": r["stay_id"], "prediction_time": r["prediction_time"], "target": float(t), "prediction": float(p)} for r, t, p in zip(val_icu, val_hours, pred_hours)]}))
        finalists.append({"rank": rank + 1, "family": "gru", "config_hash": record["config_hash"], "config": config, "cv_mean_median_ae": record["mean_median_ae"], "validation_metrics": evaluated.metrics, "validation_counts": evaluated.counts, "prediction_artifact": str(pred_path.relative_to(ROOT)), "prediction_artifact_sha256": sha256_file(pred_path)})
        print("ICU GRU rank", rank + 1, "validation median AE", evaluated.metrics["median_absolute_error"])

    (OUT_DIR / "cv_results_gru.json").write_bytes(canonical_json_bytes({"status": "DEVELOPMENT_VALIDATION_ONLY", "scope_reduction_note": "6 configs (spec allows <=12), 3-fold CV, max_epochs=25/patience=6 (spec allows 60/8).", "records": cv_records}))
    (OUT_DIR / "finalist_metrics_gru.json").write_bytes(canonical_json_bytes({"status": "DEVELOPMENT_VALIDATION_ONLY", "finalists": finalists}))
    print("wrote GRU ICU search artifacts to", OUT_DIR)


if __name__ == "__main__":
    main()
