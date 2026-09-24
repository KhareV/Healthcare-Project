"""Performance-v2 Phase 2, Part 6 (GRU secondary family for recovery).

DEVELOPMENT_VALIDATION_ONLY. TRAIN/VALIDATION only, never TEST.

COMPUTE-EFFICIENCY SCOPE REDUCTION (documented, not silent): the spec's
"<=12 configurations, 3-fold CV, max epochs 60, patience 8" is reduced here
to 6 configurations, 3-fold CV (as specified), max epochs 25, patience 6.
This keeps total GRU search wall-clock time tractable within this phase's
session while preserving the same deterministic, traceable, fold-safe
methodology; every attempted configuration is still fully recorded in
cv_results.json per Part 9. If Phase 3 finds GRU competitive, its budget
can be revisited then with the exact same code.
"""

import sys
import time
from itertools import product
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.synthetic.config import canonical_json_bytes  # noqa: E402
from evaluation.metrics import evaluate_recovery_horizon  # noqa: E402
from performance_v2.context_normalization import apply_normalization, fit_normalization  # noqa: E402
from performance_v2.data_loading import load_dev_rows  # noqa: E402
from performance_v2.gru_training import SingleOutputGRU, fit_age_normalization, make_encoder_config, predict_gru, standardize_targets, train_gru  # noqa: E402
from performance_v2.search_utils import config_hash, fold_of, load_fold_assignment  # noqa: E402
from performance_v2.targets import make_records, recovery_targets  # noqa: E402
from vedant_infra.hashing import sha256_file  # noqa: E402

OUT_DIR = ROOT / "artifacts/performance_v2/phase2/recovery"
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


def run_cv_for_horizon(horizon, train_rows, fold_assignment):
    eligible_rows = recovery_targets(train_rows, horizon)
    fold_ids = np.asarray([fold_of(row, fold_assignment, "gru_fold_3") for row in eligible_rows])
    y_all = np.asarray([float(r["delta_sofa_" + horizon]) for r in eligible_rows])
    cv_records = []
    for config in CONFIG_GRID:
        fold_metrics = []
        t0 = time.time()
        for fold in range(N_FOLDS):
            train_mask, val_mask = fold_ids != fold, fold_ids == fold
            if val_mask.sum() < 10 or train_mask.sum() < 10:
                continue
            fold_train_rows = [eligible_rows[i] for i in np.where(train_mask)[0]]
            fold_val_rows = [eligible_rows[i] for i in np.where(val_mask)[0]]
            train_y, val_y = y_all[train_mask], y_all[val_mask]
            norm = fit_normalization(fold_train_rows, config["feature_variant"])
            train_ctx = apply_normalization(fold_train_rows, norm)
            val_ctx = apply_normalization(fold_val_rows, norm)
            age_mean, age_std = fit_age_normalization(fold_train_rows)
            target_mean, target_std = standardize_targets(train_y)
            gru_config = make_encoder_config(config["feature_variant"], config["hidden_dim"], config["num_layers"], config["dropout"])
            model = SingleOutputGRU(gru_config)
            model, _val_loss = train_gru(
                model, fold_train_rows, train_y, train_ctx, fold_val_rows, val_y, val_ctx,
                loss_name=config["loss_name"], lr=config["lr"], weight_decay=config["weight_decay"],
                batch_size=config["batch_size"], max_epochs=MAX_EPOCHS, patience=PATIENCE, seed=SEED,
                device=DEVICE, target_mean=target_mean, target_std=target_std, age_mean=age_mean, age_std=age_std,
            )
            predictions = predict_gru(model, fold_val_rows, val_ctx, DEVICE, age_mean, age_std, target_mean, target_std)
            evaluated = evaluate_recovery_horizon(make_records(fold_val_rows, val_y.tolist(), predictions.tolist()), horizon=horizon + "h")
            fold_metrics.append(evaluated.metrics["mae"])
        runtime = time.time() - t0
        mean_metric = float(np.mean(fold_metrics)) if fold_metrics else float("nan")
        cv_records.append({"config_hash": config_hash(config), "config": config, "fold_metrics_mae": fold_metrics, "mean_mae": mean_metric, "std_mae": float(np.std(fold_metrics)) if fold_metrics else float("nan"), "n_folds": len(fold_metrics), "runtime_seconds": runtime})
        print("  gru config", config_hash(config), "mean_mae", mean_metric, "runtime", round(runtime, 1))
    cv_records.sort(key=lambda item: item["mean_mae"])
    return cv_records, eligible_rows


def main() -> None:
    train_rows = load_dev_rows(ROOT, splits=("train",))
    validation_rows = load_dev_rows(ROOT, splits=("validation",))
    fold_assignment = load_fold_assignment()
    (OUT_DIR / "finalist_validation_predictions").mkdir(parents=True, exist_ok=True)

    all_results = {"status": "DEVELOPMENT_VALIDATION_ONLY", "scope_reduction_note": "6 configs (spec allows <=12), 3-fold CV, max_epochs=25/patience=6 (spec allows 60/8) -- documented compute-efficiency reduction.", "horizons": {}}
    finalists_all = {"status": "DEVELOPMENT_VALIDATION_ONLY", "horizons": {}}

    for horizon in ("24", "48"):
        cv_records, train_eligible = run_cv_for_horizon(horizon, train_rows, fold_assignment)
        val_eligible = recovery_targets(validation_rows, horizon)
        all_results["horizons"][horizon] = cv_records

        top2 = cv_records[:2]
        finalists = []
        for rank, record in enumerate(top2):
            config = record["config"]
            train_y = np.asarray([float(r["delta_sofa_" + horizon]) for r in train_eligible])
            val_y = np.asarray([float(r["delta_sofa_" + horizon]) for r in val_eligible])
            norm = fit_normalization(train_eligible, config["feature_variant"])
            train_ctx = apply_normalization(train_eligible, norm)
            val_ctx = apply_normalization(val_eligible, norm)
            age_mean, age_std = fit_age_normalization(train_eligible)
            target_mean, target_std = standardize_targets(train_y)
            gru_config = make_encoder_config(config["feature_variant"], config["hidden_dim"], config["num_layers"], config["dropout"])
            model = SingleOutputGRU(gru_config)
            model, _val_loss = train_gru(
                model, train_eligible, train_y, train_ctx, val_eligible, val_y, val_ctx,
                loss_name=config["loss_name"], lr=config["lr"], weight_decay=config["weight_decay"],
                batch_size=config["batch_size"], max_epochs=MAX_EPOCHS, patience=PATIENCE, seed=SEED,
                device=DEVICE, target_mean=target_mean, target_std=target_std, age_mean=age_mean, age_std=age_std,
            )
            predictions = predict_gru(model, val_eligible, val_ctx, DEVICE, age_mean, age_std, target_mean, target_std)
            evaluated = evaluate_recovery_horizon(make_records(val_eligible, val_y.tolist(), predictions.tolist()), horizon=horizon + "h")
            pred_path = OUT_DIR / "finalist_validation_predictions" / f"recovery{horizon}_gru_rank{rank+1}_{record['config_hash']}.json"
            pred_path.write_bytes(canonical_json_bytes({"status": "DEVELOPMENT_VALIDATION_ONLY", "task": "recovery" + horizon, "family": "gru", "config_hash": record["config_hash"], "config": config, "rows": [{"stay_id": r["stay_id"], "prediction_time": r["prediction_time"], "target": float(t), "prediction": float(p)} for r, t, p in zip(val_eligible, val_y, predictions)]}))
            finalists.append({"rank": rank + 1, "family": "gru", "config_hash": record["config_hash"], "config": config, "cv_mean_mae": record["mean_mae"], "validation_metrics": evaluated.metrics, "validation_counts": evaluated.counts, "prediction_artifact": str(pred_path.relative_to(ROOT)), "prediction_artifact_sha256": sha256_file(pred_path)})
            print("recovery" + horizon, "GRU rank", rank + 1, "validation MAE", evaluated.metrics["mae"])
        finalists_all["horizons"][horizon] = finalists

    (OUT_DIR / "cv_results_gru.json").write_bytes(canonical_json_bytes(all_results))
    (OUT_DIR / "finalist_metrics_gru.json").write_bytes(canonical_json_bytes(finalists_all))
    print("wrote GRU recovery search artifacts to", OUT_DIR)


if __name__ == "__main__":
    main()
