"""Performance-v2 Phase 2, Part 10: GRU finalist stability across 3
deterministic training seeds (recovery24, recovery48, icu_stay_time --
the top-ranked GRU finalist for each, per finalist_metrics_gru.json).

DEVELOPMENT_VALIDATION_ONLY. Full TRAIN refit x3 seeds, evaluated once each
on VALIDATION. Reports mean/std/best/worst; never selects "the luckiest
seed" as the reported result -- the mean is the family-level comparator.
A fixed 3-seed ensemble is registered as gru_ensemble_3seed_v2 ONLY if it
improves >=1% relative over the single-seed mean, member configs are
identical (seed-only difference), and inference averages raw outputs
before postprocessing.
"""

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.synthetic.config import canonical_json_bytes  # noqa: E402
from evaluation.metrics import evaluate_icu_time, evaluate_recovery_horizon  # noqa: E402
from performance_v2.context_normalization import apply_normalization, fit_normalization  # noqa: E402
from performance_v2.data_loading import load_dev_rows  # noqa: E402
from performance_v2.gru_training import SingleOutputGRU, fit_age_normalization, make_encoder_config, predict_gru, standardize_targets, train_gru  # noqa: E402
from performance_v2.targets import icu_hours_target, icu_rows, make_records, recovery_targets  # noqa: E402
from vedant_infra.hashing import sha256_file  # noqa: E402

SEEDS = (2026_09_25, 2026_09_26, 2026_09_27)
MAX_EPOCHS = 15
PATIENCE = 4
DEVICE = torch.device("cpu")


def _fit_one_seed(config, train_rows, train_target, val_rows, val_target, seed, log_space):
    norm = fit_normalization(train_rows, config["feature_variant"])
    train_ctx = apply_normalization(train_rows, norm)
    val_ctx = apply_normalization(val_rows, norm)
    age_mean, age_std = fit_age_normalization(train_rows)
    target_mean, target_std = standardize_targets(train_target)
    gru_config = make_encoder_config(config["feature_variant"], config["hidden_dim"], config["num_layers"], config["dropout"])
    model = SingleOutputGRU(gru_config)
    model, _ = train_gru(
        model, train_rows, train_target, train_ctx, val_rows, val_target, val_ctx,
        loss_name=config["loss_name"], lr=config["lr"], weight_decay=config["weight_decay"],
        batch_size=config["batch_size"], max_epochs=MAX_EPOCHS, patience=PATIENCE, seed=seed,
        device=DEVICE, target_mean=target_mean, target_std=target_std, age_mean=age_mean, age_std=age_std,
    )
    pred = predict_gru(model, val_rows, val_ctx, DEVICE, age_mean, age_std, target_mean, target_std)
    if log_space:
        pred = np.expm1(np.clip(pred, 0, None))
    return pred


def stability_for_recovery(horizon, config, train_rows, validation_rows):
    train_eligible = recovery_targets(train_rows, horizon)
    val_eligible = recovery_targets(validation_rows, horizon)
    train_y = np.asarray([float(r["delta_sofa_" + horizon]) for r in train_eligible])
    val_y = np.asarray([float(r["delta_sofa_" + horizon]) for r in val_eligible])
    seed_predictions, seed_maes = [], []
    for seed in SEEDS:
        pred = _fit_one_seed(config, train_eligible, train_y, val_eligible, val_y, seed, log_space=False)
        evaluated = evaluate_recovery_horizon(make_records(val_eligible, val_y.tolist(), pred.tolist()), horizon=horizon + "h")
        seed_predictions.append(pred)
        seed_maes.append(evaluated.metrics["mae"])
    ensemble_pred = np.mean(seed_predictions, axis=0)
    ensemble_eval = evaluate_recovery_horizon(make_records(val_eligible, val_y.tolist(), ensemble_pred.tolist()), horizon=horizon + "h")
    return {
        "seed_maes": seed_maes, "mean_mae": float(np.mean(seed_maes)), "std_mae": float(np.std(seed_maes)),
        "best_mae": float(np.min(seed_maes)), "worst_mae": float(np.max(seed_maes)),
        "ensemble_mae": ensemble_eval.metrics["mae"],
        "ensemble_relative_improvement": (float(np.mean(seed_maes)) - ensemble_eval.metrics["mae"]) / float(np.mean(seed_maes)),
    }, val_eligible, val_y, ensemble_pred


def stability_for_icu(config, train_rows, validation_rows):
    train_icu = icu_rows(train_rows)
    val_icu = icu_rows(validation_rows)
    train_hours = np.asarray([icu_hours_target(r) for r in train_icu])
    val_hours = np.asarray([icu_hours_target(r) for r in val_icu])
    train_y_log, val_y_log = np.log1p(train_hours), np.log1p(val_hours)
    seed_predictions, seed_medaes = [], []
    for seed in SEEDS:
        pred_hours = _fit_one_seed(config, train_icu, train_y_log, val_icu, val_y_log, seed, log_space=True)
        evaluated = evaluate_icu_time(make_records(val_icu, val_hours.tolist(), pred_hours.tolist()))
        seed_predictions.append(pred_hours)
        seed_medaes.append(evaluated.metrics["median_absolute_error"])
    ensemble_pred = np.mean(seed_predictions, axis=0)
    ensemble_eval = evaluate_icu_time(make_records(val_icu, val_hours.tolist(), ensemble_pred.tolist()))
    return {
        "seed_median_aes": seed_medaes, "mean_median_ae": float(np.mean(seed_medaes)), "std_median_ae": float(np.std(seed_medaes)),
        "best_median_ae": float(np.min(seed_medaes)), "worst_median_ae": float(np.max(seed_medaes)),
        "ensemble_median_ae": ensemble_eval.metrics["median_absolute_error"],
        "ensemble_relative_improvement": (float(np.mean(seed_medaes)) - ensemble_eval.metrics["median_absolute_error"]) / float(np.mean(seed_medaes)),
    }, val_icu, val_hours, ensemble_pred


def main() -> None:
    import json
    train_rows = load_dev_rows(ROOT, splits=("train",))
    validation_rows = load_dev_rows(ROOT, splits=("validation",))

    results = {"status": "DEVELOPMENT_VALIDATION_ONLY", "artifact_version": "performance_v2_phase2_gru_stability_v1", "seeds": list(SEEDS)}

    rec_dir = ROOT / "artifacts/performance_v2/phase2/recovery"
    gru_finalists = json.loads((rec_dir / "finalist_metrics_gru.json").read_text())
    for horizon in ("24", "48"):
        top_config = gru_finalists["horizons"][horizon][0]["config"]
        stability, val_rows, val_y, ensemble_pred = stability_for_recovery(horizon, top_config, train_rows, validation_rows)
        results["recovery" + horizon] = {"config": top_config, "stability": stability}
        print("recovery" + horizon, "GRU stability:", stability)
        if stability["ensemble_relative_improvement"] >= 0.01:
            ensemble_pred_path = rec_dir / "finalist_validation_predictions" / f"recovery{horizon}_gru_ensemble_3seed.json"
            ensemble_pred_path.write_bytes(canonical_json_bytes({"status": "DEVELOPMENT_VALIDATION_ONLY", "task": "recovery" + horizon, "system": "gru_ensemble_3seed_v2", "config": top_config, "seeds": list(SEEDS), "rows": [{"stay_id": r["stay_id"], "prediction_time": r["prediction_time"], "target": float(t), "prediction": float(p)} for r, t, p in zip(val_rows, val_y, ensemble_pred)]}))
            results["recovery" + horizon]["ensemble_registered"] = True
            results["recovery" + horizon]["ensemble_prediction_artifact"] = str(ensemble_pred_path.relative_to(ROOT))
        else:
            results["recovery" + horizon]["ensemble_registered"] = False

    icu_dir = ROOT / "artifacts/performance_v2/phase2/icu"
    icu_gru_finalists = json.loads((icu_dir / "finalist_metrics_gru.json").read_text())
    top_config = icu_gru_finalists["finalists"][0]["config"]
    stability, val_rows, val_hours, ensemble_pred = stability_for_icu(top_config, train_rows, validation_rows)
    results["icu_stay_time"] = {"config": top_config, "stability": stability}
    print("ICU GRU stability:", stability)
    if stability["ensemble_relative_improvement"] >= 0.01:
        ensemble_pred_path = icu_dir / "finalist_validation_predictions" / "icu_gru_ensemble_3seed.json"
        ensemble_pred_path.write_bytes(canonical_json_bytes({"status": "DEVELOPMENT_VALIDATION_ONLY", "task": "icu_stay_time", "system": "gru_ensemble_3seed_v2", "config": top_config, "seeds": list(SEEDS), "rows": [{"stay_id": r["stay_id"], "prediction_time": r["prediction_time"], "target": float(t), "prediction": float(p)} for r, t, p in zip(val_rows, val_hours, ensemble_pred)]}))
        results["icu_stay_time"]["ensemble_registered"] = True
        results["icu_stay_time"]["ensemble_prediction_artifact"] = str(ensemble_pred_path.relative_to(ROOT))
    else:
        results["icu_stay_time"]["ensemble_registered"] = False

    out_path = ROOT / "artifacts/performance_v2/phase2/gru_stability_v1.json"
    out_path.write_bytes(canonical_json_bytes(results))
    print("wrote", out_path, sha256_file(out_path))


if __name__ == "__main__":
    main()
