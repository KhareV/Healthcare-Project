"""Performance-v2 Phase 3, Parts 9-10: final full-DEV XGBoost refit (4
tasks) using the exact frozen Phase-2 hyperparameters/objective/feature
variant/training-weight-mode and the Phase-3 fixed training budget, plus a
save/reload serialization-parity check. No early stopping against anything
(no holdout exists at this stage by design); no fresh-test access.
"""

import json
import sys
from pathlib import Path

import numpy as np
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.synthetic.config import canonical_json_bytes  # noqa: E402
from evaluation.weights import compute_stay_weights  # noqa: E402
from performance_v2.data_loading import load_dev_rows  # noqa: E402
from performance_v2.targets import icu_hours_target, icu_rows, recovery_targets, support_rows  # noqa: E402
from performance_v2.v2_features import build_xgb_matrix_single  # noqa: E402
from vedant_infra.hashing import sha256_file  # noqa: E402

OUT_DIR = ROOT / "artifacts/performance_v2/phase3/models"
SEED = 2026_09_25


def _stay_balanced_sample_weight(rows):
    result = compute_stay_weights([row["stay_id"] for row in rows], [True] * len(rows))
    return np.asarray(result.weights, dtype=np.float64)


def main() -> None:
    selection = json.loads((ROOT / "artifacts/performance_v2/phase2/validation_selection_v2.json").read_text())
    budget = json.loads((ROOT / "artifacts/performance_v2/phase3/final_training_budget_v1.json").read_text())

    train_rows = load_dev_rows(ROOT, splits=("train",))
    val_rows = load_dev_rows(ROOT, splits=("validation",))
    dev_rows = train_rows + val_rows

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest_entries = {}
    repro_results = {}

    # --- Recovery 24h / 48h ---
    for horizon, filename in (("24", "recovery24_xgb_v2.json"), ("48", "recovery48_xgb_v2.json")):
        task_key = "recovery" + horizon
        config = selection["tasks"][task_key]["hyperparameters"]
        n_estimators = budget["tasks"][task_key]["final_n_estimators"]
        rows = recovery_targets(dev_rows, horizon)
        x_all, names, normalization = build_xgb_matrix_single(rows, config["feature_variant"])
        y = np.asarray([float(r["delta_sofa_" + horizon]) for r in rows])
        sample_weight = _stay_balanced_sample_weight(rows) if config.get("weight_mode") == "stay_balanced" else None
        model = xgb.XGBRegressor(
            objective=config["objective"], random_state=SEED, missing=np.nan,
            max_depth=config["max_depth"], min_child_weight=config["min_child_weight"],
            learning_rate=config["learning_rate"], subsample=config["subsample"],
            colsample_bytree=config["colsample_bytree"], reg_alpha=config["reg_alpha"],
            reg_lambda=config["reg_lambda"], gamma=config["gamma"], n_estimators=n_estimators,
        )
        model.fit(x_all, y, sample_weight=sample_weight)
        model_path = OUT_DIR / filename
        model.save_model(str(model_path))
        model_sha256 = sha256_file(model_path)

        fixture_x = x_all[:20]
        pre_save_predictions = model.predict(fixture_x)
        reloaded = xgb.XGBRegressor()
        reloaded.load_model(str(model_path))
        post_load_predictions = reloaded.predict(fixture_x)
        matches = bool(np.allclose(pre_save_predictions, post_load_predictions, atol=1e-6))
        repro_results[task_key] = {"matches": matches, "max_abs_diff": float(np.max(np.abs(pre_save_predictions - post_load_predictions)))}

        manifest_entries[task_key] = {
            "family": "xgboost", "feature_variant": config["feature_variant"], "objective": config["objective"],
            "training_weight_mode": config.get("weight_mode", "uniform"), "hyperparameters": config,
            "final_n_estimators": n_estimators, "n_dev_rows": len(rows),
            "model_path": str(model_path.relative_to(ROOT)), "model_sha256": model_sha256,
            "feature_names_count": len(names), "context_normalization_ref": f"artifacts/performance_v2/phase3/preprocessing/context_normalization_v2_{config['feature_variant']}.json",
        }
        print(task_key, "trained, n_estimators", n_estimators, "n_rows", len(rows), "model_sha256", model_sha256, "reload_matches", matches)

    # --- ICU ---
    icu_config = selection["tasks"]["icu_stay_time"]["config"]
    icu_n_estimators = budget["tasks"]["icu_stay_time"]["final_n_estimators"]
    icu_train_rows = icu_rows(dev_rows)
    x_all, names, normalization = build_xgb_matrix_single(icu_train_rows, icu_config["feature_variant"])
    hours = np.asarray([icu_hours_target(r) for r in icu_train_rows])
    y_log = np.log1p(hours)
    sample_weight = _stay_balanced_sample_weight(icu_train_rows) if icu_config.get("weight_mode") == "stay_balanced" else None
    model = xgb.XGBRegressor(
        objective=icu_config["objective"], random_state=SEED, missing=np.nan,
        max_depth=icu_config["max_depth"], min_child_weight=icu_config["min_child_weight"],
        learning_rate=icu_config["learning_rate"], subsample=icu_config["subsample"],
        colsample_bytree=icu_config["colsample_bytree"], reg_alpha=icu_config["reg_alpha"],
        reg_lambda=icu_config["reg_lambda"], gamma=icu_config["gamma"], n_estimators=icu_n_estimators,
    )
    model.fit(x_all, y_log, sample_weight=sample_weight)
    model_path = OUT_DIR / "icu_time_xgb_v2.json"
    model.save_model(str(model_path))
    model_sha256 = sha256_file(model_path)
    fixture_x = x_all[:20]
    pre_save_predictions = model.predict(fixture_x)
    reloaded = xgb.XGBRegressor()
    reloaded.load_model(str(model_path))
    post_load_predictions = reloaded.predict(fixture_x)
    matches = bool(np.allclose(pre_save_predictions, post_load_predictions, atol=1e-6))
    repro_results["icu_stay_time"] = {"matches": matches, "max_abs_diff": float(np.max(np.abs(pre_save_predictions - post_load_predictions)))}
    manifest_entries["icu_stay_time"] = {
        "family": "xgboost", "feature_variant": icu_config["feature_variant"], "objective": icu_config["objective"],
        "training_weight_mode": icu_config.get("weight_mode", "uniform"), "hyperparameters": icu_config,
        "final_n_estimators": icu_n_estimators, "n_dev_rows": len(icu_train_rows),
        "raw_output_meaning": "log1p_remaining_current_icu_hours",
        "postprocess": "expm1(clamp_min(raw,0))",
        "model_path": str(model_path.relative_to(ROOT)), "model_sha256": model_sha256,
        "feature_names_count": len(names), "context_normalization_ref": f"artifacts/performance_v2/phase3/preprocessing/context_normalization_v2_{icu_config['feature_variant']}.json",
    }
    print("icu_stay_time trained, n_estimators", icu_n_estimators, "n_rows", len(icu_train_rows), "model_sha256", model_sha256, "reload_matches", matches)

    # --- Support ---
    support_config = selection["tasks"]["organ_support"]["hyperparameters"]
    support_n_estimators = budget["tasks"]["organ_support"]["final_n_estimators"]
    support_train_rows = support_rows(dev_rows)
    x_all, names, normalization = build_xgb_matrix_single(support_train_rows, support_config["feature_variant"])
    labels = np.asarray([float(r["organ_support_label"]) for r in support_train_rows])
    model = xgb.XGBClassifier(
        objective="binary:logistic", random_state=SEED, missing=np.nan,
        max_depth=support_config["max_depth"], min_child_weight=support_config["min_child_weight"],
        learning_rate=support_config["learning_rate"], subsample=support_config["subsample"],
        colsample_bytree=support_config["colsample_bytree"], reg_alpha=support_config["reg_alpha"],
        reg_lambda=support_config["reg_lambda"], gamma=support_config["gamma"], n_estimators=support_n_estimators,
    )
    model.fit(x_all, labels)
    model_path = OUT_DIR / "organ_support_xgb_v2.json"
    model.save_model(str(model_path))
    model_sha256 = sha256_file(model_path)
    fixture_x = x_all[:20]
    pre_save_predictions = model.predict_proba(fixture_x)[:, 1]
    reloaded = xgb.XGBClassifier()
    reloaded.load_model(str(model_path))
    post_load_predictions = reloaded.predict_proba(fixture_x)[:, 1]
    matches = bool(np.allclose(pre_save_predictions, post_load_predictions, atol=1e-6))
    repro_results["organ_support"] = {"matches": matches, "max_abs_diff": float(np.max(np.abs(pre_save_predictions - post_load_predictions)))}
    manifest_entries["organ_support"] = {
        "family": "xgboost", "feature_variant": support_config["feature_variant"],
        "hyperparameters": support_config, "final_n_estimators": support_n_estimators, "n_dev_rows": len(support_train_rows),
        "model_path": str(model_path.relative_to(ROOT)), "model_sha256": model_sha256,
        "feature_names_count": len(names), "context_normalization_ref": f"artifacts/performance_v2/phase3/preprocessing/context_normalization_v2_{support_config['feature_variant']}.json",
    }
    print("organ_support trained, n_estimators", support_n_estimators, "n_rows", len(support_train_rows), "model_sha256", model_sha256, "reload_matches", matches)

    assert all(r["matches"] for r in repro_results.values()), "serialization parity check failed"

    manifest_path = OUT_DIR / "final_refit_manifest_v1.json"
    manifest_path.write_bytes(canonical_json_bytes({"status": "FINAL_DEV_MODELS_TRAINED", "tasks": manifest_entries, "reproducibility_check": repro_results}))
    print("wrote", manifest_path, sha256_file(manifest_path))
    print("ALL_RELOAD_CHECKS_PASS:", all(r["matches"] for r in repro_results.values()))


if __name__ == "__main__":
    main()
