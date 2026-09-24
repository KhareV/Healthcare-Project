"""Performance-v2 Phase 1, Part 8: LATENT_ORACLE.

NON_SERVING_LATENT_ORACLE / NEVER_ALLOWED_IN_PRODUCTION_FEATURES.

Uses the frozen generator's own subject-level latent state (z0, equilibrium,
support_propensity from data.synthetic.latent_state.initial_state) as
features -- reconstructed deterministically from the generator config plus
each subject's own identity/intime (never from any TEST row; TRAIN/
VALIDATION subjects only). This state is fixed at generation time before
any observation is drawn and is NEVER available to a served model. Trained
on TRAIN, evaluated on VALIDATION only, exactly like the observable oracle.
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
from xgboost import XGBClassifier, XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.synthetic.config import canonical_json_bytes, load_runtime_config  # noqa: E402
from data.synthetic.latent_state import initial_state  # noqa: E402
from evaluation.metrics import evaluate_icu_time, evaluate_organ_support, evaluate_recovery_horizon  # noqa: E402
from performance_v2.data_loading import load_dev_rows  # noqa: E402
from performance_v2.targets import icu_hours_target, icu_rows, make_records, recovery_targets, support_rows  # noqa: E402
from vedant_infra.hashing import sha256_file  # noqa: E402

OUT_DIR = ROOT / "artifacts/performance_v2/phase1/latent_oracle"
SEED = 2026_09_24


def _latent_features(row, config, condition_groups):
    subject_ordinal = int(row["subject_id"][-8:])
    group_index = condition_groups.index(row["cardiac_condition_group"])
    z0, equilibrium, support_propensity = initial_state(config, subject_ordinal, group_index)
    return np.concatenate([z0, equilibrium, [support_propensity, row["elapsed_episode_hours_at_t"]]]).astype(np.float32)


FEATURE_NAMES = [
    "z0_systemic_severity", "z0_cardiac_stress", "z0_respiratory_burden", "z0_renal_burden", "z0_neurologic_burden",
    "equilibrium_systemic_severity", "equilibrium_cardiac_stress", "equilibrium_respiratory_burden", "equilibrium_renal_burden", "equilibrium_neurologic_burden",
    "support_propensity", "elapsed_episode_hours_at_t",
]


def _build_matrix(rows, config, condition_groups):
    return np.stack([_latent_features(row, config, condition_groups) for row in rows])


def main() -> None:
    config = load_runtime_config(ROOT / "configs/synthetic/final_benchmark_v1.json", ROOT)
    condition_groups = config.values["population"]["condition_groups"]

    train_rows = load_dev_rows(ROOT, splits=("train",))
    validation_rows = load_dev_rows(ROOT, splits=("validation",))

    summary = {
        "status": "NON_SERVING_LATENT_ORACLE",
        "never_allowed_in_production_features": True,
        "artifact_version": "performance_v2_phase1_latent_oracle_v1",
        "feature_names": FEATURE_NAMES,
        "feature_source": "data.synthetic.latent_state.initial_state(config, subject_ordinal, group_index) -- subject-level z0/equilibrium/support_propensity, fixed once at generation time, never available to any served model.",
    }

    # Recovery
    for horizon in ("24", "48"):
        train_h = recovery_targets(train_rows, horizon)
        val_h = recovery_targets(validation_rows, horizon)
        train_x = _build_matrix(train_h, config, condition_groups)
        val_x = _build_matrix(val_h, config, condition_groups)
        train_y = np.asarray([float(r["delta_sofa_" + horizon]) for r in train_h])
        val_y = np.asarray([float(r["delta_sofa_" + horizon]) for r in val_h])
        model = XGBRegressor(objective="reg:absoluteerror", random_state=SEED, max_depth=4, n_estimators=300, learning_rate=0.05, subsample=0.8, colsample_bytree=0.9)
        model.fit(train_x, train_y)
        predictions = model.predict(val_x)
        result = evaluate_recovery_horizon(make_records(val_h, val_y.tolist(), predictions.tolist()), horizon=horizon + "h")
        summary["recovery" + horizon] = {"metrics": result.metrics, "counts": result.counts}
        (OUT_DIR / f"recovery{horizon}_validation_predictions_v1.json").parent.mkdir(parents=True, exist_ok=True)
        (OUT_DIR / f"recovery{horizon}_validation_predictions_v1.json").write_bytes(canonical_json_bytes({
            "status": "NON_SERVING_LATENT_ORACLE", "task": "recovery" + horizon,
            "rows": [{"stay_id": r["stay_id"], "prediction_time": r["prediction_time"], "target": float(t), "prediction": float(p)} for r, t, p in zip(val_h, val_y, predictions)],
        }))

    # ICU
    train_icu = icu_rows(train_rows)
    val_icu = icu_rows(validation_rows)
    train_x = _build_matrix(train_icu, config, condition_groups)
    val_x = _build_matrix(val_icu, config, condition_groups)
    train_hours = np.asarray([icu_hours_target(r) for r in train_icu])
    val_hours = np.asarray([icu_hours_target(r) for r in val_icu])
    train_log1p = np.log1p(train_hours)
    model = XGBRegressor(objective="reg:absoluteerror", random_state=SEED, max_depth=4, n_estimators=300, learning_rate=0.05, subsample=0.8, colsample_bytree=0.9)
    model.fit(train_x, train_log1p)
    predictions = np.expm1(np.clip(model.predict(val_x), 0, None))
    result = evaluate_icu_time(make_records(val_icu, val_hours.tolist(), predictions.tolist()))
    summary["icu_stay_time"] = {"metrics": result.metrics, "counts": result.counts}
    (OUT_DIR / "icu_stay_time_validation_predictions_v1.json").write_bytes(canonical_json_bytes({
        "status": "NON_SERVING_LATENT_ORACLE", "task": "icu_stay_time",
        "rows": [{"stay_id": r["stay_id"], "prediction_time": r["prediction_time"], "target": float(t), "prediction": float(p)} for r, t, p in zip(val_icu, val_hours, predictions)],
    }))

    # Support
    train_sup = support_rows(train_rows)
    val_sup = support_rows(validation_rows)
    train_x = _build_matrix(train_sup, config, condition_groups)
    val_x = _build_matrix(val_sup, config, condition_groups)
    train_y = [float(r["organ_support_label"]) for r in train_sup]
    val_y = [float(r["organ_support_label"]) for r in val_sup]
    model = XGBClassifier(objective="binary:logistic", random_state=SEED, max_depth=4, n_estimators=300, learning_rate=0.05, subsample=0.8, colsample_bytree=0.9, eval_metric="aucpr")
    model.fit(train_x, train_y)
    predictions = model.predict_proba(val_x)[:, 1]
    result = evaluate_organ_support(make_records(val_sup, val_y, predictions.tolist()))
    summary["organ_support"] = {"metrics": result.metrics, "counts": result.counts}
    (OUT_DIR / "organ_support_validation_predictions_v1.json").write_bytes(canonical_json_bytes({
        "status": "NON_SERVING_LATENT_ORACLE", "task": "organ_support",
        "rows": [{"stay_id": r["stay_id"], "prediction_time": r["prediction_time"], "target": float(t), "prediction": float(p)} for r, t, p in zip(val_sup, val_y, predictions)],
    }))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "latent_oracle_summary_v1.json"
    out_path.write_bytes(canonical_json_bytes(summary))
    print("wrote", out_path, sha256_file(out_path))


if __name__ == "__main__":
    main()
