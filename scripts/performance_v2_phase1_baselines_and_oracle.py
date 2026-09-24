"""Performance-v2 Phase 1, Parts 6-7 and 15: stronger simple baselines,
OBSERVABLE_ORACLE, and feature-group ablation attribution.

DEVELOPMENT_DIAGNOSTIC_ONLY / NON_SERVING_DIAGNOSTIC_ORACLE. Fits only on
TRAIN, ranks only on VALIDATION, never touches v1 TEST. Not a serving
artifact, not a retrain/reselection of any Benchmark-v1 model.
"""

import json
import math
import sys
from pathlib import Path
from typing import Dict

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from xgboost import XGBClassifier, XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.synthetic.config import canonical_json_bytes  # noqa: E402
from evaluation.metrics import evaluate_icu_time, evaluate_organ_support, evaluate_recovery_horizon  # noqa: E402
from performance_v2.data_loading import load_dev_rows  # noqa: E402
from performance_v2.diagnostic_features import GROUP_ORDER  # noqa: E402
from performance_v2.feature_matrix import build_feature_matrix_fast, columns_for_groups, load_sofa_trend_cache  # noqa: E402
from performance_v2.targets import icu_hours_target, icu_rows, make_records, recovery_targets, support_rows  # noqa: E402
from vedant_infra.hashing import sha256_file  # noqa: E402

OUT_DIR = ROOT / "artifacts/performance_v2/phase1"
ORACLE_DIR = OUT_DIR / "observable_oracle"
SEED = 2026_09_24


def _split_mask(rows, split):
    return np.asarray([row["split"] == split for row in rows], dtype=bool)


def _impute(train_x, other_x):
    imputer = SimpleImputer(strategy="median")
    train_imputed = imputer.fit_transform(train_x)
    other_imputed = imputer.transform(other_x)
    return train_imputed, other_imputed


# ---------------------------------------------------------------------------
# Recovery
# ---------------------------------------------------------------------------

def _recovery_linear_trend_prediction(row, horizon_hours):
    trend = row.get("_sofa_trend", {})
    points = [(-lag, trend[key]) for lag, key in ((0, "t"), (6, "t_minus_6h"), (12, "t_minus_12h"), (24, "t_minus_24h"), (48, "t_minus_48h")) if key in trend]
    if len(points) < 2:
        return 0.0
    xs = np.asarray([p[0] for p in points], dtype=float)
    ys = np.asarray([p[1] for p in points], dtype=float)
    if np.allclose(xs, xs[0]):
        return 0.0
    slope, _ = np.polyfit(xs, ys, 1)
    return float(slope * horizon_hours)


def run_recovery(train_rows, validation_rows, sofa_trends):
    result = {}
    ablation = {}
    oracle_summary = {}
    for horizon in ("24", "48"):
        train_h = recovery_targets(train_rows, horizon)
        val_h = recovery_targets(validation_rows, horizon)
        for row in train_h + val_h:
            row["_sofa_trend"] = sofa_trends.get((row["stay_id"], row["prediction_time"]), {})

        train_targets = np.asarray([float(r["delta_sofa_" + horizon]) for r in train_h])
        val_targets = np.asarray([float(r["delta_sofa_" + horizon]) for r in val_h])

        # R1: linear SOFA-trend extrapolation.
        val_r1 = [_recovery_linear_trend_prediction(r, int(horizon)) for r in val_h]
        r1_result = evaluate_recovery_horizon(make_records(val_h, val_targets.tolist(), val_r1), horizon=horizon + "h")

        # Feature matrix (shared for R2 and observable oracle + ablation).
        all_rows = train_h + val_h
        matrix, names = build_feature_matrix_fast(all_rows, sofa_trends)
        n_train = len(train_h)
        train_x, val_x = matrix[:n_train], matrix[n_train:]

        # R2: ridge on median-imputed full diagnostic feature set.
        train_x_imp, val_x_imp = _impute(train_x, val_x)
        ridge = Ridge(alpha=10.0, random_state=SEED).fit(train_x_imp, train_targets)
        val_r2 = ridge.predict(val_x_imp)
        r2_result = evaluate_recovery_horizon(make_records(val_h, val_targets.tolist(), val_r2.tolist()), horizon=horizon + "h")

        # Observable oracle: small deterministic grid, <=8 configs, XGBoost.
        configs = [
            {"max_depth": d, "n_estimators": n, "learning_rate": lr}
            for d in (3, 5) for n in (150, 400) for lr in (0.05,)
        ][:8]
        best = None
        for config in configs:
            model = XGBRegressor(
                objective="reg:absoluteerror", random_state=SEED, missing=np.nan,
                subsample=0.8, colsample_bytree=0.8, **config,
            )
            model.fit(train_x, train_targets)
            predictions = model.predict(val_x)
            evaluated = evaluate_recovery_horizon(make_records(val_h, val_targets.tolist(), predictions.tolist()), horizon=horizon + "h")
            mae = evaluated.metrics["mae"]
            if best is None or mae < best["mae"]:
                best = {"config": config, "mae": mae, "result": evaluated, "predictions": predictions, "model": model}

        oracle_summary["recovery" + horizon] = {
            "chosen_config": best["config"], "configs_tried": len(configs),
            "metrics": best["result"].metrics, "counts": best["result"].counts,
        }
        predictions_path = ORACLE_DIR / f"recovery{horizon}_validation_predictions_v1.json"
        predictions_path.parent.mkdir(parents=True, exist_ok=True)
        predictions_path.write_bytes(canonical_json_bytes({
            "status": "NON_SERVING_DIAGNOSTIC_ORACLE",
            "task": "recovery" + horizon,
            "rows": [
                {"stay_id": r["stay_id"], "prediction_time": r["prediction_time"], "target": float(t), "prediction": float(p)}
                for r, t, p in zip(val_h, val_targets, best["predictions"])
            ],
        }))

        # Ablation A..F, same config (the chosen oracle config) each time.
        ablation_h = {}
        chosen_config = best["config"]
        cumulative = []
        for group in GROUP_ORDER:
            cumulative.append(group)
            columns = columns_for_groups(names, cumulative)
            model = XGBRegressor(
                objective="reg:absoluteerror", random_state=SEED, missing=np.nan,
                subsample=0.8, colsample_bytree=0.8, **chosen_config,
            )
            model.fit(train_x[:, columns], train_targets)
            predictions = model.predict(val_x[:, columns])
            evaluated = evaluate_recovery_horizon(make_records(val_h, val_targets.tolist(), predictions.tolist()), horizon=horizon + "h")
            ablation_h["+".join(cumulative)] = {"mae": evaluated.metrics["mae"], "rmse": evaluated.metrics["rmse"], "n_features": int(columns.sum())}
        ablation["recovery" + horizon] = ablation_h

        result["recovery" + horizon] = {
            "n_train": len(train_h), "n_validation": len(val_h),
            "R0_naive_zero_delta": {"mae": float(np.mean(np.abs(val_targets)))},
            "R1_linear_sofa_trend": {"metrics": r1_result.metrics, "counts": r1_result.counts},
            "R2_ridge_diagnostic_features": {"metrics": r2_result.metrics, "counts": r2_result.counts},
        }
    return result, oracle_summary, ablation


# ---------------------------------------------------------------------------
# ICU
# ---------------------------------------------------------------------------

def _median_by_key(train_rows_h, key_fn):
    """Stay-balanced weighted median per bucket, matching the same
    stay-balanced estimator evaluate_icu_time uses -- an unweighted
    per-row median would systematically over-represent stays with many
    eligible rows (typically longer stays with a different remaining-time
    distribution), silently biasing the conditional baselines relative to
    the naive baseline's own stay-balanced constant."""

    from evaluation.weighted_stats import weighted_median
    from evaluation.weights import compute_stay_weights

    buckets_hours: Dict[object, list] = {}
    buckets_stays: Dict[object, list] = {}
    for row, hours in train_rows_h:
        key = key_fn(row)
        buckets_hours.setdefault(key, []).append(hours)
        buckets_stays.setdefault(key, []).append(row["stay_id"])
    result = {}
    for key, hours in buckets_hours.items():
        weights = compute_stay_weights(buckets_stays[key], [True] * len(hours)).weights
        result[key] = float(weighted_median(hours, weights))
    return result


def _global_weighted_median(rows, hours):
    from evaluation.weighted_stats import weighted_median
    from evaluation.weights import compute_stay_weights

    weights = compute_stay_weights([row["stay_id"] for row in rows], [True] * len(rows)).weights
    return float(weighted_median(hours, weights))


def _elapsed_band(elapsed):
    edges = [24, 42, 60, 78, 96, 1e9]
    return next(i for i in range(len(edges) - 1) if edges[i] <= elapsed < edges[i + 1])


def _severity_band(sofa):
    if sofa is None:
        return "unknown"
    return "low" if sofa <= 3 else ("mid" if sofa <= 7 else "high")


def run_icu(train_rows, validation_rows, naive_median_hours, sofa_trends):
    train_icu = icu_rows(train_rows)
    val_icu = icu_rows(validation_rows)
    train_hours = [icu_hours_target(r) for r in train_icu]
    val_hours = [icu_hours_target(r) for r in val_icu]

    # I0: existing train-median naive baseline (already frozen elsewhere).
    i0_pred = [naive_median_hours] * len(val_icu)
    i0_result = evaluate_icu_time(make_records(val_icu, val_hours, i0_pred))

    # I1: train-only conditional median by elapsed-hours band. Bucket
    # medians and the fallback are stay-balanced weighted medians, matching
    # the same estimator evaluate_icu_time uses and the naive baseline's
    # own fitting convention (see _median_by_key).
    by_elapsed = _median_by_key(list(zip(train_icu, train_hours)), lambda r: _elapsed_band(r["elapsed_episode_hours_at_t"]))
    global_median = _global_weighted_median(train_icu, train_hours)
    i1_pred = [by_elapsed.get(_elapsed_band(r["elapsed_episode_hours_at_t"]), global_median) for r in val_icu]
    i1_result = evaluate_icu_time(make_records(val_icu, val_hours, i1_pred))

    # I2: elapsed x cutoff x cardiac x severity, min-cell-count fallback.
    def i2_key(row):
        return (_elapsed_band(row["elapsed_episode_hours_at_t"]), row["cutoff_index"], row["cardiac_condition_group"], _severity_band(row.get("baseline_sofa")))
    fine_medians = _median_by_key(list(zip(train_icu, train_hours)), i2_key)
    fine_counts: Dict[object, int] = {}
    for row in train_icu:
        fine_counts[i2_key(row)] = fine_counts.get(i2_key(row), 0) + 1
    MIN_CELL = 20
    i2_pred = []
    for row in val_icu:
        key = i2_key(row)
        if fine_counts.get(key, 0) >= MIN_CELL:
            i2_pred.append(fine_medians[key])
        else:
            i2_pred.append(by_elapsed.get(_elapsed_band(row["elapsed_episode_hours_at_t"]), global_median))
    i2_result = evaluate_icu_time(make_records(val_icu, val_hours, i2_pred))

    # I3: ridge in log1p space over diagnostic features.
    matrix, names = build_feature_matrix_fast(train_icu + val_icu, sofa_trends)
    n_train = len(train_icu)
    train_x, val_x = matrix[:n_train], matrix[n_train:]
    train_x_imp, val_x_imp = _impute(train_x, val_x)
    train_log1p = np.log1p(train_hours)
    ridge = Ridge(alpha=10.0, random_state=SEED).fit(train_x_imp, train_log1p)
    i3_pred = np.expm1(np.clip(ridge.predict(val_x_imp), 0, None))
    i3_result = evaluate_icu_time(make_records(val_icu, val_hours, i3_pred.tolist()))

    # Observable oracle.
    configs = [
        {"max_depth": d, "n_estimators": n, "learning_rate": lr}
        for d in (3, 5) for n in (150, 400) for lr in (0.05,)
    ][:8]
    best = None
    for config in configs:
        model = XGBRegressor(objective="reg:absoluteerror", random_state=SEED, missing=np.nan, subsample=0.8, colsample_bytree=0.8, **config)
        model.fit(train_x, train_log1p)
        predictions = np.expm1(np.clip(model.predict(val_x), 0, None))
        evaluated = evaluate_icu_time(make_records(val_icu, val_hours, predictions.tolist()))
        mae = evaluated.metrics["median_absolute_error"]
        if best is None or mae < best["mae"]:
            best = {"config": config, "mae": mae, "result": evaluated, "predictions": predictions}
    ORACLE_DIR.mkdir(parents=True, exist_ok=True)
    (ORACLE_DIR / "icu_stay_time_validation_predictions_v1.json").write_bytes(canonical_json_bytes({
        "status": "NON_SERVING_DIAGNOSTIC_ORACLE", "task": "icu_stay_time",
        "rows": [
            {"stay_id": r["stay_id"], "prediction_time": r["prediction_time"], "target": float(t), "prediction": float(p)}
            for r, t, p in zip(val_icu, val_hours, best["predictions"])
        ],
    }))

    ablation = {}
    cumulative = []
    for group in GROUP_ORDER:
        cumulative.append(group)
        columns = columns_for_groups(names, cumulative)
        model = XGBRegressor(objective="reg:absoluteerror", random_state=SEED, missing=np.nan, subsample=0.8, colsample_bytree=0.8, **best["config"])
        model.fit(train_x[:, columns], train_log1p)
        predictions = np.expm1(np.clip(model.predict(val_x[:, columns]), 0, None))
        evaluated = evaluate_icu_time(make_records(val_icu, val_hours, predictions.tolist()))
        ablation["+".join(cumulative)] = {"median_absolute_error": evaluated.metrics["median_absolute_error"], "mae": evaluated.metrics["mae"], "n_features": int(columns.sum())}

    result = {
        "n_train": len(train_icu), "n_validation": len(val_icu),
        "I0_naive_train_median": {"metrics": i0_result.metrics, "counts": i0_result.counts},
        "I1_conditional_median_by_elapsed_band": {"metrics": i1_result.metrics, "counts": i1_result.counts},
        "I2_conditional_median_fine_grid": {"metrics": i2_result.metrics, "counts": i2_result.counts, "min_cell_count": MIN_CELL},
        "I3_ridge_log1p_diagnostic_features": {"metrics": i3_result.metrics, "counts": i3_result.counts},
    }
    oracle_summary = {"chosen_config": best["config"], "configs_tried": len(configs), "metrics": best["result"].metrics, "counts": best["result"].counts}
    return result, oracle_summary, ablation


# ---------------------------------------------------------------------------
# Support
# ---------------------------------------------------------------------------

def run_support(train_rows, validation_rows, train_prevalence, sofa_trends):
    train_sup = support_rows(train_rows)
    val_sup = support_rows(validation_rows)
    train_y = [float(r["organ_support_label"]) for r in train_sup]
    val_y = [float(r["organ_support_label"]) for r in val_sup]

    naive_pred = [train_prevalence] * len(val_sup)
    naive_result = evaluate_organ_support(make_records(val_sup, val_y, naive_pred))

    matrix, names = build_feature_matrix_fast(train_sup + val_sup, sofa_trends)
    n_train = len(train_sup)
    train_x, val_x = matrix[:n_train], matrix[n_train:]
    train_x_imp, val_x_imp = _impute(train_x, val_x)
    logistic = LogisticRegression(max_iter=2000, C=1.0, random_state=SEED).fit(train_x_imp, train_y)
    val_logistic = logistic.predict_proba(val_x_imp)[:, 1]
    logistic_result = evaluate_organ_support(make_records(val_sup, val_y, val_logistic.tolist()))

    configs = [
        {"max_depth": d, "n_estimators": n, "learning_rate": lr}
        for d in (3, 5) for n in (150, 400) for lr in (0.05,)
    ][:8]
    best = None
    for config in configs:
        model = XGBClassifier(objective="binary:logistic", random_state=SEED, missing=np.nan, subsample=0.8, colsample_bytree=0.8, eval_metric="aucpr", **config)
        model.fit(train_x, train_y)
        predictions = model.predict_proba(val_x)[:, 1]
        evaluated = evaluate_organ_support(make_records(val_sup, val_y, predictions.tolist()))
        auprc = evaluated.metrics["auprc"]
        if best is None or auprc > best["auprc"]:
            best = {"config": config, "auprc": auprc, "result": evaluated, "predictions": predictions}
    ORACLE_DIR.mkdir(parents=True, exist_ok=True)
    (ORACLE_DIR / "organ_support_validation_predictions_v1.json").write_bytes(canonical_json_bytes({
        "status": "NON_SERVING_DIAGNOSTIC_ORACLE", "task": "organ_support",
        "rows": [
            {"stay_id": r["stay_id"], "prediction_time": r["prediction_time"], "target": float(t), "prediction": float(p)}
            for r, t, p in zip(val_sup, val_y, best["predictions"])
        ],
    }))

    ablation = {}
    cumulative = []
    for group in GROUP_ORDER:
        cumulative.append(group)
        columns = columns_for_groups(names, cumulative)
        model = XGBClassifier(objective="binary:logistic", random_state=SEED, missing=np.nan, subsample=0.8, colsample_bytree=0.8, eval_metric="aucpr", **best["config"])
        model.fit(train_x[:, columns], train_y)
        predictions = model.predict_proba(val_x[:, columns])[:, 1]
        evaluated = evaluate_organ_support(make_records(val_sup, val_y, predictions.tolist()))
        ablation["+".join(cumulative)] = {"auprc": evaluated.metrics["auprc"], "auroc": evaluated.metrics["auroc"], "n_features": int(columns.sum())}

    result = {
        "n_train": len(train_sup), "n_validation": len(val_sup),
        "prevalence_baseline": {"metrics": naive_result.metrics, "counts": naive_result.counts},
        "logistic_diagnostic_features": {"metrics": logistic_result.metrics, "counts": logistic_result.counts},
    }
    oracle_summary = {"chosen_config": best["config"], "configs_tried": len(configs), "metrics": best["result"].metrics, "counts": best["result"].counts}
    return result, oracle_summary, ablation


def main() -> None:
    train_rows = load_dev_rows(ROOT, splits=("train",))
    validation_rows = load_dev_rows(ROOT, splits=("validation",))
    sofa_trends = load_sofa_trend_cache(ROOT)

    naive = json.loads((ROOT / "artifacts/final_test/naive_baseline/naive_baseline_v1.json").read_text())
    naive_median_hours = math.expm1(max(naive["icu_time_log1p_median"], 0.0))
    train_prevalence = naive["organ_support_prevalence"]

    recovery_result, recovery_oracle, recovery_ablation = run_recovery(train_rows, validation_rows, sofa_trends)
    icu_result, icu_oracle, icu_ablation = run_icu(train_rows, validation_rows, naive_median_hours, sofa_trends)
    support_result, support_oracle, support_ablation = run_support(train_rows, validation_rows, train_prevalence, sofa_trends)

    baselines_payload = {
        "status": "DEVELOPMENT_DIAGNOSTIC_ONLY",
        "artifact_version": "performance_v2_phase1_stronger_baselines_v1",
        "recovery": recovery_result, "icu_stay_time": icu_result, "organ_support": support_result,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "stronger_baselines_v1.json"
    path.write_bytes(canonical_json_bytes(baselines_payload))
    print("wrote", path, sha256_file(path))

    oracle_payload = {
        "status": "NON_SERVING_DIAGNOSTIC_ORACLE",
        "artifact_version": "performance_v2_phase1_observable_oracle_v1",
        "model_family": "xgboost",
        "search_budget": "<=8 deterministic configurations per task, ranked on validation only",
        "recovery": recovery_oracle, "icu_stay_time": icu_oracle, "organ_support": support_oracle,
    }
    ORACLE_DIR.mkdir(parents=True, exist_ok=True)
    oracle_path = ORACLE_DIR / "observable_oracle_summary_v1.json"
    oracle_path.write_bytes(canonical_json_bytes(oracle_payload))
    print("wrote", oracle_path, sha256_file(oracle_path))

    ablation_payload = {
        "status": "DEVELOPMENT_DIAGNOSTIC_ONLY",
        "artifact_version": "performance_v2_phase1_feature_ablation_v1",
        "groups_definition": {
            "A": "existing canonical features only (value+mask+tslo per bin per channel, statics)",
            "B": "+ elapsed/cutoff structural features + current baseline SOFA",
            "C": "+ trend/delta features (per-channel deltas/slopes, SOFA trend)",
            "D": "+ rolling summary features (per-channel windowed mean/min/max/std)",
            "E": "+ missingness/TSLO summaries",
            "F": "+ support-history features (duration-in-state, transitions)",
        },
        "recovery": recovery_ablation, "icu_stay_time": icu_ablation, "organ_support": support_ablation,
    }
    ablation_path = OUT_DIR / "feature_ablation_v1.json"
    ablation_path.write_bytes(canonical_json_bytes(ablation_payload))
    print("wrote", ablation_path, sha256_file(ablation_path))


if __name__ == "__main__":
    main()
