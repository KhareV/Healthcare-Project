"""Performance-v2 Phase 4, Parts 10-16: all metrics, naive comparison,
grouped bootstrap, calibration evidence, prespecified error analysis, and
validation-to-test generalization -- computed EXCLUSIVELY from the saved
Phase-4 prediction artifacts. No model inference happens here.
"""

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.synthetic.config import canonical_json_bytes  # noqa: E402
from evaluation.bootstrap import bootstrap_icu_time, bootstrap_organ_support, bootstrap_recovery, grouped_bootstrap  # noqa: E402
from evaluation.metrics import PredictionRecord, evaluate_icu_time, evaluate_organ_support, evaluate_recovery_horizon  # noqa: E402
from evaluation.weighted_stats import weighted_mean, weighted_median  # noqa: E402
from evaluation.weights import compute_stay_weights  # noqa: E402
from vedant_infra.hashing import sha256_file  # noqa: E402

PRED_DIR = ROOT / "artifacts/performance_v2/phase4/predictions"
METRICS_DIR = ROOT / "artifacts/performance_v2/phase4/metrics"
BOOTSTRAP_DIR = ROOT / "artifacts/performance_v2/phase4/bootstrap"

N_BOOTSTRAP = 2000
SEED = 20260924
THRESHOLD = 0.39781983118092895
THRESHOLD_ID = "support_threshold_v2"


def _load_jsonl(path):
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main() -> None:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    BOOTSTRAP_DIR.mkdir(parents=True, exist_ok=True)

    rec24_rows = _load_jsonl(PRED_DIR / "recovery24_predictions_v1.jsonl")
    rec48_rows = _load_jsonl(PRED_DIR / "recovery48_predictions_v1.jsonl")
    icu_rows_ = _load_jsonl(PRED_DIR / "icu_stay_time_predictions_v1.jsonl")
    support_rows_ = _load_jsonl(PRED_DIR / "organ_support_predictions_v1.jsonl")

    rec24_records = tuple(PredictionRecord(stay_id=r["stay_id"], prediction_time=r["prediction_time"], target=r["target_delta24"], prediction=r["prediction_delta24"], eligible=True) for r in rec24_rows)
    rec48_records = tuple(PredictionRecord(stay_id=r["stay_id"], prediction_time=r["prediction_time"], target=r["target_delta48"], prediction=r["prediction_delta48"], eligible=True) for r in rec48_rows)
    icu_records = tuple(PredictionRecord(stay_id=r["stay_id"], prediction_time=r["prediction_time"], target=r["true_remaining_hours"], prediction=r["prediction_hours"], eligible=True) for r in icu_rows_)
    support_raw_records = tuple(PredictionRecord(stay_id=r["stay_id"], prediction_time=r["prediction_time"], target=float(r["label"]), prediction=r["raw_probability"], eligible=True) for r in support_rows_)
    support_cal_records = tuple(PredictionRecord(stay_id=r["stay_id"], prediction_time=r["prediction_time"], target=float(r["label"]), prediction=r["calibrated_probability"], eligible=True) for r in support_rows_)

    # --- Part 11: final denominators ---
    denominators = {
        "recovery24": {"eligible_rows": len(rec24_rows), "unique_stays": len({r["stay_id"] for r in rec24_rows})},
        "recovery48": {"eligible_rows": len(rec48_rows), "unique_stays": len({r["stay_id"] for r in rec48_rows})},
        "icu_stay_time": {"eligible_rows": len(icu_rows_), "unique_stays": len({r["stay_id"] for r in icu_rows_})},
        "organ_support": {
            "eligible_rows": len(support_rows_), "unique_stays": len({r["stay_id"] for r in support_rows_}),
            "positive_rows": sum(r["label"] for r in support_rows_),
            "positive_stays": len({r["stay_id"] for r in support_rows_ if r["label"] == 1}),
            "fresh_test_prevalence_row_level": sum(r["label"] for r in support_rows_) / len(support_rows_),
        },
    }

    # --- Part 2: final metrics ---
    def weighted_median_ae(records):
        errors = tuple(abs(r.prediction - r.target) for r in records)
        weights = compute_stay_weights([r.stay_id for r in records], [True] * len(records)).weights
        return weighted_median(errors, weights)

    rec24_eval = evaluate_recovery_horizon(rec24_records, horizon="24h")
    rec48_eval = evaluate_recovery_horizon(rec48_records, horizon="48h")
    icu_eval = evaluate_icu_time(icu_records)
    support_raw_eval = evaluate_organ_support(support_raw_records, probability_type="raw")
    support_cal_eval = evaluate_organ_support(
        support_cal_records, fixed_threshold=THRESHOLD, threshold_identifier=THRESHOLD_ID, probability_type="calibrated"
    )

    metrics = {
        "status": "V2_FRESH_TEST_METRICS_FROM_SAVED_PREDICTIONS",
        "recovery24": {**rec24_eval.metrics, "weighted_median_ae": weighted_median_ae(rec24_records), "counts": rec24_eval.counts},
        "recovery48": {**rec48_eval.metrics, "weighted_median_ae": weighted_median_ae(rec48_records), "counts": rec48_eval.counts},
        "icu_stay_time": {**icu_eval.metrics, "counts": icu_eval.counts},
        "organ_support_raw": {**support_raw_eval.metrics, "counts": support_raw_eval.counts},
        "organ_support_calibrated": {**support_cal_eval.metrics, "counts": support_cal_eval.counts},
        "denominators": denominators,
    }
    metrics_path = METRICS_DIR / "final_metrics_v2.json"
    metrics_path.write_bytes(canonical_json_bytes(metrics))
    print("wrote", metrics_path, sha256_file(metrics_path))

    # --- Part 12: naive baseline comparison (same eligible rows, constant predictions) ---
    baselines = json.loads((ROOT / "artifacts/performance_v2/phase4/naive_baselines_v2.json").read_text())
    rec24_naive_records = tuple(PredictionRecord(stay_id=r.stay_id, prediction_time=r.prediction_time, target=r.target, prediction=baselines["recovery24"]["constant_prediction"], eligible=True) for r in rec24_records)
    rec48_naive_records = tuple(PredictionRecord(stay_id=r.stay_id, prediction_time=r.prediction_time, target=r.target, prediction=baselines["recovery48"]["constant_prediction"], eligible=True) for r in rec48_records)
    icu_naive_records = tuple(PredictionRecord(stay_id=r.stay_id, prediction_time=r.prediction_time, target=r.target, prediction=baselines["icu_stay_time"]["constant_prediction_hours"], eligible=True) for r in icu_records)
    support_naive_records = tuple(PredictionRecord(stay_id=r.stay_id, prediction_time=r.prediction_time, target=r.target, prediction=baselines["organ_support"]["constant_probability"], eligible=True) for r in support_cal_records)

    rec24_naive_eval = evaluate_recovery_horizon(rec24_naive_records, horizon="24h")
    rec48_naive_eval = evaluate_recovery_horizon(rec48_naive_records, horizon="48h")
    icu_naive_eval = evaluate_icu_time(icu_naive_records)
    support_naive_eval = evaluate_organ_support(support_naive_records, fixed_threshold=THRESHOLD, threshold_identifier=THRESHOLD_ID, probability_type="calibrated")

    def pct_improvement(naive, model, lower_is_better=True):
        if naive in (0, None) or model is None or not math.isfinite(naive):
            return None
        if lower_is_better:
            return (naive - model) / naive
        return (model - naive) / naive

    naive_comparison = {
        "status": "V2_NAIVE_COMPARISON_FROM_SAVED_PREDICTIONS",
        "recovery24": {"model_mae": rec24_eval.metrics["mae"], "naive_mae": rec24_naive_eval.metrics["mae"], "relative_improvement": pct_improvement(rec24_naive_eval.metrics["mae"], rec24_eval.metrics["mae"])},
        "recovery48": {"model_mae": rec48_eval.metrics["mae"], "naive_mae": rec48_naive_eval.metrics["mae"], "relative_improvement": pct_improvement(rec48_naive_eval.metrics["mae"], rec48_eval.metrics["mae"])},
        "icu_stay_time": {"model_median_ae": icu_eval.metrics["median_absolute_error"], "naive_median_ae": icu_naive_eval.metrics["median_absolute_error"], "relative_improvement": pct_improvement(icu_naive_eval.metrics["median_absolute_error"], icu_eval.metrics["median_absolute_error"])},
        "organ_support": {"model_calibrated_auprc": support_cal_eval.metrics["auprc"], "naive_auprc": support_naive_eval.metrics["auprc"], "relative_improvement": pct_improvement(support_naive_eval.metrics["auprc"], support_cal_eval.metrics["auprc"], lower_is_better=False)},
    }
    naive_path = METRICS_DIR / "naive_comparison_v2.json"
    naive_path.write_bytes(canonical_json_bytes(naive_comparison))
    print("wrote", naive_path, sha256_file(naive_path))

    # --- Part 4/13: grouped bootstrap from saved predictions only ---
    recovery_boot = bootstrap_recovery(rec24_records, rec48_records, n_bootstrap=N_BOOTSTRAP, seed=SEED)
    icu_boot = bootstrap_icu_time(icu_records, n_bootstrap=N_BOOTSTRAP, seed=SEED)
    support_raw_boot = bootstrap_organ_support(support_raw_records, n_bootstrap=N_BOOTSTRAP, seed=SEED, probability_type="raw")
    support_cal_boot = bootstrap_organ_support(support_cal_records, n_bootstrap=N_BOOTSTRAP, seed=SEED, fixed_threshold=THRESHOLD, threshold_identifier=THRESHOLD_ID, probability_type="calibrated")

    def _weighted_median_bootstrap(records, task, horizon=None):
        def metric_fn(rows):
            errors = tuple(abs(r.prediction - r.target) for r in rows)
            weights = compute_stay_weights([r.stay_id for r in rows], [True] * len(rows)).weights
            return weighted_median(errors, weights)

        return grouped_bootstrap(records, metric_fn=metric_fn, metric_name="weighted_median_ae", task=task, n_bootstrap=N_BOOTSTRAP, seed=SEED, horizon=horizon)

    def _naive_bootstrap_recovery(records, horizon):
        def metric_fn(rows):
            return evaluate_recovery_horizon(rows, horizon=horizon).metrics["mae"]

        return grouped_bootstrap(records, metric_fn=metric_fn, metric_name="naive_mae", task="recovery_naive", n_bootstrap=N_BOOTSTRAP, seed=SEED, horizon=horizon)

    def _naive_bootstrap_icu(records):
        def metric_fn(rows):
            return evaluate_icu_time(rows).metrics["median_absolute_error"]

        return grouped_bootstrap(records, metric_fn=metric_fn, metric_name="naive_median_ae", task="icu_naive", n_bootstrap=N_BOOTSTRAP, seed=SEED)

    def _naive_bootstrap_support(records):
        def metric_fn(rows):
            return evaluate_organ_support(rows, fixed_threshold=THRESHOLD, threshold_identifier=THRESHOLD_ID, probability_type="calibrated").metrics["auprc"]

        return grouped_bootstrap(records, metric_fn=metric_fn, metric_name="naive_auprc", task="support_naive", n_bootstrap=N_BOOTSTRAP, seed=SEED)

    def _serialize(result):
        return {
            "point_estimate": result.point_estimate, "ci_lower": result.ci_lower, "ci_upper": result.ci_upper,
            "ci_level": result.ci_level, "n_requested_replicates": result.n_requested_replicates,
            "n_valid_replicates": result.n_valid_replicates, "n_invalid_replicates": result.n_invalid_replicates,
            "n_original_stays": result.n_original_stays, "n_original_examples": result.n_original_examples,
            "bootstrap_seed": result.bootstrap_seed, "method": result.method, "unit": "stay",
        }

    bootstrap_output = {
        "status": "V2_FINAL_BOOTSTRAP_FROM_SAVED_PREDICTIONS",
        "config": {"n_bootstrap": N_BOOTSTRAP, "seed": SEED, "ci_level": 0.95, "resampling_unit": "stay_id", "method": "stay_cluster_percentile_bootstrap"},
        "recovery24": {name: _serialize(result) for name, result in recovery_boot["24h"].items()},
        "recovery48": {name: _serialize(result) for name, result in recovery_boot["48h"].items()},
        "recovery24_weighted_median_ae": _serialize(_weighted_median_bootstrap(rec24_records, "recovery", horizon="24h")),
        "recovery48_weighted_median_ae": _serialize(_weighted_median_bootstrap(rec48_records, "recovery", horizon="48h")),
        "icu_stay_time": {name: _serialize(result) for name, result in icu_boot.items()},
        "organ_support_raw": {name: _serialize(result) for name, result in support_raw_boot.items()},
        "organ_support_calibrated": {name: _serialize(result) for name, result in support_cal_boot.items()},
        "naive_recovery24_mae": _serialize(_naive_bootstrap_recovery(rec24_naive_records, "24h")),
        "naive_recovery48_mae": _serialize(_naive_bootstrap_recovery(rec48_naive_records, "48h")),
        "naive_icu_median_ae": _serialize(_naive_bootstrap_icu(icu_naive_records)),
        "naive_support_auprc": _serialize(_naive_bootstrap_support(support_naive_records)),
    }
    bootstrap_path = BOOTSTRAP_DIR / "final_bootstrap_v2.json"
    bootstrap_path.write_bytes(canonical_json_bytes(bootstrap_output))
    print("wrote", bootstrap_path, sha256_file(bootstrap_path))

    # --- Part 14: calibration evidence ---
    raw_brier = support_raw_eval.metrics["brier"]
    n_bins = 10
    bin_edges = [i / n_bins for i in range(n_bins + 1)]

    def reliability_table(records):
        table = []
        for i in range(n_bins):
            lo, hi = bin_edges[i], bin_edges[i + 1]
            in_bin = [r for r in records if (lo <= r.prediction < hi) or (i == n_bins - 1 and r.prediction == hi)]
            if not in_bin:
                table.append({"bin_lower": lo, "bin_upper": hi, "n": 0, "mean_predicted": None, "observed_frequency": None})
                continue
            weights = compute_stay_weights([r.stay_id for r in in_bin], [True] * len(in_bin)).weights
            mean_pred = weighted_mean([r.prediction for r in in_bin], weights)
            observed = weighted_mean([r.target for r in in_bin], weights)
            table.append({"bin_lower": lo, "bin_upper": hi, "n": len(in_bin), "mean_predicted": mean_pred, "observed_frequency": observed})
        return table

    calibration_evidence = {
        "status": "V2_CALIBRATION_EVIDENCE_FROM_SAVED_PREDICTIONS",
        "raw_brier": raw_brier,
        "calibrated_brier": support_cal_eval.metrics["brier"],
        "reliability_table_convention": "10_equal_width_probability_bins_stay_balanced",
        "raw_reliability_table": reliability_table(support_raw_records),
        "calibrated_reliability_table": reliability_table(support_cal_records),
    }
    calibration_path = METRICS_DIR / "calibration_evidence_v2.json"
    calibration_path.write_bytes(canonical_json_bytes(calibration_evidence))
    print("wrote", calibration_path, sha256_file(calibration_path))

    # --- Part 15: prespecified error analysis ---
    plan = json.loads((ROOT / "artifacts/performance_v2/phase4/final_evaluation_plan_v1.json").read_text())
    slices = plan["prespecified_error_analysis_slices"]

    def band_of(value, t33, t67):
        if value <= t33:
            return "LOW"
        if value <= t67:
            return "MODERATE"
        return "HIGH"

    def timing_band(grid_index):
        if grid_index <= 3:
            return "EARLY"
        if grid_index <= 7:
            return "MID"
        return "LATE"

    def slice_mae(rows_with_target_pred, group_key_fn):
        groups = {}
        for row in rows_with_target_pred:
            key = group_key_fn(row)
            if key is None:
                continue
            groups.setdefault(key, []).append(row)
        result = {}
        for key, group_rows in groups.items():
            records = tuple(PredictionRecord(stay_id=r["stay_id"], prediction_time=r["prediction_time"], target=r["_target"], prediction=r["_prediction"], eligible=True) for r in group_rows)
            weights = compute_stay_weights([r.stay_id for r in records], [True] * len(records)).weights
            errors = tuple(abs(r.prediction - r.target) for r in records)
            result[key] = {"n_rows": len(records), "n_stays": len({r.stay_id for r in records}), "mae": weighted_mean(errors, weights)}
        return result

    def with_target_pred(rows, target_field, pred_field):
        out = []
        for r in rows:
            item = dict(r)
            item["_target"] = r[target_field]
            item["_prediction"] = r[pred_field]
            out.append(item)
        return out

    rec24_with = with_target_pred(rec24_rows, "target_delta24", "prediction_delta24")
    rec48_with = with_target_pred(rec48_rows, "target_delta48", "prediction_delta48")
    icu_with = with_target_pred(icu_rows_, "true_remaining_hours", "prediction_hours")

    sofa_t33 = slices["current_sofa_severity_band"]["dev_derived_tertile_boundaries"]["t33"]
    sofa_t67 = slices["current_sofa_severity_band"]["dev_derived_tertile_boundaries"]["t67"]
    icu_q25 = slices["icu_true_remaining_time_quartile"]["dev_derived_quartile_boundaries"]["q25"]
    icu_q50 = slices["icu_true_remaining_time_quartile"]["dev_derived_quartile_boundaries"]["q50"]
    icu_q75 = slices["icu_true_remaining_time_quartile"]["dev_derived_quartile_boundaries"]["q75"]

    def recovery_sign(target):
        if target < 0:
            return "NEGATIVE"
        if target == 0:
            return "ZERO"
        return "POSITIVE"

    def icu_quartile(hours):
        if hours <= icu_q25:
            return "Q1"
        if hours <= icu_q50:
            return "Q2"
        if hours <= icu_q75:
            return "Q3"
        return "Q4"

    error_analysis = {
        "status": "V2_PRESPECIFIED_ERROR_ANALYSIS_FROM_SAVED_PREDICTIONS",
        "no_post_hoc_subgroup_hunting": True,
        "recovery24": {
            "by_cutoff_timing": slice_mae(rec24_with, lambda r: timing_band(r["grid_index"])),
            "by_true_delta_sign": slice_mae(rec24_with, lambda r: recovery_sign(r["_target"])),
        },
        "recovery48": {
            "by_cutoff_timing": slice_mae(rec48_with, lambda r: timing_band(r["grid_index"])),
            "by_true_delta_sign": slice_mae(rec48_with, lambda r: recovery_sign(r["_target"])),
        },
        "icu_stay_time": {
            "by_cutoff_timing": slice_mae(icu_with, lambda r: timing_band(r["grid_index"])),
            "by_true_remaining_time_quartile": slice_mae(icu_with, lambda r: icu_quartile(r["_target"])),
        },
    }
    error_analysis_path = METRICS_DIR / "error_analysis_v2.json"
    error_analysis_path.write_bytes(canonical_json_bytes(error_analysis))
    print("wrote", error_analysis_path, sha256_file(error_analysis_path))

    # --- Part 16: generalization comparison ---
    validation_selection = json.loads((ROOT / "artifacts/performance_v2/phase2/validation_selection_v2.json").read_text())

    def gen_row(task, val_metric_key, val_value, test_value):
        return {"task": task, "phase2_validation": val_value, "fresh_test": test_value, "difference": test_value - val_value if val_value is not None and test_value is not None else None}

    val24 = validation_selection["tasks"]["recovery24"]["validation_metrics"]["mae"]
    val48 = validation_selection["tasks"]["recovery48"]["validation_metrics"]["mae"]
    val_icu_median = validation_selection["tasks"]["icu_stay_time"]["validation_metrics"]["median_absolute_error"]
    val_support_auprc = validation_selection["tasks"]["organ_support"]["validation_metrics"]["auprc"]

    generalization = {
        "status": "V2_VALIDATION_TO_TEST_GENERALIZATION",
        "recovery24_mae": gen_row("recovery24", "mae", val24, rec24_eval.metrics["mae"]),
        "recovery48_mae": gen_row("recovery48", "mae", val48, rec48_eval.metrics["mae"]),
        "icu_stay_time_median_ae": gen_row("icu_stay_time", "median_absolute_error", val_icu_median, icu_eval.metrics["median_absolute_error"]),
        "organ_support_auprc": {
            "task": "organ_support", "phase2_validation_raw_auprc": val_support_auprc,
            "fresh_test_calibrated_auprc": support_cal_eval.metrics["auprc"],
            "fresh_test_raw_auprc": support_raw_eval.metrics["auprc"],
            "note": "Phase-2 validation AUPRC predates calibration (raw); compared here against fresh-test calibrated AUPRC per the Phase-4 spec, with fresh-test raw AUPRC also reported to show calibration did not alter ranking unexpectedly.",
        },
        "interpretation": "descriptive_only_no_retraining",
    }
    generalization_path = METRICS_DIR / "generalization_comparison_v2.json"
    generalization_path.write_bytes(canonical_json_bytes(generalization))
    print("wrote", generalization_path, sha256_file(generalization_path))

    print("ALL_METRICS_BOOTSTRAP_CALIBRATION_COMPLETE")


if __name__ == "__main__":
    main()
