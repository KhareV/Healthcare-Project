"""Performance-v2 Phase 1, Parts 1-2: v1 performance-story reconstruction and
target-distribution / baseline diagnosis.

DEVELOPMENT_DIAGNOSTIC_ONLY. Reads existing frozen TRAIN/VALIDATION search
artifacts and TRAIN/VALIDATION rows only. No retraining, no TEST access.
"""

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.synthetic.config import canonical_json_bytes  # noqa: E402
from evaluation.metrics import evaluate_icu_time, evaluate_organ_support, evaluate_recovery_horizon  # noqa: E402
from evaluation.naive_baseline import load_or_fit_naive_baselines  # noqa: E402
from experiments.stage5_final_test import _observation_density  # noqa: E402
from performance_v2.data_loading import assert_no_test_rows, load_dev_rows  # noqa: E402
from performance_v2.targets import icu_hours_target, icu_rows, make_records, recovery_targets, support_rows  # noqa: E402
from vedant_infra.hashing import sha256_file  # noqa: E402

OUT_DIR = ROOT / "artifacts/performance_v2/phase1"


def _stats(values):
    arr = np.asarray(values, dtype=float)
    q1, q3 = np.percentile(arr, [25, 75])
    return {
        "n": int(arr.size), "mean": float(arr.mean()), "median": float(np.median(arr)),
        "std": float(arr.std(ddof=0)), "iqr": [float(q1), float(q3)],
        "min": float(arr.min()), "max": float(arr.max()),
    }


def _part1(train_rows, validation_rows):
    xgb = json.loads((ROOT / "artifacts/search/xgb/phase12/best_xgb_candidates_v1.json").read_text())
    gru = json.loads((ROOT / "artifacts/search/gru/final_v2/best_gru_candidates_v1.json").read_text())
    lstm = json.loads((ROOT / "artifacts/models/lstm_sensitivity_v1.json").read_text())
    lstm_metrics = {
        run["task"]: json.loads((ROOT / run["metrics_path"]).read_text())["validation_metrics"]
        for run in lstm["runs"]
    }

    naive, naive_path, naive_sha = load_or_fit_naive_baselines(
        ROOT,
        artifact_ref="artifacts/final_test/naive_baseline/naive_baseline_v1.json",
        train_ref="artifacts/data/synthetic/phase10/final/synthetic_phase10_v1/train.jsonl",
    )

    naive_validation = {}
    for horizon in ("24", "48"):
        rows = recovery_targets(validation_rows, horizon)
        targets = [float(r["delta_sofa_" + horizon]) for r in rows]
        predictions = [0.0] * len(rows)
        result = evaluate_recovery_horizon(make_records(rows, targets, predictions), horizon=horizon + "h")
        naive_validation["recovery" + horizon] = {"metrics": result.metrics, "counts": result.counts}
    rows = icu_rows(validation_rows)
    targets = [icu_hours_target(r) for r in rows]
    import math
    predictions = [math.expm1(max(naive.icu_time_log1p_median, 0.0))] * len(rows)
    result = evaluate_icu_time(make_records(rows, targets, predictions))
    naive_validation["icu_stay_time"] = {"metrics": result.metrics, "counts": result.counts}
    rows = support_rows(validation_rows)
    targets = [float(r["organ_support_label"]) for r in rows]
    predictions = [naive.organ_support_prevalence] * len(rows)
    result = evaluate_organ_support(make_records(rows, targets, predictions))
    naive_validation["organ_support"] = {"metrics": result.metrics, "counts": result.counts}

    summary = {
        "status": "DEVELOPMENT_DIAGNOSTIC_ONLY",
        "artifact_version": "performance_v2_phase1_v1_validation_performance_summary_v1",
        "naive_baseline_artifact_sha256": naive_sha,
        "naive_on_validation": naive_validation,
        "recovery": {
            "best_xgb_validation": xgb["tasks"]["recovery"]["metrics"],
            "best_gru_validation": gru["tasks"]["recovery"]["metrics"],
            "lstm_sensitivity_validation": lstm_metrics["recovery"],
            "selected_family": "xgboost", "selected_candidate": "xgb-recovery-014",
        },
        "icu_stay_time": {
            "best_xgb_validation": xgb["tasks"]["icu_time"]["metrics"],
            "best_gru_validation": gru["tasks"]["icu_time"]["metrics"],
            "lstm_sensitivity_validation": lstm_metrics["icu_time"],
            "selected_family": "gru", "selected_candidate": "gru-icu-time-026",
        },
        "organ_support": {
            "best_xgb_validation": xgb["tasks"]["organ_support"]["metrics"],
            "best_gru_validation": gru["tasks"]["organ_support"]["metrics"],
            "lstm_sensitivity_validation": lstm_metrics["organ_support"],
            "selected_family": "xgboost", "selected_candidate": "xgb-support-024",
        },
        "validation_sample_counts": {
            "recovery24_eligible_rows": len(recovery_targets(validation_rows, "24")),
            "recovery48_eligible_rows": len(recovery_targets(validation_rows, "48")),
            "icu_eligible_rows": len(icu_rows(validation_rows)),
            "support_eligible_rows": len(support_rows(validation_rows)),
            "total_validation_rows": len(validation_rows),
            "total_train_rows": len(train_rows),
        },
    }
    return summary


def _sign_distribution(values):
    arr = np.asarray(values, dtype=float)
    return {
        "negative_fraction": float(np.mean(arr < 0)), "zero_fraction": float(np.mean(arr == 0)),
        "positive_fraction": float(np.mean(arr > 0)),
        "near_zero_fraction_abs_le_1": float(np.mean(np.abs(arr) <= 1.0)),
    }


def _part2_recovery(dev_rows):
    out = {}
    for horizon in ("24", "48"):
        rows = recovery_targets(dev_rows, horizon)
        values = np.asarray([float(r["delta_sofa_" + horizon]) for r in rows], dtype=float)
        out["recovery" + horizon] = {
            "n": int(values.size),
            "target_distribution": _stats(values),
            "abs_target_distribution": _stats(np.abs(values)),
            "sign_distribution": _sign_distribution(values),
            "zero_delta_persistence_baseline_mae": float(np.mean(np.abs(values))),
        }
    return out


def _bin_conditional(values_by_key):
    out = {}
    for key, values in values_by_key.items():
        arr = np.asarray(values, dtype=float)
        if arr.size == 0:
            continue
        q1, q3 = np.percentile(arr, [25, 75])
        out[str(key)] = {
            "n": int(arr.size), "median": float(np.median(arr)), "iqr": [float(q1), float(q3)],
            "p10": float(np.percentile(arr, 10)), "p90": float(np.percentile(arr, 90)),
            "variance": float(arr.var(ddof=0)),
        }
    return out


def _part2_icu(dev_rows):
    rows = icu_rows(dev_rows)
    hours = [icu_hours_target(r) for r in rows]
    by_cutoff, by_elapsed_bin, by_cardiac, by_baseline_sofa_band, by_density_band = {}, {}, {}, {}, {}
    elapsed_edges = [24, 42, 60, 78, 96, 1e9]
    for row, h in zip(rows, hours):
        by_cutoff.setdefault(row["cutoff_index"], []).append(h)
        elapsed = row["elapsed_episode_hours_at_t"]
        band = next(i for i in range(len(elapsed_edges) - 1) if elapsed_edges[i] <= elapsed < elapsed_edges[i + 1])
        by_elapsed_bin.setdefault(band, []).append(h)
        by_cardiac.setdefault(row["cardiac_condition_group"], []).append(h)
        sofa = row.get("baseline_sofa")
        if sofa is not None:
            sofa_band = "low(<=3)" if sofa <= 3 else ("mid(4-7)" if sofa <= 7 else "high(>=8)")
            by_baseline_sofa_band.setdefault(sofa_band, []).append(h)
        density = _observation_density(row)
        if density == density:  # not NaN
            density_band = "low(<0.5)" if density < 0.5 else ("mid(0.5-0.8)" if density < 0.8 else "high(>=0.8)")
            by_density_band.setdefault(density_band, []).append(h)

    overall = np.asarray(hours, dtype=float)
    conditional_by_elapsed = _bin_conditional(by_elapsed_bin)
    variances = [v["variance"] for v in conditional_by_elapsed.values()]
    unconditional_variance = float(overall.var(ddof=0))
    mean_conditional_variance = float(np.mean(variances)) if variances else float("nan")
    variance_reduction_from_elapsed_time = (
        (unconditional_variance - mean_conditional_variance) / unconditional_variance
        if unconditional_variance > 0 else float("nan")
    )
    return {
        "n": len(rows),
        "overall_distribution": _stats(overall),
        "by_cutoff_index": _bin_conditional(by_cutoff),
        "by_elapsed_hours_band": {
            "band_edges_hours": elapsed_edges[:-1] + ["inf"],
            "conditional_distributions": conditional_by_elapsed,
        },
        "by_cardiac_subtype": _bin_conditional(by_cardiac),
        "by_baseline_sofa_band": _bin_conditional(by_baseline_sofa_band),
        "by_observation_density_band": _bin_conditional(by_density_band),
        "unconditional_variance": unconditional_variance,
        "mean_within_elapsed_band_variance": mean_conditional_variance,
        "fractional_variance_reduction_from_elapsed_time_alone": variance_reduction_from_elapsed_time,
        "interpretation": (
            "elapsed_episode_hours_at_t strongly determines remaining time"
            if variance_reduction_from_elapsed_time > 0.15 else
            "elapsed_episode_hours_at_t has only a modest unconditional association with remaining time"
        ),
    }


def _part2_support(dev_rows):
    rows = support_rows(dev_rows)
    labels = np.asarray([float(r["organ_support_label"]) for r in rows], dtype=float)
    by_cutoff, by_cardiac, by_severity, by_current_support = {}, {}, {}, {}
    for row, y in zip(rows, labels):
        by_cutoff.setdefault(row["cutoff_index"], []).append(y)
        by_cardiac.setdefault(row["cardiac_condition_group"], []).append(y)
        sofa = row.get("baseline_sofa")
        if sofa is not None:
            band = "low(<=3)" if sofa <= 3 else ("mid(4-7)" if sofa <= 7 else "high(>=8)")
            by_severity.setdefault(band, []).append(y)
        seq = row["history_values"]
        names = row["temporal_feature_names"]
        v_idx = names.index("vasopressor_on")
        i_idx = names.index("invasive_ventilation_on")
        currently_on = bool(seq[-1][v_idx]) or bool(seq[-1][i_idx])
        by_current_support.setdefault(str(currently_on), []).append(y)

    def _prevalence(values_by_key):
        return {str(k): {"n": len(v), "prevalence": float(np.mean(v))} for k, v in values_by_key.items()}

    return {
        "n": len(rows),
        "overall_prevalence": float(labels.mean()),
        "prevalence_by_cutoff_index": _prevalence(by_cutoff),
        "prevalence_by_cardiac_subtype": _prevalence(by_cardiac),
        "prevalence_by_baseline_sofa_band": _prevalence(by_severity),
        "prevalence_by_current_support_state": _prevalence(by_current_support),
    }


def main() -> None:
    train_rows = load_dev_rows(ROOT, splits=("train",))
    validation_rows = load_dev_rows(ROOT, splits=("validation",))
    assert_no_test_rows(train_rows)
    assert_no_test_rows(validation_rows)
    dev_rows = train_rows + validation_rows

    part1 = _part1(train_rows, validation_rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    p1_path = OUT_DIR / "v1_validation_performance_summary.json"
    p1_path.write_bytes(canonical_json_bytes(part1))
    print("wrote", p1_path, sha256_file(p1_path))

    part2 = {
        "status": "DEVELOPMENT_DIAGNOSTIC_ONLY",
        "artifact_version": "performance_v2_phase1_target_distribution_diagnosis_v1",
        "scope": "train_plus_validation_rows_only_never_test",
        "recovery": _part2_recovery(dev_rows),
        "icu_stay_time": _part2_icu(dev_rows),
        "organ_support": _part2_support(dev_rows),
    }
    p2_path = OUT_DIR / "target_distribution_diagnosis_v1.json"
    p2_path.write_bytes(canonical_json_bytes(part2))
    print("wrote", p2_path, sha256_file(p2_path))


if __name__ == "__main__":
    main()
