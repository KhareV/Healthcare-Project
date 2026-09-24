"""Performance-v2 Phase 4, Part 20: compact final performance summary table,
assembled only from already-saved Phase-4 metric/bootstrap artifacts (plus
historical v1 test values, allowed here for context only -- Phase-4
selection is already frozen and v2 fresh-test has already been evaluated).
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.synthetic.config import canonical_json_bytes  # noqa: E402
from vedant_infra.hashing import sha256_file  # noqa: E402

PHASE4_DIR = ROOT / "artifacts/performance_v2/phase4"

# Historical v1 final-test values (artifacts/final_test/metrics/metrics_v1.json,
# partition="test", "selected" models), for descriptive context only -- the v1
# TEST cohort and the fresh V2 test cohort are NOT the same sample (different
# seed, subject namespace, and size); see docs/performance_v2 for the caveat.
V1_HISTORICAL_TEST_REF = "artifacts/final_test/metrics/metrics_v1.json"
V1_HISTORICAL_TEST = {
    "recovery24_mae": 1.121102021240419,
    "recovery48_mae": 1.5423799902108615,
    "icu_median_ae": 9.479719411307386,
    "support_auprc": 0.633735668141501,
}


def main() -> None:
    metrics = json.loads((PHASE4_DIR / "metrics/final_metrics_v2.json").read_text())
    naive = json.loads((PHASE4_DIR / "metrics/naive_comparison_v2.json").read_text())
    bootstrap = json.loads((PHASE4_DIR / "bootstrap/final_bootstrap_v2.json").read_text())
    validation_selection = json.loads((ROOT / "artifacts/performance_v2/phase2/validation_selection_v2.json").read_text())

    def ci(entry):
        return [entry["ci_lower"], entry["ci_upper"]]

    rows = [
        {
            "task": "recovery24", "metric": "MAE",
            "v1_final_test_historical": V1_HISTORICAL_TEST["recovery24_mae"],
            "v2_phase2_validation": validation_selection["tasks"]["recovery24"]["validation_metrics"]["mae"],
            "v2_fresh_test": metrics["recovery24"]["mae"],
            "v2_fresh_test_95ci": ci(bootstrap["recovery24"]["mae"]),
            "v2_naive_baseline": naive["recovery24"]["naive_mae"],
            "improvement_over_v2_naive": naive["recovery24"]["relative_improvement"],
        },
        {
            "task": "recovery48", "metric": "MAE",
            "v1_final_test_historical": V1_HISTORICAL_TEST["recovery48_mae"],
            "v2_phase2_validation": validation_selection["tasks"]["recovery48"]["validation_metrics"]["mae"],
            "v2_fresh_test": metrics["recovery48"]["mae"],
            "v2_fresh_test_95ci": ci(bootstrap["recovery48"]["mae"]),
            "v2_naive_baseline": naive["recovery48"]["naive_mae"],
            "improvement_over_v2_naive": naive["recovery48"]["relative_improvement"],
        },
        {
            "task": "icu_stay_time", "metric": "weighted_median_AE_hours",
            "v1_final_test_historical": V1_HISTORICAL_TEST["icu_median_ae"],
            "v2_phase2_validation": validation_selection["tasks"]["icu_stay_time"]["validation_metrics"]["median_absolute_error"],
            "v2_fresh_test": metrics["icu_stay_time"]["median_absolute_error"],
            "v2_fresh_test_95ci": ci(bootstrap["icu_stay_time"]["median_absolute_error"]),
            "v2_naive_baseline": naive["icu_stay_time"]["naive_median_ae"],
            "improvement_over_v2_naive": naive["icu_stay_time"]["relative_improvement"],
        },
        {
            "task": "organ_support", "metric": "calibrated_AUPRC",
            "v1_final_test_historical": V1_HISTORICAL_TEST["support_auprc"],
            "v2_phase2_validation": validation_selection["tasks"]["organ_support"]["validation_metrics"]["auprc"],
            "v2_fresh_test": metrics["organ_support_calibrated"]["auprc"],
            "v2_fresh_test_95ci": ci(bootstrap["organ_support_calibrated"]["auprc"]),
            "v2_naive_baseline": naive["organ_support"]["naive_auprc"],
            "improvement_over_v2_naive": naive["organ_support"]["relative_improvement"],
        },
    ]

    summary = {
        "status": "V2_FINAL_PERFORMANCE_SUMMARY",
        "artifact_version": "performance_v2_phase4_final_performance_summary_v1",
        "note": "v1_final_test_historical is provided for descriptive context only; the v1 and v2 test cohorts are NOT the same sample (different seed, subject namespace, and cohort size) and are not directly comparable as a controlled experiment.",
        "v1_historical_test_source": {"ref": V1_HISTORICAL_TEST_REF, "sha256": sha256_file(ROOT / V1_HISTORICAL_TEST_REF)},
        "rows": rows,
    }
    summary_path = PHASE4_DIR / "final_performance_summary_v2.json"
    summary_path.write_bytes(canonical_json_bytes(summary))
    print("wrote", summary_path, sha256_file(summary_path))
    for row in rows:
        print(row["task"], row["metric"], "v2_test=", row["v2_fresh_test"], "CI=", row["v2_fresh_test_95ci"], "naive=", row["v2_naive_baseline"], "improvement=", row["improvement_over_v2_naive"])


if __name__ == "__main__":
    main()
