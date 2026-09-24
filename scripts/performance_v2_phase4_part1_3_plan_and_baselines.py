"""Performance-v2 Phase 4, Parts 1 and 3: freeze the final evaluation plan
and the naive baselines BEFORE any fresh-test access. Reads DEV rows only
(performance_v2.data_loading enforces train/validation-only access) -- never
opens the fresh cohort.
"""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.synthetic.config import canonical_json_bytes  # noqa: E402
from evaluation.weighted_stats import weighted_percentile  # noqa: E402
from evaluation.weights import compute_stay_weights  # noqa: E402
from performance_v2.data_loading import load_dev_rows  # noqa: E402
from performance_v2.targets import icu_hours_target, icu_rows, support_rows  # noqa: E402
from vedant_infra.hashing import sha256_file  # noqa: E402

PHASE4_DIR = ROOT / "artifacts/performance_v2/phase4"


def _ref(path: str) -> dict:
    return {"ref": path, "sha256": sha256_file(ROOT / path)}


def main() -> None:
    code_commit = subprocess.check_output(("git", "rev-parse", "HEAD"), cwd=ROOT, text=True).strip()

    train_rows = load_dev_rows(ROOT, splits=("train",))
    val_rows = load_dev_rows(ROOT, splits=("validation",))
    dev_rows = train_rows + val_rows

    # --- Part 3: naive baselines (DEV only) ---
    icu_dev = icu_rows(dev_rows)
    icu_hours = [icu_hours_target(r) for r in icu_dev]
    icu_weights = compute_stay_weights([r["stay_id"] for r in icu_dev], [True] * len(icu_dev)).weights
    icu_naive_median_hours = weighted_percentile(icu_hours, icu_weights, 0.5)

    support_dev = support_rows(dev_rows)
    support_labels = [float(r["organ_support_label"]) for r in support_dev]
    support_weights = compute_stay_weights([r["stay_id"] for r in support_dev], [True] * len(support_dev)).weights
    total_weight = sum(support_weights)
    support_dev_prevalence = sum(l * w for l, w in zip(support_labels, support_weights)) / total_weight

    baselines = {
        "status": "V2_NAIVE_BASELINES_FROZEN_PRE_ACCESS",
        "artifact_version": "performance_v2_phase4_naive_baselines_v1",
        "recovery24": {"definition": "constant_zero_delta_sofa", "constant_prediction": 0.0},
        "recovery48": {"definition": "constant_zero_delta_sofa", "constant_prediction": 0.0},
        "icu_stay_time": {
            "definition": "DEV_stay_balanced_weighted_median_of_remaining_ICU_hours_in_hours_space",
            "constant_prediction_hours": icu_naive_median_hours,
            "n_dev_icu_eligible_rows": len(icu_dev),
            "n_dev_icu_eligible_stays": len({r["stay_id"] for r in icu_dev}),
            "convention_note": (
                "Computed in raw hours space (weighted_percentile over "
                "expm1(icu_time_log1p), q=0.5), not log space, per the explicit "
                "Phase-4 baseline instruction; documented here as the single "
                "consistent definition used throughout Phase 4."
            ),
        },
        "organ_support": {
            "definition": "DEV_eligible_stay_balanced_prevalence_as_constant_probability",
            "constant_probability": support_dev_prevalence,
            "n_dev_support_eligible_rows": len(support_dev),
            "n_dev_support_eligible_stays": len({r["stay_id"] for r in support_dev}),
        },
        "dev_subject_manifest": _ref("artifacts/performance_v2/phase3/dev_subject_manifest_v1.json"),
    }
    PHASE4_DIR.mkdir(parents=True, exist_ok=True)
    baselines_path = PHASE4_DIR / "naive_baselines_v2.json"
    baselines_path.write_bytes(canonical_json_bytes(baselines))
    print("wrote", baselines_path, sha256_file(baselines_path))

    # --- Prespecified error-analysis slices (DEV-derived boundaries, frozen now) ---
    def density(row):
        mask = row["observation_mask"]
        total = sum(len(bin_row) for bin_row in mask)
        observed = sum(1 for bin_row in mask for v in bin_row if v)
        return observed / total

    dens = [density(r) for r in dev_rows]
    dens_weights = compute_stay_weights([r["stay_id"] for r in dev_rows], [True] * len(dev_rows)).weights
    density_t33 = weighted_percentile(dens, dens_weights, 1.0 / 3.0)
    density_t67 = weighted_percentile(dens, dens_weights, 2.0 / 3.0)

    sofa_rows = [r for r in dev_rows if r.get("baseline_sofa") is not None]
    sofa_vals = [float(r["baseline_sofa"]) for r in sofa_rows]
    sofa_weights = compute_stay_weights([r["stay_id"] for r in sofa_rows], [True] * len(sofa_rows)).weights
    sofa_t33 = weighted_percentile(sofa_vals, sofa_weights, 1.0 / 3.0)
    sofa_t67 = weighted_percentile(sofa_vals, sofa_weights, 2.0 / 3.0)

    icu_q25 = weighted_percentile(icu_hours, icu_weights, 0.25)
    icu_q50 = weighted_percentile(icu_hours, icu_weights, 0.50)
    icu_q75 = weighted_percentile(icu_hours, icu_weights, 0.75)

    slices = {
        "status": "PREDECLARED_BEFORE_ACCESS",
        "no_post_hoc_subgroup_hunting": True,
        "cutoff_timing": {
            "field": "grid_index",
            "bands": {"EARLY": [0, 3], "MID": [4, 7], "LATE": [8, 11]},
            "band_rule": "inclusive_integer_range",
        },
        "cardiac_subtype": {
            "field": "cardiac_condition_group",
            "categories": ["SYNTHETIC_ARRHYTHMIA", "SYNTHETIC_HF", "SYNTHETIC_ISCHEMIC"],
        },
        "current_sofa_severity_band": {
            "field": "baseline_sofa",
            "applies_only_to_rows_with_defined_baseline_sofa": True,
            "dev_derived_tertile_boundaries": {"t33": sofa_t33, "t67": sofa_t67},
            "bands": {"LOW": "<= t33", "MODERATE": "(t33, t67]", "HIGH": "> t67"},
        },
        "observation_density_band": {
            "field": "observation_mask_fraction_observed",
            "dev_derived_tertile_boundaries": {"t33": density_t33, "t67": density_t67},
            "bands": {"LOW": "<= t33", "MODERATE": "(t33, t67]", "HIGH": "> t67"},
        },
        "icu_true_remaining_time_quartile": {
            "field": "true_remaining_hours",
            "applies_to": "icu_stay_time",
            "dev_derived_quartile_boundaries": {"q25": icu_q25, "q50": icu_q50, "q75": icu_q75},
            "bands": {"Q1": "<= q25", "Q2": "(q25, q50]", "Q3": "(q50, q75]", "Q4": "> q75"},
        },
        "recovery_true_delta_sign": {
            "field": "target_delta",
            "applies_to": ["recovery24", "recovery48"],
            "categories": {"NEGATIVE": "< 0", "ZERO": "== 0", "POSITIVE": "> 0"},
        },
    }

    # --- Part 1: final evaluation plan ---
    plan = {
        "status": "V2_FINAL_EVALUATION_PLAN_FROZEN_PRE_ACCESS",
        "artifact_version": "performance_v2_phase4_final_evaluation_plan_v1",
        "decision_authority": "USER_DELEGATED_AI_PROJECT_DECISION",
        "parents": {
            "selected_models_v2": _ref("artifacts/performance_v2/phase3/selected_models_v2.json"),
            "v2_model_freeze": _ref("artifacts/performance_v2/governance/v2_model_freeze_v1.json"),
            "v2_fresh_test_freeze": _ref("artifacts/performance_v2/governance/v2_fresh_test_freeze_v1.json"),
            "generator_equivalence_audit": _ref("artifacts/performance_v2/governance/generator_equivalence_audit_v1.json"),
            "prediction_contract_v2": _ref("artifacts/performance_v2/phase3/prediction_contract_v2.json"),
            "isotonic_support_v2": _ref("artifacts/performance_v2/phase3/calibration/isotonic_support_v2.json"),
            "support_threshold_v2": _ref("artifacts/performance_v2/phase3/thresholds/support_threshold_v2.json"),
            "environment_lock": _ref("artifacts/performance_v2/phase3/environment_lock_v1.json"),
            "naive_baselines_v2": _ref("artifacts/performance_v2/phase4/naive_baselines_v2.json"),
        },
        "code_commit": code_commit,
        "model_contract": {
            "recovery24": {"family": "xgboost", "feature_variant": "B_MIN", "n_estimators": 169},
            "recovery48": {"family": "xgboost", "feature_variant": "B_MIN", "n_estimators": 316},
            "icu_stay_time": {"family": "xgboost", "feature_variant": "B_PLUS_F", "n_estimators": 428, "postprocess": "expm1(clamp_min(raw,0))"},
            "organ_support": {"family": "xgboost", "feature_variant": "B_FULL", "n_estimators": 56},
        },
        "support_postprocess_chain": [
            "raw XGBoost probability",
            "isotonic_support_v2 (frozen, no refit)",
            "calibrated probability",
            "threshold 0.39781983118092895 (comparator >=), frozen, no reselection",
        ],
        "metrics": {
            "recovery24": {"primary": "stay_balanced_mae", "secondary": ["rmse", "weighted_median_ae"], "no_clipping": True},
            "recovery48": {"primary": "stay_balanced_mae", "secondary": ["rmse", "weighted_median_ae"], "no_chaining_from_24h": True},
            "icu_stay_time": {"primary": "stay_balanced_weighted_median_ae_hours", "secondary": ["mae_hours", "rmse_hours"], "domain": "postprocessed_hours"},
            "organ_support": {
                "raw_probability_metrics": ["auprc", "auroc", "brier"],
                "calibrated_probability_metrics": ["auprc", "auroc", "brier"],
                "threshold_metrics": ["f1", "precision", "recall", "specificity", "weighted_TP", "weighted_FP", "weighted_TN", "weighted_FN"],
                "primary_discrimination_metric": "calibrated_auprc",
            },
        },
        "bootstrap_config": {
            "n_bootstrap": 2000,
            "seed": 20260924,
            "ci_level": 0.95,
            "ci_quantiles": [0.025, 0.975],
            "resampling_unit": "stay_id",
            "method": "stay_cluster_percentile_bootstrap (evaluation.bootstrap.grouped_bootstrap, reused unmodified from Benchmark-v1)",
            "undefined_metric_handling": "exclude_for_that_metric_only_report_n_requested_n_valid_n_invalid",
            "seed_is_new_phase4_evaluation_seed_distinct_from_generator_and_training_seeds": True,
        },
        "prespecified_error_analysis_slices": slices,
        "prediction_schema": {
            "recovery24": ["subject_id", "stay_id", "prediction_time", "grid_index", "target_delta24", "prediction_delta24", "eligible", "model_sha256", "feature_contract_sha256", "fresh_cohort_freeze_sha256"],
            "recovery48": ["subject_id", "stay_id", "prediction_time", "grid_index", "target_delta48", "prediction_delta48", "eligible", "model_sha256", "feature_contract_sha256", "fresh_cohort_freeze_sha256"],
            "icu_stay_time": ["subject_id", "stay_id", "prediction_time", "grid_index", "true_remaining_hours", "raw_log_prediction", "prediction_hours", "eligible", "model_sha256", "feature_contract_sha256", "fresh_cohort_freeze_sha256"],
            "organ_support": ["subject_id", "stay_id", "prediction_time", "grid_index", "label", "raw_probability", "calibrated_probability", "frozen_threshold", "predicted_alert", "eligible", "model_sha256", "feature_contract_sha256", "fresh_cohort_freeze_sha256"],
        },
        "generalization_comparison_tasks": ["recovery24", "recovery48", "icu_stay_time", "organ_support"],
        "no_optimization_statement": (
            "No model, feature, hyperparameter, calibration, or threshold change is "
            "permitted after this plan is frozen. Exactly one governed fresh-test "
            "access event is authorized; all analysis after that event operates "
            "only on saved prediction artifacts."
        ),
    }
    plan_path = PHASE4_DIR / "final_evaluation_plan_v1.json"
    plan_path.write_bytes(canonical_json_bytes(plan))
    print("wrote", plan_path, sha256_file(plan_path))
    print("V2_FINAL_EVALUATION_PLAN_FROZEN_PRE_ACCESS")


if __name__ == "__main__":
    main()
