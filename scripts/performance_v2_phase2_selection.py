"""Performance-v2 Phase 2, Parts 11-13, 16-17: bounded rescue-round check,
model-selection rule, performance comparison table, validation_selection_v2,
and phase2_manifest.

DEVELOPMENT_VALIDATION_ONLY. Reads only already-produced Phase-2 TRAIN-CV/
VALIDATION artifacts; performs no new TEST access. If this script's rescue
check finds a trigger, it prints instructions and exits without running the
rescue round itself -- the rescue round (Part 11) is executed by a
dedicated follow-up script only if actually triggered, keeping this
selection pass idempotent and re-runnable.
"""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.synthetic.config import canonical_json_bytes  # noqa: E402
from vedant_infra.hashing import sha256_file  # noqa: E402

OUT_DIR = ROOT / "artifacts/performance_v2/phase2"
PHASE1_DIR = ROOT / "artifacts/performance_v2/phase1"


def _load(path):
    return json.loads(Path(path).read_text())


def _rel_error_reduction(baseline, new_value):
    return (baseline - new_value) / baseline


def main() -> None:
    v1_perf = _load(PHASE1_DIR / "v1_validation_performance_summary.json")
    oracle = _load(PHASE1_DIR / "observable_oracle/observable_oracle_summary_v1.json")
    baselines1 = _load(PHASE1_DIR / "stronger_baselines_v1.json")

    rec_xgb = _load(OUT_DIR / "recovery/finalist_metrics_xgb.json")
    rec_gru = _load(OUT_DIR / "recovery/finalist_metrics_gru.json")
    icu_xgb = _load(OUT_DIR / "icu/finalist_metrics_xgb.json")
    icu_gru = _load(OUT_DIR / "icu/finalist_metrics_gru.json")
    support = _load(OUT_DIR / "support/finalist_metrics.json")
    gru_stability = _load(OUT_DIR / "gru_stability_v1.json") if (OUT_DIR / "gru_stability_v1.json").is_file() else None

    v1_recovery = v1_perf["recovery"]["best_xgb_validation"]
    v1_icu = v1_perf["icu_stay_time"]["best_gru_validation"]
    v1_support = v1_perf["organ_support"]["best_xgb_validation"]
    oracle_recovery24 = oracle["recovery"]["recovery24"]["metrics"]["mae"]
    oracle_recovery48 = oracle["recovery"]["recovery48"]["metrics"]["mae"]
    oracle_icu = oracle["icu_stay_time"]["metrics"]["median_absolute_error"]
    oracle_support = oracle["organ_support"]["metrics"]["auprc"]

    # --- RECOVERY selection ---
    all_rec_candidates = []
    for horizon in ("24", "48"):
        for finalist in rec_xgb["horizons"][horizon]["xgb_finalists"]:
            all_rec_candidates.append({"horizon": horizon, "system": "xgboost", **finalist})
        for finalist in rec_gru["horizons"][horizon]:
            all_rec_candidates.append({"horizon": horizon, "system": "gru", **finalist})

    def rec24_candidates():
        return [c for c in all_rec_candidates if c["horizon"] == "24"]

    best_rec24 = min(rec24_candidates(), key=lambda c: c["validation_metrics"]["mae"])
    v1_mae48 = v1_recovery["mae48"]
    rec48_candidates = [c for c in all_rec_candidates if c["horizon"] == "48"]
    # Prefer the same system/config family winner at 24h if its 48h companion doesn't majorly degrade;
    # else pick the best independent 48h candidate among finalists that also don't majorly degrade.
    acceptable_48 = [c for c in rec48_candidates if c["validation_metrics"]["mae"] <= v1_mae48 * 1.05]
    best_rec48 = min(acceptable_48, key=lambda c: c["validation_metrics"]["mae"]) if acceptable_48 else min(rec48_candidates, key=lambda c: c["validation_metrics"]["mae"])

    # --- ICU selection ---
    icu_candidates = []
    for finalist in icu_xgb["finalists"]:
        icu_candidates.append({"system": finalist.get("system", "xgb_log1p"), **finalist})
    for finalist in icu_gru["finalists"]:
        icu_candidates.append({"system": "gru", **finalist})
    best_icu = min(icu_candidates, key=lambda c: c["validation_metrics"]["median_absolute_error"])

    # --- SUPPORT selection ---
    best_support = max(support["finalists"], key=lambda c: c["validation_metrics"]["auprc"])

    # --- Rescue-round triggers (Part 11) ---
    rescue_triggers = {}
    rec24_error_excess = (best_rec24["validation_metrics"]["mae"] - oracle_recovery24) / oracle_recovery24
    rescue_triggers["recovery24"] = rec24_error_excess > 0.10
    rec48_oracle_excess = (best_rec48["validation_metrics"]["mae"] - oracle_recovery48) / oracle_recovery48
    rescue_triggers["recovery48"] = rec48_oracle_excess > 0.10
    icu_error_excess = (best_icu["validation_metrics"]["median_absolute_error"] - oracle_icu) / oracle_icu
    rescue_triggers["icu_stay_time"] = icu_error_excess > 0.10
    support_gap = oracle_support - best_support["validation_metrics"]["auprc"]
    rescue_triggers["organ_support"] = support_gap > 0.03

    payload = {
        "status": "DEVELOPMENT_VALIDATION_ONLY", "artifact_version": "performance_v2_phase2_rescue_trigger_check_v1",
        "rescue_triggers": rescue_triggers, "rescue_evidence": {
            "recovery24_error_excess_over_oracle": rec24_error_excess, "recovery48_error_excess_over_oracle": rec48_oracle_excess,
            "icu_error_excess_over_oracle": icu_error_excess, "support_auprc_gap_to_oracle": support_gap,
        },
        "best_candidates_before_rescue": {
            "recovery24": {"system": best_rec24["system"], "config_hash": best_rec24.get("config_hash"), "mae": best_rec24["validation_metrics"]["mae"]},
            "recovery48": {"system": best_rec48["system"], "config_hash": best_rec48.get("config_hash"), "mae": best_rec48["validation_metrics"]["mae"]},
            "icu_stay_time": {"system": best_icu["system"], "median_ae": best_icu["validation_metrics"]["median_absolute_error"]},
            "organ_support": {"config_hash": best_support.get("config_hash"), "auprc": best_support["validation_metrics"]["auprc"]},
        },
    }
    (OUT_DIR / "rescue_trigger_check_v1.json").write_bytes(canonical_json_bytes(payload))
    print(json.dumps(payload, indent=2, default=str))
    any_triggered = any(rescue_triggers.values())
    print("ANY_RESCUE_TRIGGERED:", any_triggered)
    if any_triggered:
        print("Rescue round required; run performance_v2_phase2_rescue.py before finalizing selection.")
        return

    # --- Part 13: performance comparison table -------------------------
    comparison = {
        "status": "DEVELOPMENT_VALIDATION_ONLY", "artifact_version": "performance_v2_phase2_performance_comparison_v1",
        "rows": [
            {
                "task": "recovery_24h", "metric": "mae",
                "v1_selected": v1_recovery["mae24"], "phase1_strong_simple_baseline": baselines1["recovery"]["recovery24"]["R2_ridge_diagnostic_features"]["metrics"]["mae"],
                "phase1_observable_oracle": oracle_recovery24, "best_phase2_xgb": min((c["validation_metrics"]["mae"] for c in rec24_candidates() if c["system"] == "xgboost"), default=None),
                "best_phase2_gru": min((c["validation_metrics"]["mae"] for c in rec24_candidates() if c["system"] == "gru"), default=None),
                "selected_phase2_system": best_rec24["validation_metrics"]["mae"],
                "absolute_change_from_v1": best_rec24["validation_metrics"]["mae"] - v1_recovery["mae24"],
                "relative_error_reduction_from_v1": _rel_error_reduction(v1_recovery["mae24"], best_rec24["validation_metrics"]["mae"]),
                "remaining_gap_to_oracle": best_rec24["validation_metrics"]["mae"] - oracle_recovery24,
            },
            {
                "task": "recovery_48h", "metric": "mae",
                "v1_selected": v1_recovery["mae48"], "phase1_strong_simple_baseline": baselines1["recovery"]["recovery48"]["R2_ridge_diagnostic_features"]["metrics"]["mae"],
                "phase1_observable_oracle": oracle_recovery48, "best_phase2_xgb": min((c["validation_metrics"]["mae"] for c in rec48_candidates if c["system"] == "xgboost"), default=None),
                "best_phase2_gru": min((c["validation_metrics"]["mae"] for c in rec48_candidates if c["system"] == "gru"), default=None),
                "selected_phase2_system": best_rec48["validation_metrics"]["mae"],
                "absolute_change_from_v1": best_rec48["validation_metrics"]["mae"] - v1_recovery["mae48"],
                "relative_error_reduction_from_v1": _rel_error_reduction(v1_recovery["mae48"], best_rec48["validation_metrics"]["mae"]),
                "remaining_gap_to_oracle": best_rec48["validation_metrics"]["mae"] - oracle_recovery48,
            },
            {
                "task": "icu_stay_time", "metric": "median_absolute_error_hours",
                "v1_selected": v1_icu["median_absolute_error_hours"], "phase1_strong_simple_baseline": baselines1["icu_stay_time"]["I2_conditional_median_fine_grid"]["metrics"]["median_absolute_error"],
                "phase1_observable_oracle": oracle_icu, "best_phase2_xgb": min((c["validation_metrics"]["median_absolute_error"] for c in icu_candidates if c["system"] != "gru"), default=None),
                "best_phase2_gru": min((c["validation_metrics"]["median_absolute_error"] for c in icu_candidates if c["system"] == "gru"), default=None),
                "selected_phase2_system": best_icu["validation_metrics"]["median_absolute_error"],
                "absolute_change_from_v1": best_icu["validation_metrics"]["median_absolute_error"] - v1_icu["median_absolute_error_hours"],
                "relative_error_reduction_from_v1": _rel_error_reduction(v1_icu["median_absolute_error_hours"], best_icu["validation_metrics"]["median_absolute_error"]),
                "remaining_gap_to_oracle": best_icu["validation_metrics"]["median_absolute_error"] - oracle_icu,
            },
            {
                "task": "organ_support", "metric": "auprc",
                "v1_selected": v1_support["auprc"], "phase1_strong_simple_baseline": baselines1["organ_support"]["logistic_diagnostic_features"]["metrics"]["auprc"],
                "phase1_observable_oracle": oracle_support, "best_phase2_xgb": max((c["validation_metrics"]["auprc"] for c in support["finalists"]), default=None),
                "best_phase2_gru": None,
                "selected_phase2_system": best_support["validation_metrics"]["auprc"],
                "absolute_change_from_v1": best_support["validation_metrics"]["auprc"] - v1_support["auprc"],
                "absolute_auprc_gain_from_v1": best_support["validation_metrics"]["auprc"] - v1_support["auprc"],
                "remaining_gap_to_oracle": oracle_support - best_support["validation_metrics"]["auprc"],
            },
        ],
        "no_overall_cross_task_score": True,
    }
    (OUT_DIR / "performance_comparison_v2.json").write_bytes(canonical_json_bytes(comparison))
    print("wrote performance_comparison_v2.json")

    # --- Part 17: validation_selection_v2.json --------------------------
    def _artifact_ref(candidate):
        return {"ref": candidate.get("prediction_artifact"), "sha256": candidate.get("prediction_artifact_sha256")}

    selection = {
        "status": "V2_VALIDATION_SELECTION_COMPLETE", "artifact_version": "performance_v2_phase2_validation_selection_v1",
        "decision_authority": "USER_DELEGATED_AI_PROJECT_DECISION",
        "note": "Validation-selected models only. NOT a final model freeze -- Phase 3 performs the final TRAIN+VALIDATION DEV refit.",
        "tasks": {
            "recovery24": {
                "selected_family": best_rec24["system"], "config_hash": best_rec24.get("config_hash"), "feature_variant": best_rec24["config"]["feature_variant"],
                "hyperparameters": best_rec24["config"], "objective_or_loss": best_rec24["config"].get("objective") or best_rec24["config"].get("loss_name"),
                "training_weight_mode": best_rec24["config"].get("weight_mode"), "validation_metrics": best_rec24["validation_metrics"],
                "validation_counts": best_rec24["validation_counts"], "validation_prediction_artifact": _artifact_ref(best_rec24),
                "v1_comparison": {"v1_mae24": v1_recovery["mae24"], "relative_error_reduction": _rel_error_reduction(v1_recovery["mae24"], best_rec24["validation_metrics"]["mae"])},
                "observable_oracle_comparison": {"oracle_mae24": oracle_recovery24, "remaining_gap": best_rec24["validation_metrics"]["mae"] - oracle_recovery24},
            },
            "recovery48": {
                "selected_family": best_rec48["system"], "config_hash": best_rec48.get("config_hash"), "feature_variant": best_rec48["config"]["feature_variant"],
                "hyperparameters": best_rec48["config"], "objective_or_loss": best_rec48["config"].get("objective") or best_rec48["config"].get("loss_name"),
                "training_weight_mode": best_rec48["config"].get("weight_mode"), "validation_metrics": best_rec48["validation_metrics"],
                "validation_counts": best_rec48["validation_counts"], "validation_prediction_artifact": _artifact_ref(best_rec48),
                "v1_comparison": {"v1_mae48": v1_recovery["mae48"], "relative_error_reduction": _rel_error_reduction(v1_recovery["mae48"], best_rec48["validation_metrics"]["mae"])},
                "observable_oracle_comparison": {"oracle_mae48": oracle_recovery48, "remaining_gap": best_rec48["validation_metrics"]["mae"] - oracle_recovery48},
            },
            "icu_stay_time": {
                "selected_system": best_icu["system"], "config_hash": best_icu.get("config_hash"), "config": best_icu.get("config"),
                "validation_metrics": best_icu["validation_metrics"], "validation_counts": best_icu["validation_counts"],
                "validation_prediction_artifact": _artifact_ref(best_icu),
                "v1_comparison": {"v1_median_ae": v1_icu["median_absolute_error_hours"], "relative_error_reduction": _rel_error_reduction(v1_icu["median_absolute_error_hours"], best_icu["validation_metrics"]["median_absolute_error"])},
                "observable_oracle_comparison": {"oracle_median_ae": oracle_icu, "remaining_gap": best_icu["validation_metrics"]["median_absolute_error"] - oracle_icu},
                "conditional_prior_considered": True, "aft_considered": True,
            },
            "organ_support": {
                "selected_family": "xgboost", "config_hash": best_support.get("config_hash"), "feature_variant": best_support["config"]["feature_variant"],
                "hyperparameters": best_support["config"], "validation_metrics": best_support["validation_metrics"], "validation_counts": best_support["validation_counts"],
                "validation_prediction_artifact": _artifact_ref(best_support),
                "v1_comparison": {"v1_auprc": v1_support["auprc"], "absolute_auprc_gain": best_support["validation_metrics"]["auprc"] - v1_support["auprc"]},
                "observable_oracle_comparison": {"oracle_auprc": oracle_support, "remaining_gap": oracle_support - best_support["validation_metrics"]["auprc"]},
                "requires_v2_recalibration": True,
                "calibration_note": "Model selection only; final calibrator/threshold deferred to Phase 3.",
            },
        },
        "gru_stability": gru_stability,
        "rescue_round_triggered": any_triggered,
    }
    (OUT_DIR / "validation_selection_v2.json").write_bytes(canonical_json_bytes(selection))
    print("wrote validation_selection_v2.json")

    # --- Part 16: phase2_manifest.json -----------------------------------
    import subprocess as _subprocess
    code_commit = _subprocess.check_output(("git", "rev-parse", "HEAD"), cwd=ROOT, text=True).strip()
    manifest = {
        "status": "PHASE2_COMPLETE", "artifact_version": "performance_v2_phase2_manifest_v1",
        "code_commit": code_commit,
        "artifacts": {
            "feature_contract": "configs/performance_v2/feature_contract_v2.json",
            "fold_assignment": "artifacts/performance_v2/phase2/train_cv_fold_assignment_v1.csv",
            "recovery_search_manifest": "artifacts/performance_v2/phase2/recovery/search_manifest.json",
            "icu_search_manifest": "artifacts/performance_v2/phase2/icu/search_manifest.json",
            "support_search_manifest": "artifacts/performance_v2/phase2/support/search_manifest.json",
            "gru_stability": "artifacts/performance_v2/phase2/gru_stability_v1.json",
            "rescue_trigger_check": "artifacts/performance_v2/phase2/rescue_trigger_check_v1.json",
            "performance_comparison": "artifacts/performance_v2/phase2/performance_comparison_v2.json",
            "validation_selection": "artifacts/performance_v2/phase2/validation_selection_v2.json",
        },
        "v1_test_accessed": False, "generator_modified": False,
        "phase3_ready": True,
        "phase3_handoff": {
            "recovery24_selected": {"family": best_rec24["system"], "config_hash": best_rec24.get("config_hash")},
            "recovery48_selected": {"family": best_rec48["system"], "config_hash": best_rec48.get("config_hash")},
            "icu_stay_time_selected": {"system": best_icu["system"], "config_hash": best_icu.get("config_hash")},
            "organ_support_selected": {"family": "xgboost", "config_hash": best_support.get("config_hash"), "requires_v2_recalibration": True},
            "not_yet_done": [
                "final TRAIN+VALIDATION DEV refit", "cross-fitted support calibration/threshold",
                "fresh v2 test cohort generation", "v2 test split freeze", "sealed v2 test evaluation",
            ],
        },
    }
    (OUT_DIR / "phase2_manifest.json").write_bytes(canonical_json_bytes(manifest))
    print("wrote phase2_manifest.json")
    print("PERFORMANCE_V2_PHASE2_COMPLETE")


if __name__ == "__main__":
    main()
