"""Performance-v2 Phase 3, Parts 11-13: selected_models_v2.json, the
prediction contract, and the pre-generation scientific freeze
(v2_model_freeze_v1.json). Run AFTER Parts 1-10; run BEFORE any fresh
cohort generation.
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy
import scipy
import sklearn
import xgboost

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.synthetic.config import canonical_json_bytes  # noqa: E402
from vedant_infra.hashing import sha256_file  # noqa: E402

PHASE3_DIR = ROOT / "artifacts/performance_v2/phase3"
GOV_DIR = ROOT / "artifacts/performance_v2/governance"


def _ref(path: str) -> dict:
    return {"ref": path, "sha256": sha256_file(ROOT / path)}


def main() -> None:
    refit_manifest = json.loads((PHASE3_DIR / "models/final_refit_manifest_v1.json").read_text())
    budget = json.loads((PHASE3_DIR / "final_training_budget_v1.json").read_text())
    selection = json.loads((ROOT / "artifacts/performance_v2/phase2/validation_selection_v2.json").read_text())
    calibrator_path = PHASE3_DIR / "calibration/isotonic_support_v2.json"
    threshold_path = PHASE3_DIR / "thresholds/support_threshold_v2.json"
    code_commit = subprocess.check_output(("git", "rev-parse", "HEAD"), cwd=ROOT, text=True).strip()

    # --- Part 11: selected_models_v2.json ---
    selected_models = {
        "status": "V2_SELECTED_MODELS_FROZEN_PRE_TEST",
        "artifact_version": "performance_v2_phase3_selected_models_v2",
        "decision_authority": "USER_DELEGATED_AI_PROJECT_DECISION",
        "tasks": {
            "recovery24": {
                "family": "xgboost", "selected_phase2_candidate_config_hash": selection["tasks"]["recovery24"]["config_hash"],
                "feature_variant": "B_MIN", "hyperparameters": refit_manifest["tasks"]["recovery24"]["hyperparameters"],
                "final_training_budget": budget["tasks"]["recovery24"]["final_n_estimators"],
                "preprocessor_ref": refit_manifest["tasks"]["recovery24"]["context_normalization_ref"],
                "preprocessor_sha256": sha256_file(ROOT / refit_manifest["tasks"]["recovery24"]["context_normalization_ref"]),
                "model_ref": refit_manifest["tasks"]["recovery24"]["model_path"], "model_sha256": refit_manifest["tasks"]["recovery24"]["model_sha256"],
            },
            "recovery48": {
                "family": "xgboost", "selected_phase2_candidate_config_hash": selection["tasks"]["recovery48"]["config_hash"],
                "feature_variant": "B_MIN", "hyperparameters": refit_manifest["tasks"]["recovery48"]["hyperparameters"],
                "final_training_budget": budget["tasks"]["recovery48"]["final_n_estimators"],
                "preprocessor_ref": refit_manifest["tasks"]["recovery48"]["context_normalization_ref"],
                "preprocessor_sha256": sha256_file(ROOT / refit_manifest["tasks"]["recovery48"]["context_normalization_ref"]),
                "model_ref": refit_manifest["tasks"]["recovery48"]["model_path"], "model_sha256": refit_manifest["tasks"]["recovery48"]["model_sha256"],
            },
            "icu_stay_time": {
                "family": "xgboost", "selected_phase2_candidate_config_hash": selection["tasks"]["icu_stay_time"]["config_hash"],
                "feature_variant": "B_PLUS_F", "raw_output_meaning": "log1p_remaining_current_icu_hours",
                "postprocess": "expm1(clamp_min(raw,0))", "hyperparameters": refit_manifest["tasks"]["icu_stay_time"]["hyperparameters"],
                "final_training_budget": budget["tasks"]["icu_stay_time"]["final_n_estimators"],
                "preprocessor_ref": refit_manifest["tasks"]["icu_stay_time"]["context_normalization_ref"],
                "preprocessor_sha256": sha256_file(ROOT / refit_manifest["tasks"]["icu_stay_time"]["context_normalization_ref"]),
                "model_ref": refit_manifest["tasks"]["icu_stay_time"]["model_path"], "model_sha256": refit_manifest["tasks"]["icu_stay_time"]["model_sha256"],
            },
            "organ_support": {
                "family": "xgboost", "selected_phase2_candidate_config_hash": selection["tasks"]["organ_support"]["config_hash"],
                "feature_variant": "B_FULL", "hyperparameters": refit_manifest["tasks"]["organ_support"]["hyperparameters"],
                "final_training_budget": budget["tasks"]["organ_support"]["final_n_estimators"],
                "preprocessor_ref": refit_manifest["tasks"]["organ_support"]["context_normalization_ref"],
                "preprocessor_sha256": sha256_file(ROOT / refit_manifest["tasks"]["organ_support"]["context_normalization_ref"]),
                "model_ref": refit_manifest["tasks"]["organ_support"]["model_path"], "model_sha256": refit_manifest["tasks"]["organ_support"]["model_sha256"],
                "isotonic_calibrator_ref": "artifacts/performance_v2/phase3/calibration/isotonic_support_v2.json",
                "isotonic_calibrator_sha256": sha256_file(calibrator_path),
                "threshold_ref": "artifacts/performance_v2/phase3/thresholds/support_threshold_v2.json",
                "threshold_sha256": sha256_file(threshold_path),
            },
        },
        "parents": {
            "phase1_strategy": _ref("artifacts/performance_v2/phase1/selected_v2_strategy_v1.json"),
            "phase2_validation_selection": _ref("artifacts/performance_v2/phase2/validation_selection_v2.json"),
            "feature_contract": _ref("configs/performance_v2/feature_contract_v2.json"),
            "dev_subject_manifest": _ref("artifacts/performance_v2/phase3/dev_subject_manifest_v1.json"),
            "dev_crossfit_folds": _ref("artifacts/performance_v2/phase3/dev_crossfit_fold_assignment_v1.csv"),
            "generator_config": {"ref": "configs/synthetic/final_benchmark_v1.json", "sha256": sha256_file(ROOT / "configs/synthetic/final_benchmark_v1.json")},
        },
        "code_commit": code_commit,
    }
    selected_models_path = PHASE3_DIR / "selected_models_v2.json"
    selected_models_path.write_bytes(canonical_json_bytes(selected_models))
    print("wrote", selected_models_path, sha256_file(selected_models_path))

    # --- Part 12: prediction contract ---
    prediction_contract = {
        "status": "V2_PREDICTION_CONTRACT_FROZEN",
        "artifact_version": "performance_v2_phase3_prediction_contract_v1",
        "recovery": {"raw_output": "raw DeltaSOFA24/DeltaSOFA48", "postprocess": "none -- raw unclipped value is the scientific prediction"},
        "icu_stay_time": {"raw_output": "log1p(remaining_current_icu_hours)", "postprocess": "expm1(clamp_min(raw, 0))"},
        "organ_support": {
            "raw_output": "xgboost raw probability",
            "postprocess_chain": [
                "raw_probability -> isotonic_support_v2.json (sklearn IsotonicRegression, y_min=0, y_max=1, out_of_bounds='clip')",
                "-> calibrated_probability",
                "-> compare to support_threshold_v2.json's threshold_value with comparator '>=' -> alert state",
            ],
        },
    }
    contract_path = PHASE3_DIR / "prediction_contract_v2.json"
    contract_path.write_bytes(canonical_json_bytes(prediction_contract))
    print("wrote", contract_path, sha256_file(contract_path))

    # --- Part 20 environment freeze (bundled here since it is a dependency of the model freeze) ---
    environment = {
        "status": "PHASE3_ENVIRONMENT_FROZEN", "python": sys.version, "numpy": numpy.__version__,
        "scipy": scipy.__version__, "scikit_learn": sklearn.__version__, "xgboost": xgboost.__version__,
        "note": "All Phase-3 selected final models are XGBoost; no neural/torch dependency is part of the v2 evaluation contract.",
    }
    env_path = PHASE3_DIR / "environment_lock_v1.json"
    env_path.write_bytes(canonical_json_bytes(environment))
    print("wrote", env_path, sha256_file(env_path))

    # --- Part 13: pre-generation scientific freeze ---
    oof_audit_path = PHASE3_DIR / "calibration/support_dev_oof_audit_v1.json"
    freeze = {
        "status": "V2_MODEL_FREEZE_ACTIVE",
        "artifact_version": "v2_model_freeze_v1",
        "decision_authority": "USER_DELEGATED_AI_PROJECT_DECISION",
        "parents": {
            "phase1_strategy": _ref("artifacts/performance_v2/phase1/selected_v2_strategy_v1.json"),
            "phase2_validation_selection": _ref("artifacts/performance_v2/phase2/validation_selection_v2.json"),
            "feature_contract": _ref("configs/performance_v2/feature_contract_v2.json"),
            "dev_subject_manifest": _ref("artifacts/performance_v2/phase3/dev_subject_manifest_v1.json"),
            "dev_crossfit_folds": _ref("artifacts/performance_v2/phase3/dev_crossfit_fold_assignment_v1.csv"),
            "final_training_budget": _ref("artifacts/performance_v2/phase3/final_training_budget_v1.json"),
            "preprocessing_b_min": _ref("artifacts/performance_v2/phase3/preprocessing/context_normalization_v2_B_MIN.json"),
            "preprocessing_b_full": _ref("artifacts/performance_v2/phase3/preprocessing/context_normalization_v2_B_FULL.json"),
            "preprocessing_b_plus_f": _ref("artifacts/performance_v2/phase3/preprocessing/context_normalization_v2_B_PLUS_F.json"),
            "recovery24_model": _ref(refit_manifest["tasks"]["recovery24"]["model_path"]),
            "recovery48_model": _ref(refit_manifest["tasks"]["recovery48"]["model_path"]),
            "icu_model": _ref(refit_manifest["tasks"]["icu_stay_time"]["model_path"]),
            "support_model": _ref(refit_manifest["tasks"]["organ_support"]["model_path"]),
            "support_oof_raw": _ref("artifacts/performance_v2/phase3/calibration/support_dev_oof_raw_predictions_v1.jsonl"),
            "support_oof_audit": _ref("artifacts/performance_v2/phase3/calibration/support_dev_oof_audit_v1.json"),
            "support_calibrator": _ref("artifacts/performance_v2/phase3/calibration/isotonic_support_v2.json"),
            "support_threshold": _ref("artifacts/performance_v2/phase3/thresholds/support_threshold_v2.json"),
            "selected_models_v2": _ref("artifacts/performance_v2/phase3/selected_models_v2.json"),
            "prediction_contract": _ref("artifacts/performance_v2/phase3/prediction_contract_v2.json"),
            "environment_lock": _ref("artifacts/performance_v2/phase3/environment_lock_v1.json"),
            "generator_config": {"ref": "configs/synthetic/final_benchmark_v1.json", "sha256": sha256_file(ROOT / "configs/synthetic/final_benchmark_v1.json")},
        },
        "code_commit": code_commit,
        "fresh_test_generated": False,
        "fresh_test_accessed": False,
    }
    GOV_DIR.mkdir(parents=True, exist_ok=True)
    freeze_path = GOV_DIR / "v2_model_freeze_v1.json"
    freeze_path.write_bytes(canonical_json_bytes(freeze))
    print("wrote", freeze_path, sha256_file(freeze_path))
    print("V2_MODEL_FREEZE_ACTIVE")


if __name__ == "__main__":
    main()
