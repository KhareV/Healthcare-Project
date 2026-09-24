"""Freeze artifacts/governance/stage5_final_test_plan_v1.json — Part A of
Stage 5. Must run and be committed BEFORE any final-test access. Binds every
remaining evaluation choice (bootstrap config, model exposure plan, naive
baseline, error-analysis spec, sensitivity spec, retry policy) so nothing is
decided after the sealed test partition is opened.
"""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from data.synthetic.config import canonical_json_bytes  # noqa: E402
from evaluation.naive_baseline import load_or_fit_naive_baselines  # noqa: E402
from vedant_infra.g3 import validate_g3_marker  # noqa: E402
from vedant_infra.hashing import sha256_file  # noqa: E402


PLAN_PATH = ROOT / "artifacts/governance/stage5_final_test_plan_v1.json"


def _ref(path: str) -> dict:
    return {"ref": path, "sha256": sha256_file(ROOT / path)}


def main() -> None:
    marker = validate_g3_marker(ROOT / "artifacts/governance/g3_freeze.json", ROOT, expected_scope="real")
    naive, naive_path, naive_sha256 = load_or_fit_naive_baselines(
        ROOT,
        artifact_ref="artifacts/final_test/naive_baseline/naive_baseline_v1.json",
        train_ref="artifacts/data/synthetic/phase10/final/synthetic_phase10_v1/train.jsonl",
    )
    final_test_config = json.loads((ROOT / "configs/final_test_v1.json").read_text(encoding="utf-8"))
    bootstrap_config = json.loads((ROOT / "configs/evaluation/bootstrap_v1.json").read_text(encoding="utf-8"))
    error_analysis_config = json.loads((ROOT / "configs/error_analysis/error_analysis_v1.json").read_text(encoding="utf-8"))
    code_commit = subprocess.check_output(("git", "rev-parse", "HEAD"), cwd=ROOT, text=True).strip()

    plan = {
        "manifest_version": "stage5_final_test_plan_v1",
        "status": "FINAL_TEST_PLAN_FROZEN_PRE_ACCESS",
        "decision_authority": "USER_DELEGATED_AI_PROJECT_DECISION",
        "authorization": "STAGE5_FINAL_TEST_PLAN_FREEZE_REQUESTED",
        "human_member_signoff_claimed": False,
        "parents": {
            "stage4_acceptance": _ref("artifacts/acceptance/stage4_serving_integration_freeze_v1.json"),
            "stage4_serving_bundle": _ref("artifacts/serving/final_serving_bundle_v1.json"),
            "selected_models_v1": {"ref": "artifacts/models/selected_models_v1.json", "sha256": marker["selected_models_sha256"]},
            "g3_freeze": {"ref": "artifacts/governance/g3_freeze.json", "sha256": sha256_file(ROOT / "artifacts/governance/g3_freeze.json")},
        },
        "bootstrap": {
            "n_bootstrap": bootstrap_config["n_bootstrap"],
            "seed": bootstrap_config["seed"],
            "ci_level": bootstrap_config["ci_level"],
            "ci_quantiles": bootstrap_config["ci_quantiles"],
            "percentile_convention": bootstrap_config["percentile_convention"],
            "resampling_unit": "stay_id",
            "config_ref": "configs/evaluation/bootstrap_v1.json",
            "config_sha256": sha256_file(ROOT / "configs/evaluation/bootstrap_v1.json"),
        },
        "model_exposure_plan": {
            "selected_models_only": True,
            "recovery": {"family": "xgboost", "candidate": "xgb-recovery-014"},
            "icu_stay_time": {"family": "gru", "candidate": "gru-icu-time-026"},
            "organ_support": {"family": "xgboost", "candidate": "xgb-support-024"},
            "evaluate_naive": True,
            "evaluate_nonselected_families": False,
            "evaluate_lstm": False,
            "lstm_status": "VALIDATION_SENSITIVITY_ONLY_NEVER_TEST_ELIGIBLE",
        },
        "naive_baseline_artifact": {
            "ref": "artifacts/final_test/naive_baseline/naive_baseline_v1.json",
            "sha256": naive_sha256,
            "recovery24_constant_delta": naive.recovery24_constant_delta,
            "recovery48_constant_delta": naive.recovery48_constant_delta,
            "icu_time_log1p_median": naive.icu_time_log1p_median,
            "organ_support_prevalence": naive.organ_support_prevalence,
            "source_partition": "train",
        },
        "error_analysis_specification": {
            "ref": "configs/error_analysis/error_analysis_v1.json",
            "sha256": sha256_file(ROOT / "configs/error_analysis/error_analysis_v1.json"),
            "spec_version": error_analysis_config["spec_version"],
            "active_slices": [item["slice_name"] for item in error_analysis_config["slices"] if item["status"] == "FROZEN"],
            "not_reported_slices": [
                item["slice_name"] for item in error_analysis_config["slices"]
                if item["status"] == "NOT_REPORTED_NO_AUTHORITATIVE_MAPPING"
            ],
        },
        "sensitivity_specification": {
            "analysis": "complete_component_sofa",
            "implementation": "evaluation.sensitivity.evaluate_complete_component_sensitivity",
            "membership_logic": "evaluation.sensitivity.complete_component_included (frozen, unmodified)",
            "recomputes_stay_weights_within_subset": True,
            "is_primary_analysis": False,
        },
        "retry_policy": final_test_config["retry_policy"],
        "code_commit": code_commit,
        "test_accessed": False,
    }
    PLAN_PATH.parent.mkdir(parents=True, exist_ok=True)
    PLAN_PATH.write_bytes(canonical_json_bytes(plan))
    print("plan_sha256=" + sha256_file(PLAN_PATH))
    print("naive_baseline_sha256=" + naive_sha256)


if __name__ == "__main__":
    main()
