"""Performance-v2 Phase 4, Part 17: freeze the final V2 evaluation. Run
LAST, after predictions/metrics/bootstrap/calibration/error-analysis/
generalization/second-run-guard all validate.
"""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.synthetic.config import canonical_json_bytes  # noqa: E402
from vedant_infra.hashing import sha256_file  # noqa: E402

PHASE4_DIR = ROOT / "artifacts/performance_v2/phase4"
GOV_DIR = ROOT / "artifacts/performance_v2/governance"

PRE_ACCESS_COMMIT = "f1e9e31726a9b0da5abe70128f1af43d06c769e3"


def _ref(path: str) -> dict:
    return {"ref": path, "sha256": sha256_file(ROOT / path)}


def main() -> None:
    code_commit = subprocess.check_output(("git", "rev-parse", "HEAD"), cwd=ROOT, text=True).strip()
    access_state = json.loads((GOV_DIR / "v2_fresh_test_access_state.json").read_text())
    if access_state["status"] != "FINAL_V2_RUN_COMPLETED":
        raise SystemExit("refusing to freeze: access state is not FINAL_V2_RUN_COMPLETED")

    freeze = {
        "status": "V2_FINAL_EVALUATION_FROZEN",
        "artifact_version": "v2_final_evaluation_freeze_v1",
        "decision_authority": "USER_DELEGATED_AI_PROJECT_DECISION",
        "parents": {
            "v2_model_freeze": _ref("artifacts/performance_v2/governance/v2_model_freeze_v1.json"),
            "v2_fresh_test_freeze": _ref("artifacts/performance_v2/governance/v2_fresh_test_freeze_v1.json"),
            "final_evaluation_plan": _ref("artifacts/performance_v2/phase4/final_evaluation_plan_v1.json"),
            "naive_baselines_v2": _ref("artifacts/performance_v2/phase4/naive_baselines_v2.json"),
            "pre_access_commit": PRE_ACCESS_COMMIT,
            "access_state_history": _ref("artifacts/performance_v2/governance/v2_fresh_test_access_state.json"),
            "selected_models_v2": _ref("artifacts/performance_v2/phase3/selected_models_v2.json"),
            "recovery24_model": _ref("artifacts/performance_v2/phase3/models/recovery24_xgb_v2.json"),
            "recovery48_model": _ref("artifacts/performance_v2/phase3/models/recovery48_xgb_v2.json"),
            "icu_model": _ref("artifacts/performance_v2/phase3/models/icu_time_xgb_v2.json"),
            "support_model": _ref("artifacts/performance_v2/phase3/models/organ_support_xgb_v2.json"),
            "isotonic_support_v2": _ref("artifacts/performance_v2/phase3/calibration/isotonic_support_v2.json"),
            "support_threshold_v2": _ref("artifacts/performance_v2/phase3/thresholds/support_threshold_v2.json"),
            "predictions_manifest": _ref("artifacts/performance_v2/phase4/predictions/predictions_manifest_v1.json"),
            "recovery24_predictions": _ref("artifacts/performance_v2/phase4/predictions/recovery24_predictions_v1.jsonl"),
            "recovery48_predictions": _ref("artifacts/performance_v2/phase4/predictions/recovery48_predictions_v1.jsonl"),
            "icu_predictions": _ref("artifacts/performance_v2/phase4/predictions/icu_stay_time_predictions_v1.jsonl"),
            "support_predictions": _ref("artifacts/performance_v2/phase4/predictions/organ_support_predictions_v1.jsonl"),
            "final_metrics": _ref("artifacts/performance_v2/phase4/metrics/final_metrics_v2.json"),
            "naive_comparison": _ref("artifacts/performance_v2/phase4/metrics/naive_comparison_v2.json"),
            "calibration_evidence": _ref("artifacts/performance_v2/phase4/metrics/calibration_evidence_v2.json"),
            "error_analysis": _ref("artifacts/performance_v2/phase4/metrics/error_analysis_v2.json"),
            "generalization_comparison": _ref("artifacts/performance_v2/phase4/metrics/generalization_comparison_v2.json"),
            "final_bootstrap": _ref("artifacts/performance_v2/phase4/bootstrap/final_bootstrap_v2.json"),
            "second_run_guard": _ref("artifacts/performance_v2/governance/v2_second_run_guard_v1.json"),
        },
        "code_commit": code_commit,
        "fresh_test_accessed": True,
        "successful_scientific_evaluations": 1,
        "post_test_tuning": False,
    }
    freeze_path = GOV_DIR / "v2_final_evaluation_freeze_v1.json"
    freeze_path.write_bytes(canonical_json_bytes(freeze))
    print("wrote", freeze_path, sha256_file(freeze_path))
    print("V2_FINAL_EVALUATION_FROZEN")


if __name__ == "__main__":
    main()
