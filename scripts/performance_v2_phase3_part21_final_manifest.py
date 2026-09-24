"""Performance-v2 Phase 3, Part 21: the final Phase-3 acceptance manifest,
binding every governance artifact produced in this phase. Run LAST.
"""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.synthetic.config import canonical_json_bytes  # noqa: E402
from vedant_infra.hashing import sha256_file  # noqa: E402

PHASE3_DIR = ROOT / "artifacts/performance_v2/phase3"
GOV_DIR = ROOT / "artifacts/performance_v2/governance"

PRE_GENERATION_FREEZE_COMMIT = "393414f987161a8f2f4cb9728329b2b21021badd"


def _ref(path: str) -> dict:
    return {"ref": path, "sha256": sha256_file(ROOT / path)}


def main() -> None:
    code_commit = subprocess.check_output(("git", "rev-parse", "HEAD"), cwd=ROOT, text=True).strip()

    manifest = {
        "status": [
            "PERFORMANCE_V2_PHASE3_COMPLETE",
            "V2_PRETEST_FROZEN",
            "FRESH_V2_TEST_SEALED",
        ],
        "artifact_version": "performance_v2_phase3_pretest_freeze_manifest_v1",
        "decision_authority": "USER_DELEGATED_AI_PROJECT_DECISION",
        "parents": {
            "v2_model_freeze": _ref("artifacts/performance_v2/governance/v2_model_freeze_v1.json"),
            "selected_models_v2": _ref("artifacts/performance_v2/phase3/selected_models_v2.json"),
            "isotonic_support_v2": _ref("artifacts/performance_v2/phase3/calibration/isotonic_support_v2.json"),
            "support_threshold_v2": _ref("artifacts/performance_v2/phase3/thresholds/support_threshold_v2.json"),
            "prediction_contract_v2": _ref("artifacts/performance_v2/phase3/prediction_contract_v2.json"),
            "fresh_test_cohort_spec": _ref("artifacts/performance_v2/phase3/fresh_test_cohort_spec_v1.json"),
            "generator_equivalence_audit": _ref("artifacts/performance_v2/governance/generator_equivalence_audit_v1.json"),
            "fresh_test_structural_audit": _ref("artifacts/performance_v2/phase3/fresh_test_cohort_structural_audit_v1.json"),
            "fresh_test_split_manifest": _ref("artifacts/performance_v2/phase3/fresh_test_split_manifest_v1.json"),
            "v2_fresh_test_freeze": _ref("artifacts/performance_v2/governance/v2_fresh_test_freeze_v1.json"),
            "v2_fresh_test_access_state": _ref("artifacts/performance_v2/governance/v2_fresh_test_access_state.json"),
            "pre_generation_freeze_commit": PRE_GENERATION_FREEZE_COMMIT,
        },
        "code_commit": code_commit,
        "fresh_test_accessed": False,
        "final_v2_predictions_exist": False,
        "final_v2_metrics_exist": False,
        "fresh_test_access_state": "SEALED_NOT_ACCESSED",
        "phase4_readiness": "PHASE4_READY",
        "stop_after_phase3": True,
    }
    manifest_path = PHASE3_DIR / "phase3_pretest_freeze_manifest_v1.json"
    manifest_path.write_bytes(canonical_json_bytes(manifest))
    print("wrote", manifest_path, sha256_file(manifest_path))
    print("PERFORMANCE_V2_PHASE3_COMPLETE / V2_PRETEST_FROZEN / FRESH_V2_TEST_SEALED")


if __name__ == "__main__":
    main()
