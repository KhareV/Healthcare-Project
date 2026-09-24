"""Performance-v2 Phase 3, Part 16: seal the fresh test immediately after
generation/canonical packaging. Writes the access-state marker
(SEALED_NOT_ACCESSED) and the v2_fresh_test_freeze_v1.json governance
freeze, binding every upstream artifact hash. No outcome data is read here.
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
FRESH_DIR = PHASE3_DIR / "fresh_test_cohort"

PRE_GENERATION_FREEZE_COMMIT = "393414f987161a8f2f4cb9728329b2b21021badd"


def _ref(path: str) -> dict:
    return {"ref": path, "sha256": sha256_file(ROOT / path)}


def main() -> None:
    code_commit = subprocess.check_output(("git", "rev-parse", "HEAD"), cwd=ROOT, text=True).strip()

    access_state = {
        "status": "SEALED_NOT_ACCESSED",
        "artifact_version": "v2_fresh_test_access_state_v1",
        "allowed_transitions": [
            "SEALED_NOT_ACCESSED -> AUTHORIZED_FOR_ONE_FINAL_RUN",
            "AUTHORIZED_FOR_ONE_FINAL_RUN -> FINAL_V2_TEST_ACCESS_CONSUMED",
            "FINAL_V2_TEST_ACCESS_CONSUMED -> FINAL_V2_RUN_COMPLETED",
        ],
        "note": "Phase 3 stops here. Only the guarded Phase-4 command may advance this state.",
    }
    GOV_DIR.mkdir(parents=True, exist_ok=True)
    access_state_path = GOV_DIR / "v2_fresh_test_access_state.json"
    access_state_path.write_bytes(canonical_json_bytes(access_state))
    print("wrote", access_state_path, sha256_file(access_state_path))

    freeze = {
        "status": "V2_FRESH_TEST_FROZEN",
        "artifact_version": "v2_fresh_test_freeze_v1",
        "decision_authority": "USER_DELEGATED_AI_PROJECT_DECISION",
        "parents": {
            "cohort_config": _ref("configs/performance_v2/fresh_test_cohort_v1.json"),
            "cohort_spec": _ref("artifacts/performance_v2/phase3/fresh_test_cohort_spec_v1.json"),
            "generation_seed": 402995653,
            "generator_scientific_parent_v1_config": _ref("configs/synthetic/final_benchmark_v1.json"),
            "generator_equivalence_audit": _ref("artifacts/performance_v2/governance/generator_equivalence_audit_v1.json"),
            "raw_generator_manifest": _ref("artifacts/performance_v2/phase3/fresh_test_cohort/raw/synthetic_dataset_manifest_v1.json"),
            "cohort_manifest": _ref("artifacts/performance_v2/phase3/fresh_test_cohort/cohort/synthetic_cohort_manifest_v2.json"),
            "timeline_manifest": _ref("artifacts/performance_v2/phase3/fresh_test_cohort/timeline/synthetic_processed_manifest_v1.json"),
            "feature_manifest": _ref("artifacts/performance_v2/phase3/fresh_test_cohort/features/synthetic_feature_input_manifest_v1.json"),
            "pre_split_scientific_package_manifest": _ref("artifacts/performance_v2/phase3/fresh_test_cohort/pre_split/phase9_pre_split_manifest_v1.json"),
            "structural_audit": _ref("artifacts/performance_v2/phase3/fresh_test_cohort_structural_audit_v1.json"),
            "split_csv": _ref("artifacts/performance_v2/phase3/fresh_test_split_v1.csv"),
            "split_manifest": _ref("artifacts/performance_v2/phase3/fresh_test_split_manifest_v1.json"),
            "selected_models_v2": _ref("artifacts/performance_v2/phase3/selected_models_v2.json"),
            "v2_model_freeze": _ref("artifacts/performance_v2/governance/v2_model_freeze_v1.json"),
            "pre_generation_freeze_commit": PRE_GENERATION_FREEZE_COMMIT,
        },
        "code_commit": code_commit,
        "fresh_test_access_state": "SEALED_NOT_ACCESSED",
    }
    freeze_path = GOV_DIR / "v2_fresh_test_freeze_v1.json"
    freeze_path.write_bytes(canonical_json_bytes(freeze))
    print("wrote", freeze_path, sha256_file(freeze_path))
    print("V2_FRESH_TEST_FROZEN")


if __name__ == "__main__":
    main()
