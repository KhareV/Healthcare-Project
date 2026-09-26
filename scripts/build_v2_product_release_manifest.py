"""Build the two Part-13 product-release artifacts for the FINAL PRODUCT
RELEASE work: the product serving bundle and the final product release
manifest. Purely a release/evidence binding over already-frozen scientific
artifacts and already-completed product work -- creates no new scientific
freeze, runs no inference, and does not alter any V2 scientific artifact.
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.synthetic.config import canonical_json_bytes  # noqa: E402
from vedant_infra.hashing import sha256_file  # noqa: E402

BUNDLE_PATH = ROOT / "artifacts/performance_v2/product/v2_product_serving_bundle_v1.json"
RELEASE_PATH = ROOT / "artifacts/performance_v2/product/final_product_release_v2.json"


def _ref(path: str) -> dict:
    return {"ref": path, "sha256": sha256_file(ROOT / path)}


def _code_commit() -> str:
    return subprocess.check_output(("git", "rev-parse", "HEAD"), cwd=ROOT, text=True).strip()


def build_serving_bundle(code_commit: str) -> dict:
    return {
        "artifact_version": "v2_product_serving_bundle_v1",
        "status": "V2_PRODUCT_SERVING_FROZEN",
        "kind": "ENGINEERING_APPLICATION_LAYER_ARTIFACT_NOT_A_SCIENTIFIC_FREEZE",
        "code_commit": code_commit,
        "product_api_version": "v2_product_api_v1",
        "dashboard_api_schema_version": "v2_prediction_response_v1",
        "frontend_version": "0.1.0",
        "custom_record_schema_version": "custom_record_v1",
        "auth_mode_versions": ["clerk_verified", "local_dev_no_auth"],
        "parents": {
            "selected_models_v2": _ref("artifacts/performance_v2/phase3/selected_models_v2.json"),
            "recovery24_model": _ref("artifacts/performance_v2/phase3/models/recovery24_xgb_v2.json"),
            "recovery48_model": _ref("artifacts/performance_v2/phase3/models/recovery48_xgb_v2.json"),
            "icu_model": _ref("artifacts/performance_v2/phase3/models/icu_time_xgb_v2.json"),
            "support_model": _ref("artifacts/performance_v2/phase3/models/organ_support_xgb_v2.json"),
            "feature_contract": _ref("configs/performance_v2/feature_contract_v2.json"),
            "feature_schema_v2": _ref("configs/synthetic/feature_schema_v2.json"),
            "event_dict_v2": _ref("configs/event_dict_v2.yaml"),
            "preprocessing_b_full": _ref("artifacts/performance_v2/phase3/preprocessing/context_normalization_v2_B_FULL.json"),
            "preprocessing_b_min": _ref("artifacts/performance_v2/phase3/preprocessing/context_normalization_v2_B_MIN.json"),
            "preprocessing_b_plus_f": _ref("artifacts/performance_v2/phase3/preprocessing/context_normalization_v2_B_PLUS_F.json"),
            "isotonic_support_v2": _ref("artifacts/performance_v2/phase3/calibration/isotonic_support_v2.json"),
            "support_threshold_v2": _ref("artifacts/performance_v2/phase3/thresholds/support_threshold_v2.json"),
            "v2_model_freeze": _ref("artifacts/performance_v2/governance/v2_model_freeze_v1.json"),
            "v2_demo_manifest": _ref("configs/performance_v2/v2_demo_manifest_v1.json"),
            "demo_patients_v1": _ref("artifacts/performance_v2/product/demo_patients_v1.json"),
            "trajectory_copilot_prompt_v1": _ref("configs/product_v2/trajectory_copilot_prompt_v1.txt"),
        },
        "fresh_test_protection": (
            "src/serving/v2/guard.py::guard_demo_stay -- only demo-manifest and "
            "CUSTOM-namespaced ephemeral stays are servable; every fresh-V2-test "
            "subject is rejected identically to an unknown stay"
        ),
        "custom_record_authorization": (
            "src/serving/v2/auth.py::ClerkAuthenticator + api/v2_app.py::_authorize_custom_stay "
            "-- owner_user_id is derived from a verified Clerk token, never client-supplied; "
            "cross-user access to a custom record returns 404, identical to a nonexistent stay"
        ),
    }


def build_final_release(code_commit: str, bundle_sha256: str) -> dict:
    evidence_dir = ROOT / "docs/evidence/product_v2"
    return {
        "artifact_version": "final_product_release_v2",
        "status": "FINAL_PRODUCT_RELEASE_READY",
        "kind": "ENGINEERING_APPLICATION_LAYER_ARTIFACT_NOT_A_SCIENTIFIC_FREEZE",
        "code_commit": code_commit,
        "parents": {
            "v2_final_evaluation_freeze": _ref("artifacts/performance_v2/governance/v2_final_evaluation_freeze_v1.json"),
            "v2_product_serving_bundle": {"ref": str(BUNDLE_PATH.relative_to(ROOT)), "sha256": bundle_sha256},
            "demo_patients_v1": _ref("artifacts/performance_v2/product/demo_patients_v1.json"),
            "final_v2_report": _ref("docs/performance_v2/FINAL_V2_TEST_EVALUATION.md"),
            "readme": _ref("README.md"),
            "runbook": _ref("RUNBOOK.md"),
            "product_architecture_doc": _ref("docs/product_v2/PRODUCT_ARCHITECTURE.md"),
            "custom_record_flow_doc": _ref("docs/product_v2/CUSTOM_RECORD_FLOW.md"),
            "auth_and_data_boundaries_doc": _ref("docs/product_v2/AUTH_AND_DATA_BOUNDARIES.md"),
            "trajectory_copilot_doc": _ref("docs/product_v2/TRAJECTORY_COPILOT.md"),
            "final_demo_script_doc": _ref("docs/product_v2/FINAL_DEMO_SCRIPT.md"),
            "product_viva_notes_doc": _ref("docs/product_v2/PRODUCT_VIVA_NOTES.md"),
        },
        "custom_record_schema_version": "custom_record_v1",
        "auth_mode": "clerk_verified_with_local_dev_fallback",
        "screenshot_evidence_count": len(list(evidence_dir.glob("*.png"))),
        "screenshot_evidence_dir": "docs/evidence/product_v2/",
        "scientific_artifacts_unchanged_confirmation": {
            "selected_models_v2_sha256": "6fed05f3c71ce648658b2f864e69fa9f80f3ec8538f53a4e370fa8a4eaf176a8",
            "isotonic_support_v2_sha256": "1ef311fd8a81a3e392db8a4cb7668eb4f404b0d21d818a94fc5ce308fdfe0b4f",
            "final_metrics_v2_sha256": "dfcfb6a4bcba461c40a2ea5e182c40f975ab051b8452bd94b64dec3df0cedfbc",
            "final_bootstrap_v2_sha256": "578638b6fce8578f86ca86240af730fb656ee1f12cb3f0d522d93e8917d92c79",
            "v2_final_evaluation_freeze_sha256": "a433c1ee203f6cf9f01b131a2a9a10c73da57097564705ce64d7351c0057b649",
        },
        "fresh_test_not_reinferred": True,
    }


def main() -> None:
    code_commit = _code_commit()

    bundle = build_serving_bundle(code_commit)
    BUNDLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    BUNDLE_PATH.write_bytes(canonical_json_bytes(bundle))
    bundle_sha256 = sha256_file(BUNDLE_PATH)
    print("wrote", BUNDLE_PATH, bundle_sha256)

    release = build_final_release(code_commit, bundle_sha256)
    RELEASE_PATH.write_bytes(canonical_json_bytes(release))
    print("wrote", RELEASE_PATH, sha256_file(RELEASE_PATH))


if __name__ == "__main__":
    main()
