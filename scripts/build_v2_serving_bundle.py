"""Build the V2 dashboard/API serving bundle -- an ENGINEERING artifact
binding the frozen Phase-3/4 scientific artifacts to the application layer.
Creates no new scientific freeze; does not modify selected_models_v2.json.
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.synthetic.config import canonical_json_bytes  # noqa: E402
from vedant_infra.hashing import sha256_file  # noqa: E402

OUT_PATH = ROOT / "artifacts/performance_v2/serving/v2_dashboard_serving_bundle_v1.json"


def _ref(path: str) -> dict:
    return {"ref": path, "sha256": sha256_file(ROOT / path)}


def main() -> None:
    code_commit = subprocess.check_output(("git", "rev-parse", "HEAD"), cwd=ROOT, text=True).strip()
    bundle = {
        "status": "V2_DASHBOARD_SERVING_FROZEN",
        "artifact_version": "v2_dashboard_serving_bundle_v1",
        "kind": "ENGINEERING_APPLICATION_LAYER_ARTIFACT_NOT_A_SCIENTIFIC_FREEZE",
        "parents": {
            "selected_models_v2": _ref("artifacts/performance_v2/phase3/selected_models_v2.json"),
            "v2_model_freeze": _ref("artifacts/performance_v2/governance/v2_model_freeze_v1.json"),
            "recovery24_model": _ref("artifacts/performance_v2/phase3/models/recovery24_xgb_v2.json"),
            "recovery48_model": _ref("artifacts/performance_v2/phase3/models/recovery48_xgb_v2.json"),
            "icu_model": _ref("artifacts/performance_v2/phase3/models/icu_time_xgb_v2.json"),
            "support_model": _ref("artifacts/performance_v2/phase3/models/organ_support_xgb_v2.json"),
            "feature_contract": _ref("configs/performance_v2/feature_contract_v2.json"),
            "preprocessing_b_min": _ref("artifacts/performance_v2/phase3/preprocessing/context_normalization_v2_B_MIN.json"),
            "preprocessing_b_full": _ref("artifacts/performance_v2/phase3/preprocessing/context_normalization_v2_B_FULL.json"),
            "preprocessing_b_plus_f": _ref("artifacts/performance_v2/phase3/preprocessing/context_normalization_v2_B_PLUS_F.json"),
            "isotonic_support_v2": _ref("artifacts/performance_v2/phase3/calibration/isotonic_support_v2.json"),
            "support_threshold_v2": _ref("artifacts/performance_v2/phase3/thresholds/support_threshold_v2.json"),
            "v2_demo_manifest": _ref("configs/performance_v2/v2_demo_manifest_v1.json"),
        },
        "dashboard_api_schema_version": "v2_prediction_response_v1",
        "explainability_configuration": {
            "method": "tree_shap",
            "library": "shap.TreeExplainer",
            "config": {"feature_perturbation": "tree_path_dependent", "model_output": "raw"},
            "applies_to_all_four_tasks": True,
            "note": "Support explanation decomposes the raw XGBoost margin only; isotonic calibration is applied after and is not SHAP-additive.",
        },
        "fresh_test_protection": "src/serving/v2/guard.py::guard_demo_stay -- only configs/performance_v2/v2_demo_manifest_v1.json subjects are servable; unknown/fresh-test stay_ids are rejected identically to unknown stays",
        "code_commit": code_commit,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_bytes(canonical_json_bytes(bundle))
    print("wrote", OUT_PATH, sha256_file(OUT_PATH))


if __name__ == "__main__":
    main()
