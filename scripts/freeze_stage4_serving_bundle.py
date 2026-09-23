"""Freeze the Stage-4 final serving manifest and acceptance artifact.

Reuses the exact Stage-4 resolver (src/serving/real/bundle.py) that the real
FastAPI/dashboard composition roots already use — this script performs zero
new hash logic. It only records what the resolver already independently
verified, plus focused-suite/audit evidence, into two frozen JSON artifacts:

    artifacts/serving/final_serving_bundle_v1.json
    artifacts/acceptance/stage4_serving_integration_freeze_v1.json

Does not mutate selected_models_v1.json, G3, or any other G3-bound artifact.
Performs no model training, no calibration, and no final-test access.
"""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from data.synthetic.config import canonical_json_bytes  # noqa: E402
from serving.real.bundle import resolve_stage4_bundle  # noqa: E402
from serving.real.explanations import real_ig_config, real_tree_shap_config  # noqa: E402
from vedant_infra.hashing import sha256_file  # noqa: E402


ENV_LOCK_PATH = ROOT / "artifacts/serving/stage4_environment_lock_v1.txt"
BUNDLE_PATH = ROOT / "artifacts/serving/final_serving_bundle_v1.json"
ACCEPTANCE_PATH = ROOT / "artifacts/acceptance/stage4_serving_integration_freeze_v1.json"


def _write(path: Path, payload: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(payload))
    return sha256_file(path)


def main() -> None:
    resolution = resolve_stage4_bundle(ROOT)
    bundle = resolution.bundle
    manifest = resolution.manifest
    tasks = manifest["tasks"]
    bindings = manifest["governance_bindings"]
    env_lock_sha256 = sha256_file(ENV_LOCK_PATH)
    tree_config = real_tree_shap_config()
    ig_config = real_ig_config()

    serving_bundle = {
        "artifact_version": "final_serving_bundle_v1",
        "status": "SERVING_BUNDLE_FROZEN_PRE_TEST",
        "decision_authority": "USER_DELEGATED_AI_PROJECT_DECISION",
        "selected_models": {
            "ref": resolution.manifest_path.relative_to(ROOT).as_posix(),
            "sha256": resolution.manifest_sha256,
        },
        "g3": {"ref": "artifacts/governance/g3_freeze.json", "sha256": resolution.g3_sha256},
        "g1": bindings["g1"],
        "phase14_handoff": bindings["phase14_handoff"],
        "feature_schema": bindings["feature_schema"],
        "preprocessor": bindings["preprocessor"],
        "recovery_target_scaler": bindings["recovery_target_scaler"],
        "support_class_weight": bindings["support_class_weight"],
        "split": bindings["split"],
        "tasks": {
            "recovery": {
                "family": "xgboost", "candidate_id": tasks["recovery"]["candidate_id"],
                "model_artifacts": tasks["recovery"]["model_artifacts"],
                "explanation_method": "tree_shap",
            },
            "icu_stay_time": {
                "family": "gru", "candidate_id": tasks["icu_stay_time"]["candidate_id"],
                "checkpoint_ref": tasks["icu_stay_time"]["artifact_ref"],
                "checkpoint_sha256": tasks["icu_stay_time"]["artifact_sha256"],
                "explanation_method": "integrated_gradients",
            },
            "organ_support": {
                "family": "xgboost", "candidate_id": tasks["organ_support"]["candidate_id"],
                "model_artifact_ref": tasks["organ_support"]["artifact_ref"],
                "model_artifact_sha256": tasks["organ_support"]["artifact_sha256"],
                "explanation_method": "tree_shap",
            },
        },
        "support_calibrator": {"ref": tasks["organ_support"]["calibrator"]["artifact_ref"], "sha256": bundle.support_calibrator_sha256},
        "support_threshold": {
            "ref": tasks["organ_support"]["threshold"]["artifact_ref"], "sha256": bundle.support_threshold_sha256,
            "value": bundle.support_threshold, "comparator": "greater_than_or_equal",
        },
        "explanation_config": {
            "tree_shap": {**tree_config.__dict__, "sha256": tree_config.sha256},
            "integrated_gradients": {**ig_config.__dict__, "sha256": ig_config.sha256},
            "ig_baseline_policy": "REAL_NEUTRAL_VALUE_STATIC_ZERO_TSLO_FROZEN_SENTINEL",
            "ig_output_domain": "remaining_icu_hours_postprocessed_v1",
        },
        "prediction_schema_version": "prediction_schema_v1",
        "flat_feature_map_sha256": resolution.flat_feature_map.sha256,
        "environment_lock": {"ref": "artifacts/serving/stage4_environment_lock_v1.txt", "sha256": env_lock_sha256},
        "serving_code_commit": resolution.code_commit,
        "test_accessed": False,
    }
    bundle_sha256 = _write(BUNDLE_PATH, serving_bundle)

    junit_report = json.loads((ROOT / "docs/evidence/system/integration/full_test_report_v1.json").read_text())

    acceptance = {
        "artifact_version": "stage4_serving_integration_freeze_v1",
        "status": "STAGE4_SERVING_INTEGRATION_FROZEN",
        "decision_authority": "USER_DELEGATED_AI_PROJECT_DECISION",
        "final_serving_bundle": {"ref": "artifacts/serving/final_serving_bundle_v1.json", "sha256": bundle_sha256},
        "selected_models": {"ref": resolution.manifest_path.relative_to(ROOT).as_posix(), "sha256": resolution.manifest_sha256},
        "g3": {"ref": "artifacts/governance/g3_freeze.json", "sha256": resolution.g3_sha256},
        "environment_lock": {"ref": "artifacts/serving/stage4_environment_lock_v1.txt", "sha256": env_lock_sha256},
        "api_pipeline_contract": "api.real_app.build_real_app -> api.main.create_app(pipeline=PredictionPipeline)",
        "explanation_configs": {
            "tree_shap_sha256": tree_config.sha256,
            "integrated_gradients_sha256": ig_config.sha256,
        },
        "focused_integration_evidence": {
            "test_file": "tests/test_stage4_real_serving.py",
            "dashboard_test_file": "tests/test_stage4_dashboard.py",
        },
        "full_regression": {
            "passed": junit_report["counts"]["passed"], "failed": junit_report["counts"]["failures"],
            "skipped": junit_report["counts"]["skipped"], "status": junit_report["status"],
        },
        "serving_code_commit": resolution.code_commit,
        "test_accessed": False,
        "final_test_state": "AUTHORIZED_NOT_RUN",
    }
    _write(ACCEPTANCE_PATH, acceptance)
    print("serving_bundle_sha256=" + bundle_sha256)
    print("acceptance_path=" + str(ACCEPTANCE_PATH))


if __name__ == "__main__":
    main()
