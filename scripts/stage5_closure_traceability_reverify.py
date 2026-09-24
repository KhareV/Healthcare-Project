"""Stage-5 FINAL CLOSURE: re-verify validate_final_result_traceability()
against the real, currently-registered final-test artifacts -- read-only,
no test-partition access, no model inference, no guarded_test_access call.

This reconstructs the same "row" shape stage5_evaluator() built at the time
of the successful run (which necessarily passed this exact check then,
since the run completed and registered its results) and re-runs the check
now, against the current experiments/artifacts.csv, to prove closure work
did not silently break final-result traceability.
"""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluation.final_test import validate_final_result_traceability  # noqa: E402
from experiments.lineage import read_artifact_index  # noqa: E402
from vedant_infra.hashing import sha256_file  # noqa: E402

MODEL_ARTIFACT_IDS = {
    "recovery24": "phase12-xgb-recovery-014-attempt-1:model:recovery24",
    "recovery48": "phase12-xgb-recovery-014-attempt-1:model:recovery48",
    "icu": "final-v2-gru-icu-time-026-attempt-1:model",
    "support": "phase12-xgb-support-024-attempt-1:model:organ_support",
}
PREDICTION_IDS = {
    "recovery24": "stage5-final-test-prediction-recovery24-v1",
    "recovery48": "stage5-final-test-prediction-recovery48-v1",
    "icu": "stage5-final-test-prediction-icu-v1",
    "support": "stage5-final-test-prediction-support-v1",
}
TASK_BY_NAME = {"recovery24": "recovery", "recovery48": "recovery", "icu": "icu_stay_time", "support": "organ_support"}
LABEL_VERSION_BY_NAME = {
    "recovery24": "synthetic_xgb_target_contract_v1", "recovery48": "synthetic_xgb_target_contract_v1",
    "icu": "synthetic_phase9_final_target_contract_v1", "support": "synthetic_xgb_target_contract_v1",
}
MANIFEST_ARTIFACT_ID = "stage3-selected-models-manifest"
PREPROCESSOR_ARTIFACT_ID = "stage5-registered-synthetic-feature-preprocessor-v1"


def main() -> None:
    artifacts = {item.artifact_id: item for item in read_artifact_index(ROOT / "experiments/artifacts.csv")}
    manifest = artifacts[MANIFEST_ARTIFACT_ID]
    preprocessor = artifacts[PREPROCESSOR_ARTIFACT_ID]
    g3_path = ROOT / "artifacts/governance/g3_freeze.json"
    code_commit = subprocess.check_output(("git", "rev-parse", "HEAD"), cwd=ROOT, text=True).strip()

    rows = []
    for name, prediction_id in PREDICTION_IDS.items():
        prediction = artifacts[prediction_id]
        model_id = MODEL_ARTIFACT_IDS[name]
        model = artifacts[model_id]
        rows.append({
            "task": TASK_BY_NAME[name],
            "model_run_id": model.producing_run_id,
            "feature_version": prediction.feature_version,
            "label_version": LABEL_VERSION_BY_NAME[name],
            "code_commit": code_commit,
            "prediction_artifact_id": prediction_id,
            "prediction_sha256": prediction.artifact_sha256,
            "model_artifact_id": model_id,
            "model_sha256": model.artifact_sha256,
            "manifest_artifact_id": MANIFEST_ARTIFACT_ID,
            "manifest_sha256": manifest.artifact_sha256,
            "preprocessor_artifact_id": PREPROCESSOR_ARTIFACT_ID,
            "preprocessor_sha256": preprocessor.artifact_sha256,
            "g3_marker_ref": "artifacts/governance/g3_freeze.json",
            "g3_marker_sha256": sha256_file(g3_path),
            "split_hash": prediction.split_hash,
        })

    validate_final_result_traceability(ROOT, rows)
    print("validate_final_result_traceability: PASS for all 4 prediction artifacts")
    for row in rows:
        print(" -", row["prediction_artifact_id"], "task=" + row["task"])


if __name__ == "__main__":
    main()
