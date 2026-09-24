"""Performance-v2 Phase 4, Parts 6-9: the ONE-TIME governed fresh-test
access + coordinated four-model inference pass. Run exactly once.

Transaction order (per the Phase-4 spec's TRANSACTIONAL ACCESS RULE):
  1. complete preflight validation
  2. prepare temporary output location
  3. authorize (SEALED_NOT_ACCESSED -> AUTHORIZED_FOR_ONE_FINAL_RUN)
  4. consume (AUTHORIZED_FOR_ONE_FINAL_RUN -> FINAL_V2_TEST_ACCESS_CONSUMED)
  5. open fresh cohort through the guarded accessor (the only read of fresh
     rows in the entire project)
  6. run all four frozen models in this one pass
  7. write raw immutable prediction files (temp path, then atomic rename)
  8. fsync before rename
  9. (files are promoted individually via atomic os.replace; no partial
     prediction file is ever visible under its final name)
  10. mark FINAL_V2_RUN_COMPLETED

No science decision is made here: models, feature contract, preprocessing,
calibrator, and threshold are all applied exactly as frozen in Phase 3.
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.synthetic.config import canonical_json_bytes  # noqa: E402
from performance_v2.final_evaluation import (  # noqa: E402
    TASK_MODEL_FILE,
    TASK_VARIANT,
    build_inference_matrix,
    isotonic_apply,
    load_frozen_normalization,
    load_xgb_model,
    postprocess_icu_hours,
    reconstruct_group_a_names,
)
from performance_v2.fresh_test_access import (  # noqa: E402
    authorize_one_final_run,
    complete_run,
    consume_access,
    load_fresh_test_rows,
)
from performance_v2.targets import icu_hours_target, icu_rows, recovery_targets, support_rows  # noqa: E402
from vedant_infra.hashing import sha256_file  # noqa: E402

PHASE3_DIR = ROOT / "artifacts/performance_v2/phase3"
PHASE4_DIR = ROOT / "artifacts/performance_v2/phase4"
PRED_DIR = PHASE4_DIR / "predictions"

FEATURE_CONTRACT_REF = "configs/performance_v2/feature_contract_v2.json"
FRESH_FREEZE_REF = "artifacts/performance_v2/governance/v2_fresh_test_freeze_v1.json"
THRESHOLD_VALUE = 0.39781983118092895


def _atomic_write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".tmp-", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as handle:
            for row in rows:
                handle.write(canonical_json_bytes(row))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except Exception:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)
        raise


def main() -> None:
    feature_contract_sha256 = sha256_file(ROOT / FEATURE_CONTRACT_REF)
    fresh_freeze_sha256 = sha256_file(ROOT / FRESH_FREEZE_REF)
    code_commit = subprocess.check_output(("git", "rev-parse", "HEAD"), cwd=ROOT, text=True).strip()

    # --- Steps 1-2: preflight artifacts prepared, everything below is
    # apply-only against DEV-derived, already-frozen state until step 5. ---
    group_a_names = {task: reconstruct_group_a_names(ROOT, task) for task in TASK_VARIANT}
    normalizations = {variant: load_frozen_normalization(ROOT, variant) for variant in set(TASK_VARIANT.values())}
    models = {task: load_xgb_model(ROOT, task) for task in TASK_VARIANT}
    model_hashes = {task: sha256_file(ROOT / TASK_MODEL_FILE[task]) for task in TASK_VARIANT}
    calibrator = json.loads((PHASE3_DIR / "calibration/isotonic_support_v2.json").read_text())["calibration_knots"]

    PRED_DIR.mkdir(parents=True, exist_ok=True)

    # --- Step 3: authorize ---
    authorize_one_final_run(ROOT, authorized_by="performance_v2_phase4_run_final_evaluation", reason="Phase-4 one-time scientific fresh-test evaluation")

    # --- Step 4: consume ---
    consume_access(ROOT)

    # --- Step 5: the one guarded read of fresh-test rows ---
    fresh_rows = load_fresh_test_rows(ROOT)

    # --- Step 6: run all four frozen models in this one pass ---
    written_files = {}

    for task, horizon, target_key, target_field, pred_field in (
        ("recovery24", "24", "delta_sofa_24", "target_delta24", "prediction_delta24"),
        ("recovery48", "48", "delta_sofa_48", "target_delta48", "prediction_delta48"),
    ):
        rows = recovery_targets(fresh_rows, horizon)
        variant = TASK_VARIANT[task]
        matrix = build_inference_matrix(rows, group_a_names=group_a_names[task], normalization=normalizations[variant])
        predictions = models[task].predict(matrix)
        out_rows = []
        for row, prediction in zip(rows, predictions):
            out_rows.append({
                "subject_id": row["subject_id"], "stay_id": row["stay_id"], "prediction_time": row["prediction_time"],
                "grid_index": int(row["grid_index"]), target_field: float(row[target_key]), pred_field: float(prediction),
                "eligible": True, "model_sha256": model_hashes[task], "feature_contract_sha256": feature_contract_sha256,
                "fresh_cohort_freeze_sha256": fresh_freeze_sha256,
            })
        path = PRED_DIR / f"{task}_predictions_v1.jsonl"
        _atomic_write_jsonl(path, out_rows)
        written_files[task] = path

    icu_task_rows = icu_rows(fresh_rows)
    icu_variant = TASK_VARIANT["icu_stay_time"]
    icu_matrix = build_inference_matrix(icu_task_rows, group_a_names=group_a_names["icu_stay_time"], normalization=normalizations[icu_variant])
    icu_raw_log = models["icu_stay_time"].predict(icu_matrix)
    icu_hours_pred = postprocess_icu_hours(np.asarray(icu_raw_log, dtype=np.float64))
    icu_out_rows = []
    for row, raw_log, hours in zip(icu_task_rows, icu_raw_log, icu_hours_pred):
        icu_out_rows.append({
            "subject_id": row["subject_id"], "stay_id": row["stay_id"], "prediction_time": row["prediction_time"],
            "grid_index": int(row["grid_index"]), "true_remaining_hours": float(icu_hours_target(row)),
            "raw_log_prediction": float(raw_log), "prediction_hours": float(hours), "eligible": True,
            "model_sha256": model_hashes["icu_stay_time"], "feature_contract_sha256": feature_contract_sha256,
            "fresh_cohort_freeze_sha256": fresh_freeze_sha256,
        })
    icu_path = PRED_DIR / "icu_stay_time_predictions_v1.jsonl"
    _atomic_write_jsonl(icu_path, icu_out_rows)
    written_files["icu_stay_time"] = icu_path

    support_task_rows = support_rows(fresh_rows)
    support_variant = TASK_VARIANT["organ_support"]
    support_matrix = build_inference_matrix(support_task_rows, group_a_names=group_a_names["organ_support"], normalization=normalizations[support_variant])
    raw_probability = models["organ_support"].predict_proba(support_matrix)[:, 1]
    calibrated_probability = isotonic_apply(raw_probability, calibrator)
    support_out_rows = []
    for row, raw_p, cal_p in zip(support_task_rows, raw_probability, calibrated_probability):
        support_out_rows.append({
            "subject_id": row["subject_id"], "stay_id": row["stay_id"], "prediction_time": row["prediction_time"],
            "grid_index": int(row["grid_index"]), "label": int(row["organ_support_label"]),
            "raw_probability": float(raw_p), "calibrated_probability": float(cal_p),
            "frozen_threshold": THRESHOLD_VALUE, "predicted_alert": bool(cal_p >= THRESHOLD_VALUE), "eligible": True,
            "model_sha256": model_hashes["organ_support"], "feature_contract_sha256": feature_contract_sha256,
            "fresh_cohort_freeze_sha256": fresh_freeze_sha256,
        })
    support_path = PRED_DIR / "organ_support_predictions_v1.jsonl"
    _atomic_write_jsonl(support_path, support_out_rows)
    written_files["organ_support"] = support_path

    # --- Steps 7-9 complete (atomic per-file writes above). Step 10: seal. ---
    manifest = {
        "status": "FINAL_V2_PREDICTIONS_WRITTEN",
        "artifact_version": "performance_v2_phase4_predictions_manifest_v1",
        "code_commit": code_commit,
        "counts": {task: sum(1 for _ in open(path, "rb")) for task, path in written_files.items()},
        "files": {task: {"ref": str(path.relative_to(ROOT)), "sha256": sha256_file(path)} for task, path in written_files.items()},
        "model_hashes": model_hashes,
        "feature_contract_sha256": feature_contract_sha256,
        "fresh_cohort_freeze_sha256": fresh_freeze_sha256,
    }
    manifest_path = PRED_DIR / "predictions_manifest_v1.json"
    manifest_path.write_bytes(canonical_json_bytes(manifest))

    complete_run(ROOT)

    print("FINAL_V2_RUN_COMPLETED")
    for task, path in written_files.items():
        print(f"  {task}: {path} ({manifest['counts'][task]} rows) sha256={manifest['files'][task]['sha256']}")
    print("manifest:", manifest_path, sha256_file(manifest_path))


if __name__ == "__main__":
    main()
