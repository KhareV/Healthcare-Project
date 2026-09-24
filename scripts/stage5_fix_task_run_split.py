"""One-off correction, applied before the Part-B commit (second pass).

audit_registry_lineage() found "task mismatch": the aggregate
"stage5-final-test-evaluation-v1" run row declares task=
"recovery+icu_stay_time+organ_support", but the 4 prediction artifacts it
produced each declare a single real task (recovery/recovery/icu_stay_time/
organ_support), and validate_artifact_lineage requires an artifact's task to
equal its producing run's task whenever both are non-empty.
src/experiments/stage5_final_test.py is already fixed (STAGE5_TASK_RUN_IDS,
ensure_stage5_run_registered, register_stage5_results) to register one
per-task companion run and point each prediction's producing_run_id at its
own task's run instead of the aggregate one. This applies the identical
correction to the registry/artifacts rows the real Part-B run already wrote
-- pure registry-bookkeeping, no model/split/calibration/threshold/metric
content changes, and no test-partition access.
"""

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.synthetic.config import canonical_json_bytes  # noqa: E402
from vedant_infra.hashing import sha256_file  # noqa: E402
from vedant_infra.registry import validate_registry  # noqa: E402

AGGREGATE_RUN_ID = "stage5-final-test-evaluation-v1"
TASK_RUN_IDS = {
    "recovery": "stage5-final-test-evaluation-recovery-v1",
    "icu_stay_time": "stage5-final-test-evaluation-icu-stay-time-v1",
    "organ_support": "stage5-final-test-evaluation-organ-support-v1",
}
MODEL_FAMILY = {"recovery": "xgboost", "icu_stay_time": "gru", "organ_support": "xgboost"}
LABEL_VERSION = {
    "recovery": "synthetic_xgb_target_contract_v1",
    "icu_stay_time": "synthetic_phase9_final_target_contract_v1",
    "organ_support": "synthetic_xgb_target_contract_v1",
}


def fix_registry() -> None:
    registry_path = ROOT / "experiments/registry.csv"
    with registry_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        rows = list(reader)

    if any(row["run_id"] in TASK_RUN_IDS.values() for row in rows):
        print("per-task runs already present; skipping registry insert")
        return
    aggregate = next(row for row in rows if row["run_id"] == AGGREGATE_RUN_ID)
    common = {
        name: aggregate[name]
        for name in ("timestamp_utc", "code_commit", "seed", "config_ref", "config_hash",
                     "split_hash", "preprocessor_ref", "preprocessor_sha256", "split_ref",
                     "status", "run_type", "finalized", "model_artifact_ref", "model_sha256",
                     "metrics_ref", "feature_version")
    }
    new_rows = []
    for task, run_id in TASK_RUN_IDS.items():
        row = {name: "" for name in fieldnames}
        row.update(common)
        row.update({
            "run_id": run_id, "task": task, "model_family": MODEL_FAMILY[task],
            "label_version": LABEL_VERSION[task], "parent_run_id": AGGREGATE_RUN_ID,
            "notes": "Stage 5 one-time final-test evaluation, " + task + " task companion of " + AGGREGATE_RUN_ID,
        })
        new_rows.append(row)

    with registry_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        for row in new_rows:
            writer.writerow(row)
    count = validate_registry(registry_path)
    print("validate_registry_row_count=" + str(count))


def fix_artifacts() -> None:
    artifacts_path = ROOT / "experiments/artifacts.csv"
    with artifacts_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        rows = list(reader)
    changed = 0
    for row in rows:
        if row["artifact_id"].startswith("stage5-final-test-prediction-") and row["producing_run_id"] == AGGREGATE_RUN_ID:
            task = row["task"]
            row["producing_run_id"] = TASK_RUN_IDS[task]
            changed += 1
    if changed:
        with artifacts_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    print("prediction_rows_repointed=" + str(changed))


def fix_g4() -> None:
    g4_path = ROOT / "artifacts/governance/g4_test_evaluation_freeze_v1.json"
    payload = json.loads(g4_path.read_text())
    payload["registry"]["artifacts_sha256"] = sha256_file(ROOT / "experiments/artifacts.csv")
    payload["registry"]["registry_sha256"] = sha256_file(ROOT / "experiments/registry.csv")
    g4_path.write_bytes(canonical_json_bytes(payload))
    print("g4_artifacts_sha256=" + payload["registry"]["artifacts_sha256"])
    print("g4_registry_sha256=" + payload["registry"]["registry_sha256"])
    print("g4_file_sha256=" + sha256_file(g4_path))


if __name__ == "__main__":
    fix_registry()
    fix_artifacts()
    fix_g4()
