"""One-off correction, applied before the Part-B commit.

ensure_stage5_run_registered() originally left several
COMPLETED_REQUIRED_FIELDS columns empty on the aggregate
"stage5-final-test-evaluation-v1" registry row (task, model_family, seed,
config_ref, config_hash, feature_version, label_version, model_artifact_ref,
model_sha256, metrics_ref), which validate_registry() correctly rejects.
That is a pure registry-bookkeeping/metadata gap -- it does not change any
model, split, calibration, threshold, or scientific result -- but every
future call into vedant_infra.registry's write path calls
validate_registry() and would fail until it is fixed. src/experiments/
stage5_final_test.py's ensure_stage5_run_registered() is already fixed to
populate these fields correctly for any future registration; this script
applies the identical values to the one row that was already written by
the real Part-B run (which will not run again -- the function is
idempotent and returns early once the run_id exists).

Because experiments/registry.csv's bytes change, this also recomputes and
patches the one dependent field in the already-created G4 freeze
(artifacts/governance/g4_test_evaluation_freeze_v1.json): registry.
registry_sha256. Nothing else in G4 changes. No model, prediction, or
metric artifact is touched, and no test-partition data is read.
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


def main() -> None:
    registry_path = ROOT / "experiments/registry.csv"
    with registry_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        rows = list(reader)

    config_hash = sha256_file(ROOT / "configs/final_test_v1.json")
    manifest_sha256 = sha256_file(ROOT / "artifacts/models/selected_models_v1.json")
    metrics_ref = "artifacts/final_test/metrics/metrics_v1.json"

    found = False
    for row in rows:
        if row["run_id"] == "stage5-final-test-evaluation-v1":
            found = True
            row.update({
                "task": "recovery+icu_stay_time+organ_support",
                "model_family": "xgboost+gru",
                "seed": "20260921",
                "config_ref": "configs/final_test_v1.json",
                "config_hash": config_hash,
                "feature_version": "synthetic_feature_schema_v2",
                "label_version": "synthetic_xgb_target_contract_v1+synthetic_phase9_final_target_contract_v1",
                "model_artifact_ref": "artifacts/models/selected_models_v1.json",
                "model_sha256": manifest_sha256,
                "metrics_ref": metrics_ref,
            })
    if not found:
        raise SystemExit("stage5-final-test-evaluation-v1 row not found; nothing to fix")

    with registry_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    row_count = validate_registry(registry_path)
    print("validate_registry_row_count=" + str(row_count))

    g4_path = ROOT / "artifacts/governance/g4_test_evaluation_freeze_v1.json"
    payload = json.loads(g4_path.read_text())
    old_hash = payload["registry"]["registry_sha256"]
    new_hash = sha256_file(registry_path)
    payload["registry"]["registry_sha256"] = new_hash
    g4_path.write_bytes(canonical_json_bytes(payload))
    print("old_registry_sha256=" + old_hash)
    print("new_registry_sha256=" + new_hash)
    print("g4_file_sha256=" + sha256_file(g4_path))


if __name__ == "__main__":
    main()
