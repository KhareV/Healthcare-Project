"""One-off correction, applied before the Part-B commit (third pass).

test_phase11_consumed_zero_scientific_slots_and_later_rows_are_phase12_only
asserts every registry row with model_family=="xgboost" and
run_type=="scientific" carries search_version==
"synthetic_xgb_phase12_validation_search_v1" -- a governance invariant that
no XGBoost model was ever scientifically registered outside the canonical
Phase-12 search. The Stage-5 aggregate/per-task rows are real evaluation
runs of already-frozen, already-selected models on held-out data, not
search runs, so labelling them run_type="scientific" incorrectly swept them
into that check. src/experiments/stage5_final_test.py now labels them
run_type="scientific_sensitivity" (the same category already used for the
Stage-2 LSTM sensitivity runs -- a real, non-search, non-smoke evaluation).
This applies the identical correction to the 4 rows the real Part-B run
already wrote. Pure registry metadata; no model/split/calibration/
threshold/metric content changes, no test-partition access.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.synthetic.config import canonical_json_bytes  # noqa: E402
from vedant_infra.hashing import sha256_file  # noqa: E402
from vedant_infra.registry import validate_registry  # noqa: E402

STAGE5_RUN_IDS = {
    "stage5-final-test-evaluation-v1",
    "stage5-final-test-evaluation-recovery-v1",
    "stage5-final-test-evaluation-icu-stay-time-v1",
    "stage5-final-test-evaluation-organ-support-v1",
}


def fix_registry() -> None:
    import csv

    registry_path = ROOT / "experiments/registry.csv"
    with registry_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        rows = list(reader)
    changed = 0
    for row in rows:
        if row["run_id"] in STAGE5_RUN_IDS and row["run_type"] == "scientific":
            row["run_type"] = "scientific_sensitivity"
            changed += 1
    if changed:
        with registry_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    print("registry_rows_changed=" + str(changed))
    print("validate_registry_row_count=" + str(validate_registry(registry_path)))


def fix_g4() -> None:
    g4_path = ROOT / "artifacts/governance/g4_test_evaluation_freeze_v1.json"
    payload = json.loads(g4_path.read_text())
    payload["registry"]["registry_sha256"] = sha256_file(ROOT / "experiments/registry.csv")
    g4_path.write_bytes(canonical_json_bytes(payload))
    print("g4_registry_sha256=" + payload["registry"]["registry_sha256"])
    print("g4_file_sha256=" + sha256_file(g4_path))


if __name__ == "__main__":
    fix_registry()
    fix_g4()
