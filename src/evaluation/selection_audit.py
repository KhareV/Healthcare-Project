"""Read-only Phase-12 prerequisite audit; never performs scientific selection."""

import csv
import json
from pathlib import Path

from evaluation.selection_validation import (
    EXACT_TIE_UNLOCKED,
    REAL_SELECTION_BLOCKER,
    TOLERANCE_UNLOCKED,
)


def audit_repository(root="."):
    root_path = Path(root)
    dry_run_manifests = sorted((root_path / "artifacts/search/dry_run").glob("*.json"))
    valid_dry_runs = 0
    for path in dry_run_manifests:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("run_type") == "synthetic_dry_run" and payload.get("candidate_count") == 30:
            valid_dry_runs += 1
    registry_path = root_path / "experiments/registry.csv"
    registry_rows = []
    if registry_path.is_file():
        with registry_path.open("r", encoding="utf-8", newline="") as handle:
            registry_rows = list(csv.DictReader(handle))
    scientific_search_rows = [
        row
        for row in registry_rows
        if row.get("model_family") in {"gru", "xgboost"}
        and row.get("candidate_id")
        and "SYNTHETIC_DRY_RUN" not in row.get("notes", "")
    ]
    return {
        "phase": 12,
        "audit_mode": "read_only_no_test_access",
        "synthetic_exact_30_manifests": valid_dry_runs,
        "scientific_candidate_registry_rows": len(scientific_search_rows),
        "real_selection_ready": False,
        "real_selection_status": REAL_SELECTION_BLOCKER,
        "selected_models_v1_exists": (root_path / "artifacts/models/selected_models_v1.json").exists(),
        "test_accessed": False,
        "unlocked": [EXACT_TIE_UNLOCKED, TOLERANCE_UNLOCKED],
    }


def main():
    print(json.dumps(audit_repository(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
