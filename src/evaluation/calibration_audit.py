"""Read-only Phase-13 prerequisite audit."""

import json
from pathlib import Path

from evaluation.calibration_validation import (
    ISOTONIC_DETAIL_UNLOCKED,
    ISOTONIC_WEIGHTING_UNLOCKED,
    REAL_SELECTED_MODEL_BLOCKER,
)
from evaluation.threshold import THRESHOLD_SEARCH_UNLOCKED, THRESHOLD_TIE_BLOCKER


def audit_repository(root="."):
    root_path = Path(root)
    selected = root_path / "artifacts/models/selected_models_v1.json"
    calibration_files = list((root_path / "artifacts/calibration").glob("*"))
    threshold_files = list((root_path / "artifacts/thresholds").glob("*"))
    return {
        "phase": 13,
        "audit_mode": "read_only_no_test_access",
        "selected_models_v1_exists": selected.is_file(),
        "real_selected_support_model_exists": False,
        "real_calibration_ready": False,
        "real_calibration_status": REAL_SELECTED_MODEL_BLOCKER,
        "calibration_artifact_count": len(calibration_files),
        "threshold_artifact_count": len(threshold_files),
        "test_accessed": False,
        "unlocked": [
            ISOTONIC_DETAIL_UNLOCKED,
            ISOTONIC_WEIGHTING_UNLOCKED,
            THRESHOLD_SEARCH_UNLOCKED,
            THRESHOLD_TIE_BLOCKER,
        ],
    }


def main():
    print(json.dumps(audit_repository(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
