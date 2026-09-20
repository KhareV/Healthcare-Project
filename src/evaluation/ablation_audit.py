"""Read-only Phase-14 prerequisite audit."""

import json
from pathlib import Path

from evaluation.ablations import (
    FINAL_TEST_DEFERRED,
    MASK_GROUP_BLOCKER,
    MATRIX_BLOCKER,
    REAL_RECOVERY_MODEL_BLOCKER,
    RETRAINING_UNLOCKED,
)


def audit_repository(root="."):
    root_path = Path(root)
    selected = root_path / "artifacts/models/selected_models_v1.json"
    artifacts = list((root_path / "artifacts/ablations").glob("*"))
    return {
        "phase": 14,
        "audit_mode": "read_only_no_test_access",
        "selected_models_v1_exists": selected.is_file(),
        "real_selected_recovery_model_exists": False,
        "production_feature_schema_exists": False,
        "production_tslo_contract_frozen": False,
        "authoritative_sofa_component_flags_available": False,
        "real_ablation_ready": False,
        "real_ablation_status": REAL_RECOVERY_MODEL_BLOCKER,
        "final_test_sensitivity_status": FINAL_TEST_DEFERRED,
        "ablation_artifact_count": len(artifacts),
        "test_accessed": False,
        "blocked_or_unlocked": [MASK_GROUP_BLOCKER, MATRIX_BLOCKER, RETRAINING_UNLOCKED],
    }


def main():
    print(json.dumps(audit_repository(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
