"""Read-only Phase-15 prerequisite and governance audit."""

import hashlib
import json
from pathlib import Path
from typing import Mapping


def audit_error_analysis(root: Path) -> Mapping[str, object]:
    config = root / "configs/error_analysis/error_analysis_v1.json"
    selected = root / "artifacts/models/selected_models_v1.json"
    payload = json.loads(config.read_text(encoding="utf-8"))
    return {
        "slice_spec_exists": True,
        "slice_spec_version": payload["spec_version"],
        "slice_spec_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
        "slice_spec_status": payload["status"],
        "selected_models_v1_exists": selected.is_file(),
        "real_validation_analysis_authorized": bool(
            selected.is_file()
            and payload["status"] == "FROZEN"
            and payload["final_test_authorized"] is False
        ),
        "final_test_status": payload["final_test_status"],
        "final_test_opened": False,
    }


if __name__ == "__main__":
    print(json.dumps(audit_error_analysis(Path.cwd()), indent=2, sort_keys=True))
