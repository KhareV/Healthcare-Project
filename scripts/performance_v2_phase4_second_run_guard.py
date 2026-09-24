"""Performance-v2 Phase 4, Part 18: second-run guard. Proves a second
ordinary final evaluation would be rejected BEFORE any fresh-row loading.
Does not invoke model inference and does not call load_fresh_test_rows.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.synthetic.config import canonical_json_bytes  # noqa: E402
from performance_v2.fresh_test_access import (  # noqa: E402
    FINAL_V2_RUN_COMPLETED,
    FreshTestAccessError,
    authorize_one_final_run,
    current_state,
    has_ever_consumed_access,
)
from vedant_infra.hashing import sha256_file  # noqa: E402

GOV_DIR = ROOT / "artifacts/performance_v2/governance"


def main() -> None:
    state = current_state(ROOT)
    consumed = has_ever_consumed_access(ROOT)

    blocked = False
    detail = None
    try:
        authorize_one_final_run(ROOT, authorized_by="second_run_guard_probe", reason="MUST_BE_REJECTED")
    except FreshTestAccessError as error:
        blocked = True
        detail = str(error)

    if not blocked:
        raise SystemExit("SECOND_RUN_GUARD_FAILED: authorize_one_final_run unexpectedly succeeded on a second call")

    result = {
        "status": "BLOCKED_ALREADY_CONSUMED" if consumed and state == FINAL_V2_RUN_COMPLETED else "BLOCKED_INVALID_STATE",
        "artifact_version": "performance_v2_phase4_second_run_guard_v1",
        "current_access_state": state,
        "has_ever_consumed_access": consumed,
        "second_authorize_attempt_blocked": True,
        "second_authorize_error_detail": detail,
        "model_inference_invoked": False,
        "fresh_test_rows_loaded": False,
    }
    guard_path = GOV_DIR / "v2_second_run_guard_v1.json"
    guard_path.write_bytes(canonical_json_bytes(result))
    print("wrote", guard_path, sha256_file(guard_path))
    print(result["status"])


if __name__ == "__main__":
    main()
