"""Formal, reasoned recovery from the Stage-5 Part B loader crash.

The one authorized final-test exposure was consumed by
vedant_infra.g3.guarded_test_access before any test-partition row was read,
because src/experiments/stage5_final_test.py's load_frozen_models()
redundantly re-ran the full G3 nonuse/access-state audit inside the guarded
loader (see artifacts/governance/stage5_final_test_access_incident_v1.json).
That defect is now fixed (load_frozen_models(verify_g3=False) for the
partition="test" path). No science-affecting artifact, model, split,
calibrator, threshold, or config changed.

This script performs the only two governance operations capable of granting
a fresh, single, authorized final-test exposure: invalidate the consumed G3
marker with a documented reason, then re-freeze G3. freeze_g3() recomputes
every dependency binding live from the current repository state, so the new
marker is expected to bind byte-identical dependency hashes to the original
(nothing scientific changed between the two freezes) other than a new
created_at_utc/marker_sha256.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vedant_infra.g3 import freeze_g3, invalidate_for_reset  # noqa: E402


REASON = (
    "Stage 5 Part B guarded run crashed with a governance-timing engineering "
    "defect (redundant validate_g3_marker call inside load_frozen_models, "
    "which cannot succeed once guarded_test_access has already flipped "
    "test_access_state to FINAL_RUN_COMPLETED) before reading any "
    "test-partition row, before any inference, and before any metric or "
    "bootstrap computation. Zero science-affecting artifact changed. Fixed "
    "in load_frozen_models (verify_g3=False for the already-guarded test "
    "path); this reset grants exactly one fresh authorized exposure to "
    "complete the scientific final-test evaluation the crash prevented."
)


def main() -> None:
    original = json.loads((ROOT / "artifacts/governance/g3_freeze.json").read_text())
    archived = invalidate_for_reset(ROOT, reason=REASON)
    print("archived_marker=" + str(archived))
    new_marker_path = freeze_g3(ROOT, scope="real")
    print("new_marker=" + str(new_marker_path))
    fresh = json.loads(new_marker_path.read_text())
    original_deps = {d["ref"]: d["sha256"] for d in original["dependencies"]}
    fresh_deps = {d["ref"]: d["sha256"] for d in fresh["dependencies"]}
    identical = original_deps == fresh_deps
    print("dependency_hashes_identical_to_original=" + str(identical))
    if not identical:
        raise SystemExit("REFUSING: re-frozen G3 dependency hashes differ from the original freeze")
    print("status=" + fresh["status"])
    print("test_accessed=" + str(fresh["test_accessed"]))


if __name__ == "__main__":
    main()
