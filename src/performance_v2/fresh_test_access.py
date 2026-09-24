"""The single choke point through which fresh v2 test-cohort rows may ever
be read. Ordinary code MUST import and use this module rather than reading
``artifacts/performance_v2/phase3/fresh_test_cohort/...`` directly.

State machine (persisted in
``artifacts/performance_v2/governance/v2_fresh_test_access_state.json``):

    SEALED_NOT_ACCESSED
        -> AUTHORIZED_FOR_ONE_FINAL_RUN
        -> FINAL_V2_TEST_ACCESS_CONSUMED
        -> FINAL_V2_RUN_COMPLETED

Phase 3 leaves the state at ``SEALED_NOT_ACCESSED`` and never calls
``authorize_one_final_run``/``consume_access``/``complete_run``. Only a
future guarded Phase-4 command may advance the state. ``load_fresh_test_rows``
refuses to return rows unless the state is ``FINAL_V2_TEST_ACCESS_CONSUMED``
or ``FINAL_V2_RUN_COMPLETED`` -- i.e. after access has been explicitly
authorized and consumed, never merely because the cohort exists on disk.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Tuple

STATE_PATH = "artifacts/performance_v2/governance/v2_fresh_test_access_state.json"

SEALED_NOT_ACCESSED = "SEALED_NOT_ACCESSED"
AUTHORIZED_FOR_ONE_FINAL_RUN = "AUTHORIZED_FOR_ONE_FINAL_RUN"
FINAL_V2_TEST_ACCESS_CONSUMED = "FINAL_V2_TEST_ACCESS_CONSUMED"
FINAL_V2_RUN_COMPLETED = "FINAL_V2_RUN_COMPLETED"

_ALLOWED_TRANSITIONS = {
    SEALED_NOT_ACCESSED: AUTHORIZED_FOR_ONE_FINAL_RUN,
    AUTHORIZED_FOR_ONE_FINAL_RUN: FINAL_V2_TEST_ACCESS_CONSUMED,
    FINAL_V2_TEST_ACCESS_CONSUMED: FINAL_V2_RUN_COMPLETED,
}

_READ_ALLOWED_STATES = {FINAL_V2_TEST_ACCESS_CONSUMED, FINAL_V2_RUN_COMPLETED}

PRE_SPLIT_PATH = "artifacts/performance_v2/phase3/fresh_test_cohort/pre_split/pre_split_scientific_package.jsonl"
STATICS_PATH = "artifacts/performance_v2/phase3/fresh_test_cohort/timeline/canonical_statics.jsonl"
SPLIT_PATH = "artifacts/performance_v2/phase3/fresh_test_split_v1.csv"


class FreshTestAccessError(RuntimeError):
    """Raised whenever code attempts to read fresh v2 test rows out of turn."""


def _read_state(root: Path) -> Mapping[str, object]:
    path = root / STATE_PATH
    if not path.is_file():
        raise FreshTestAccessError("fresh v2 test access-state marker is missing; refusing to guess a state")
    return json.loads(path.read_text(encoding="utf-8"))


def current_state(root: Path) -> str:
    return str(_read_state(root)["status"])


def _write_state(root: Path, status: str, extra: Mapping[str, object]) -> None:
    from data.synthetic.config import canonical_json_bytes  # local import: avoid import cost for read-only callers

    payload = dict(_read_state(root))
    payload["status"] = status
    payload.update(extra)
    (root / STATE_PATH).write_bytes(canonical_json_bytes(payload))


def _transition(root: Path, expected_current: str, target: str, extra: Mapping[str, object]) -> None:
    state = current_state(root)
    if state != expected_current:
        raise FreshTestAccessError(f"cannot transition to {target!r}: current state is {state!r}, expected {expected_current!r}")
    if _ALLOWED_TRANSITIONS.get(expected_current) != target:
        raise FreshTestAccessError(f"illegal transition {expected_current!r} -> {target!r}")
    _write_state(root, target, extra)


def authorize_one_final_run(root: Path, *, authorized_by: str, reason: str) -> None:
    """Phase-4 only. Moves SEALED_NOT_ACCESSED -> AUTHORIZED_FOR_ONE_FINAL_RUN."""

    _transition(root, SEALED_NOT_ACCESSED, AUTHORIZED_FOR_ONE_FINAL_RUN, {"authorized_by": authorized_by, "authorization_reason": reason})


def consume_access(root: Path) -> None:
    """Phase-4 only. Moves AUTHORIZED_FOR_ONE_FINAL_RUN -> FINAL_V2_TEST_ACCESS_CONSUMED.

    Call this immediately before the one guarded read of fresh-test rows.
    """

    _transition(root, AUTHORIZED_FOR_ONE_FINAL_RUN, FINAL_V2_TEST_ACCESS_CONSUMED, {})


def complete_run(root: Path) -> None:
    """Phase-4 only. Moves FINAL_V2_TEST_ACCESS_CONSUMED -> FINAL_V2_RUN_COMPLETED."""

    _transition(root, FINAL_V2_TEST_ACCESS_CONSUMED, FINAL_V2_RUN_COMPLETED, {})


def load_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_fresh_test_rows(root: Path) -> Tuple[Mapping[str, object], ...]:
    """The only function in this repository that may return fresh v2 test rows.

    Raises FreshTestAccessError unless the access state has already been
    explicitly authorized and consumed by a guarded Phase-4 command -- never
    merely because the sealed cohort files exist on disk.
    """

    state = current_state(root)
    if state not in _READ_ALLOWED_STATES:
        raise FreshTestAccessError(
            f"fresh v2 test access is not authorized: current state is {state!r}; "
            f"reading is only permitted in {sorted(_READ_ALLOWED_STATES)!r}"
        )
    return tuple(load_jsonl(root / PRE_SPLIT_PATH))
