"""Parameterised one-time access guard for a sealed final-test cohort.

Generalises performance_v2.fresh_test_access (module-level V2 constants) to
any cohort_id / ledger path / rows loader. Ledger is a hash-chained,
append-only JSON history written atomically; any missing, corrupt or
inconsistent ledger fails closed. Rows are only ever returned by
`consume_and_load_rows`, which marks the access CONSUMED *before* invoking
the loader and can succeed at most once per ledger lifetime.

Development code must call `assert_not_sealed(...)` on cohort ids / paths it
is about to read; registered sealed cohorts always raise.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

SEALED_NOT_ACCESSED = "SEALED_NOT_ACCESSED"
AUTHORIZED_FOR_ONE_FINAL_RUN = "AUTHORIZED_FOR_ONE_FINAL_RUN"
FINAL_TEST_ACCESS_CONSUMED = "FINAL_TEST_ACCESS_CONSUMED"
FINAL_RUN_COMPLETED = "FINAL_RUN_COMPLETED"
BLOCKED_ALREADY_CONSUMED = "BLOCKED_ALREADY_CONSUMED"

LEDGER_VERSION = "performance_v3_sealed_cohort_ledger_v1"
_TRANSITIONS = {
    SEALED_NOT_ACCESSED: AUTHORIZED_FOR_ONE_FINAL_RUN,
    AUTHORIZED_FOR_ONE_FINAL_RUN: FINAL_TEST_ACCESS_CONSUMED,
    FINAL_TEST_ACCESS_CONSUMED: FINAL_RUN_COMPLETED,
}
_SHA = re.compile(r"^[0-9a-f]{64}$")
_GENESIS = "0" * 64


class SealedCohortAccessError(RuntimeError):
    """Raised for any out-of-order, repeated or tampered access."""


_SEALED_IDS: Dict[str, List[Path]] = {}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode()


def _entry_hash(previous: str, entry: Mapping[str, object]) -> str:
    body = {k: v for k, v in entry.items() if k != "entry_sha256"}
    return hashlib.sha256(previous.encode() + _canonical(body)).hexdigest()


def assert_not_sealed(path_or_cohort_id: Union[str, os.PathLike]) -> None:
    """Raise if the cohort id or filesystem path belongs to a registered
    sealed cohort. Intended for loaders in development/search code."""

    text = os.fspath(path_or_cohort_id)
    if text in _SEALED_IDS:
        raise SealedCohortAccessError("development access to sealed cohort refused: " + text)
    try:
        candidate = Path(text).resolve()
    except (OSError, RuntimeError):
        return
    for cohort_id, prefixes in _SEALED_IDS.items():
        for prefix in prefixes:
            if candidate == prefix or prefix in candidate.parents:
                raise SealedCohortAccessError("development access to sealed cohort %r path refused: %s" % (cohort_id, text))


def _reset_registry_for_tests() -> None:
    _SEALED_IDS.clear()


class SealedCohortAccess:
    def __init__(
        self,
        root: Union[str, Path],
        cohort_id: str,
        state_path: Union[str, Path],
        rows_loader: Callable[[Path], Sequence[Mapping[str, object]]],
        expected_paths: Sequence[Union[str, Path]] = (),
        sealed_paths: Sequence[Union[str, Path]] = (),
    ) -> None:
        self.root = Path(root).resolve()
        self.cohort_id = cohort_id
        state = Path(state_path)
        self.state_path = state if state.is_absolute() else self.root / state
        self.rows_loader = rows_loader
        self.expected_paths = tuple(Path(p) if Path(p).is_absolute() else self.root / p for p in expected_paths)
        prefixes = [self.root / Path(p) if not Path(p).is_absolute() else Path(p) for p in sealed_paths]
        prefixes += list(self.expected_paths)
        _SEALED_IDS[cohort_id] = [p.resolve() for p in prefixes]

    # ---- ledger io -------------------------------------------------------
    def _lock(self):
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = self.state_path.with_suffix(self.state_path.suffix + ".lock")
        try:
            return os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY), lock_path
        except FileExistsError as error:
            raise SealedCohortAccessError("ledger lock held; concurrent access refused: " + str(lock_path)) from error

    def _unlock(self, handle):
        fd, path = handle
        os.close(fd)
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass

    def _write(self, payload: Mapping[str, object]) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
        with open(tmp, "wb") as handle:
            handle.write(_canonical(payload))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, self.state_path)

    def _read(self) -> Dict[str, object]:
        if not self.state_path.is_file():
            raise SealedCohortAccessError("sealed-cohort ledger missing; refusing to guess a state: " + str(self.state_path))
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise SealedCohortAccessError("sealed-cohort ledger unreadable/corrupt") from error
        self._verify(payload)
        return payload

    def _verify(self, payload: Mapping[str, object]) -> None:
        if not isinstance(payload, dict) or payload.get("ledger_version") != LEDGER_VERSION:
            raise SealedCohortAccessError("ledger version/shape invalid")
        if payload.get("cohort_id") != self.cohort_id:
            raise SealedCohortAccessError("ledger belongs to a different cohort id")
        history = payload.get("history")
        if not isinstance(history, list) or not history:
            raise SealedCohortAccessError("ledger history missing")
        previous = _GENESIS
        status = None
        for entry in history:
            if not isinstance(entry, dict) or entry.get("entry_sha256") != _entry_hash(previous, entry):
                raise SealedCohortAccessError("ledger hash chain broken (tampered or reordered history)")
            previous = entry["entry_sha256"]
            if "to" in entry:
                if status is None and entry.get("from") is not None:
                    raise SealedCohortAccessError("ledger genesis invalid")
                if status is not None and _TRANSITIONS.get(status) != entry["to"]:
                    raise SealedCohortAccessError("ledger contains an illegal transition")
                status = entry["to"]
        if status != payload.get("status") or previous != payload.get("history_head_sha256"):
            raise SealedCohortAccessError("ledger status/head inconsistent with history")

    def _append(self, payload: Dict[str, object], entry: Dict[str, object]) -> Dict[str, object]:
        history = list(payload["history"])
        previous = history[-1]["entry_sha256"] if history else _GENESIS
        entry = dict(entry, timestamp_utc=_utc_now())
        entry["entry_sha256"] = _entry_hash(previous, entry)
        history.append(entry)
        updated = dict(payload)
        updated["history"] = history
        updated["history_head_sha256"] = entry["entry_sha256"]
        if "to" in entry:
            updated["status"] = entry["to"]
        return updated

    # ---- public api ------------------------------------------------------
    def initialize(self) -> None:
        """Create the ledger at SEALED_NOT_ACCESSED. Refuses to overwrite."""

        handle = self._lock()
        try:
            if self.state_path.exists():
                raise SealedCohortAccessError("ledger already exists; refusing to re-initialise")
            payload = {"ledger_version": LEDGER_VERSION, "cohort_id": self.cohort_id, "status": SEALED_NOT_ACCESSED, "history": [], "history_head_sha256": _GENESIS}
            payload = self._append(payload, {"from": None, "to": SEALED_NOT_ACCESSED})
            self._write(payload)
        finally:
            self._unlock(handle)

    def state(self) -> str:
        return str(self._read()["status"])

    def history(self) -> Tuple[Mapping[str, object], ...]:
        return tuple(self._read()["history"])

    def has_ever_consumed(self) -> bool:
        return any(e.get("to") == FINAL_TEST_ACCESS_CONSUMED for e in self._read()["history"])

    def effective_state(self) -> str:
        """BLOCKED_ALREADY_CONSUMED once access was consumed (a second run is denied)."""

        return BLOCKED_ALREADY_CONSUMED if self.has_ever_consumed() else self.state()

    def _transition(self, expected: str, extra: Mapping[str, object]) -> None:
        handle = self._lock()
        try:
            payload = self._read()
            if payload["status"] != expected:
                raise SealedCohortAccessError("cannot advance: state is %r, expected %r" % (payload["status"], expected))
            target = _TRANSITIONS[expected]
            self._write(self._append(payload, dict({"from": expected, "to": target}, **extra)))
        finally:
            self._unlock(handle)

    def authorize(self, freeze_manifest_sha256: str, plan_sha256: str, authorized_by: str = "coordinator") -> None:
        for name, value in (("freeze_manifest_sha256", freeze_manifest_sha256), ("plan_sha256", plan_sha256)):
            if not isinstance(value, str) or not _SHA.match(value):
                raise SealedCohortAccessError(name + " must be a 64-hex SHA-256")
        self._transition(
            SEALED_NOT_ACCESSED,
            {"freeze_manifest_sha256": freeze_manifest_sha256, "plan_sha256": plan_sha256, "authorized_by": authorized_by},
        )

    def consume_and_load_rows(self) -> Tuple[Mapping[str, object], ...]:
        """Mark access CONSUMED first, then load. Only possible once."""

        handle = self._lock()
        try:
            payload = self._read()
            if any(e.get("to") == FINAL_TEST_ACCESS_CONSUMED for e in payload["history"]):
                self._write(self._append(payload, {"event": BLOCKED_ALREADY_CONSUMED, "detail": "repeat consume attempt refused"}))
                raise SealedCohortAccessError(BLOCKED_ALREADY_CONSUMED + ": final-test access was already consumed")
            if payload["status"] != AUTHORIZED_FOR_ONE_FINAL_RUN:
                raise SealedCohortAccessError("access not authorized: state is %r" % payload["status"])
            missing = [str(p) for p in self.expected_paths if not p.exists()]
            if missing:
                raise SealedCohortAccessError("expected sealed-cohort paths missing: " + ", ".join(missing))
            self._write(self._append(payload, {"from": AUTHORIZED_FOR_ONE_FINAL_RUN, "to": FINAL_TEST_ACCESS_CONSUMED}))
        finally:
            self._unlock(handle)
        try:
            return tuple(self.rows_loader(self.root))
        except Exception as error:  # consumption stands; record the incident, no silent retry
            handle = self._lock()
            try:
                payload = self._read()
                self._write(self._append(payload, {"event": "LOADER_EXCEPTION", "detail": repr(error)[:500]}))
            finally:
                self._unlock(handle)
            raise

    def complete(self, result_sha256: Optional[str] = None) -> None:
        extra: Dict[str, object] = {}
        if result_sha256 is not None:
            if not _SHA.match(result_sha256):
                raise SealedCohortAccessError("result_sha256 must be a 64-hex SHA-256")
            extra["result_sha256"] = result_sha256
        self._transition(FINAL_TEST_ACCESS_CONSUMED, extra)
