import json

import pytest

from performance_v3 import test_access as ta
from performance_v3.test_access import (
    BLOCKED_ALREADY_CONSUMED,
    SealedCohortAccess,
    SealedCohortAccessError,
    assert_not_sealed,
)

SHA = "a" * 64
SHB = "b" * 64


@pytest.fixture(autouse=True)
def _clean_registry():
    ta._reset_registry_for_tests()
    yield
    ta._reset_registry_for_tests()


def _make(tmp_path, loader=None, cohort="SYN-V3-T"):
    data = tmp_path / "cohort"
    data.mkdir(exist_ok=True)
    (data / "rows.jsonl").write_text('{"stay_id":"s1"}\n')
    calls = []

    def default_loader(root):
        calls.append(1)
        return [{"stay_id": "s1"}]

    access = SealedCohortAccess(
        tmp_path, cohort, "gov/state.json", loader or default_loader, expected_paths=["cohort/rows.jsonl"], sealed_paths=["cohort"]
    )
    access.initialize()
    return access, calls


def test_full_lifecycle_and_consumed_before_load(tmp_path):
    seen = {}

    def loader(root):
        seen["state_during_load"] = json.loads((root / "gov/state.json").read_text())["status"]
        return [{"stay_id": "s1"}]

    access, _ = _make(tmp_path, loader)
    assert access.state() == ta.SEALED_NOT_ACCESSED
    with pytest.raises(SealedCohortAccessError):
        access.consume_and_load_rows()  # not authorized
    access.authorize(SHA, SHB)
    rows = access.consume_and_load_rows()
    assert rows == ({"stay_id": "s1"},)
    assert seen["state_during_load"] == ta.FINAL_TEST_ACCESS_CONSUMED  # marked BEFORE rows returned
    access.complete(SHA)
    assert access.state() == ta.FINAL_RUN_COMPLETED
    assert [e.get("to") for e in access.history() if "to" in e] == [
        ta.SEALED_NOT_ACCESSED, ta.AUTHORIZED_FOR_ONE_FINAL_RUN, ta.FINAL_TEST_ACCESS_CONSUMED, ta.FINAL_RUN_COMPLETED
    ]


def test_second_consumption_refused_and_logged(tmp_path):
    access, calls = _make(tmp_path)
    access.authorize(SHA, SHB)
    access.consume_and_load_rows()
    assert access.effective_state() == BLOCKED_ALREADY_CONSUMED
    with pytest.raises(SealedCohortAccessError, match=BLOCKED_ALREADY_CONSUMED):
        access.consume_and_load_rows()
    assert len(calls) == 1
    assert any(e.get("event") == BLOCKED_ALREADY_CONSUMED for e in access.history())
    access.complete()
    with pytest.raises(SealedCohortAccessError):
        access.consume_and_load_rows()
    with pytest.raises(SealedCohortAccessError):
        access.authorize(SHA, SHB)


def test_authorize_requires_sha_and_order(tmp_path):
    access, _ = _make(tmp_path)
    with pytest.raises(SealedCohortAccessError):
        access.authorize("short", SHB)
    with pytest.raises(SealedCohortAccessError):
        access.complete()


def test_missing_and_corrupt_ledger_fail_closed(tmp_path):
    access, _ = _make(tmp_path)
    state = tmp_path / "gov/state.json"
    state.write_text("{not json")
    with pytest.raises(SealedCohortAccessError):
        access.state()
    state.unlink()
    with pytest.raises(SealedCohortAccessError):
        access.consume_and_load_rows()


def test_tampered_ledgers_detected(tmp_path):
    access, _ = _make(tmp_path)
    access.authorize(SHA, SHB)
    state = tmp_path / "gov/state.json"
    original = state.read_text()
    # 1) forge status forward without history
    forged = json.loads(original); forged["status"] = ta.FINAL_TEST_ACCESS_CONSUMED
    state.write_text(json.dumps(forged))
    with pytest.raises(SealedCohortAccessError):
        access.consume_and_load_rows()
    # 2) rewrite an old history entry (reset back to SEALED by dropping the authorize entry)
    forged = json.loads(original); forged["history"] = forged["history"][:1]; forged["status"] = ta.SEALED_NOT_ACCESSED
    state.write_text(json.dumps(forged))
    with pytest.raises(SealedCohortAccessError):
        access.state()
    # 3) edit field inside an entry
    forged = json.loads(original); forged["history"][1]["plan_sha256"] = "c" * 64
    state.write_text(json.dumps(forged))
    with pytest.raises(SealedCohortAccessError):
        access.state()
    # 4) ledger from another cohort
    forged = json.loads(original); forged["cohort_id"] = "OTHER"
    state.write_text(json.dumps(forged))
    with pytest.raises(SealedCohortAccessError):
        access.state()


def test_consumption_after_rollback_to_earlier_valid_ledger_is_still_blocked_by_missing_consumption_only_if_chain_valid(tmp_path):
    # restoring an earlier *valid* snapshot is outside the ledger's power; the guard must at
    # least never allow two consumptions within one ledger lifetime.
    access, calls = _make(tmp_path)
    access.authorize(SHA, SHB)
    access.consume_and_load_rows()
    with pytest.raises(SealedCohortAccessError):
        access.consume_and_load_rows()
    assert len(calls) == 1


def test_loader_failure_keeps_consumption_and_records_incident(tmp_path):
    def boom(root):
        raise RuntimeError("disk exploded")

    access, _ = _make(tmp_path, boom)
    access.authorize(SHA, SHB)
    with pytest.raises(RuntimeError):
        access.consume_and_load_rows()
    assert access.state() == ta.FINAL_TEST_ACCESS_CONSUMED
    assert any(e.get("event") == "LOADER_EXCEPTION" for e in access.history())
    with pytest.raises(SealedCohortAccessError):
        access.consume_and_load_rows()  # no silent retry


def test_missing_expected_paths_refused_before_consumption(tmp_path):
    access, _ = _make(tmp_path)
    (tmp_path / "cohort/rows.jsonl").unlink()
    access.authorize(SHA, SHB)
    with pytest.raises(SealedCohortAccessError, match="missing"):
        access.consume_and_load_rows()
    assert access.state() == ta.AUTHORIZED_FOR_ONE_FINAL_RUN


def test_initialize_refuses_overwrite_and_lock(tmp_path):
    access, _ = _make(tmp_path)
    with pytest.raises(SealedCohortAccessError):
        access.initialize()
    lock = tmp_path / "gov/state.json.lock"
    lock.write_text("x")
    with pytest.raises(SealedCohortAccessError, match="lock"):
        access.authorize(SHA, SHB)
    lock.unlink()


def test_assert_not_sealed_blocks_ids_and_paths(tmp_path):
    _make(tmp_path)
    with pytest.raises(SealedCohortAccessError):
        assert_not_sealed("SYN-V3-T")
    with pytest.raises(SealedCohortAccessError):
        assert_not_sealed(tmp_path / "cohort" / "rows.jsonl")
    with pytest.raises(SealedCohortAccessError):
        assert_not_sealed(str(tmp_path / "cohort"))
    assert_not_sealed("SYN-V3-D1")
    assert_not_sealed(tmp_path / "other" / "x.jsonl")
