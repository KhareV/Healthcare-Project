import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest

from performance_v3_1 import diagnostic as D
from performance_v3_1.replay import QUALIFYING_VASO_EXCLUDED, continuation_label, replay_stay

ROOT = Path(__file__).resolve().parents[2]
HAVE_DATA = (ROOT / "artifacts/performance_v3_1/cohorts/DIAG_EVAL/raw/support_intervals.jsonl").exists()
needs_data = pytest.mark.skipif(not HAVE_DATA, reason="diagnostic cohort (gitignored artifact) absent")


@pytest.fixture(scope="module")
def ctx():
    return D.CohortContext(ROOT, "DIAG_EVAL")


@needs_data
def test_replay_reproduces_generator_support_rows_bitwise(ctx):
    out = D.verify_replay(ctx, max_stays=120)
    assert out["bitwise_identical"] and out["stays_checked"] == 120


@needs_data
def test_snapshot_features_invariant_to_truncation_and_free_of_forbidden_tokens(ctx):
    stay, cand = next(iter(ctx.by_stay.items()))
    intime, outtime = ctx.intime_outtime(stay)
    hours = int(math.ceil((outtime - intime).total_seconds() / 3600))
    h = 24
    _, _, full, _ = replay_stay(ctx.config, ctx.contract, cand, intime, hours, snapshot_hours={h})
    _, _, trunc, _ = replay_stay(ctx.config, ctx.contract, cand, intime, h + 1, snapshot_hours={h})
    a, b = D._state_features(full[h], cand, h), D._state_features(trunc[h], cand, h)
    assert a == b
    assert D.forbidden_token_scan(D.A_NAMES + D.B_NAMES) == []


@needs_data
def test_continuation_is_deterministic_and_binary(ctx):
    stay, cand = next(iter(ctx.by_stay.items()))
    intime, outtime = ctx.intime_outtime(stay)
    hours = int(math.ceil((outtime - intime).total_seconds() / 3600))
    _, _, snaps, _ = replay_stay(ctx.config, ctx.contract, cand, intime, hours, snapshot_hours={24})
    a = continuation_label(snaps[24], ctx.config, ctx.contract, np.random.default_rng(5), 80.0)
    b = continuation_label(snaps[24], ctx.config, ctx.contract, np.random.default_rng(5), 80.0)
    assert a == b and a in ((0, 0), (1, 0), (0, 1))


@needs_data
def test_oracle_features_exclude_realized_outcome_fields(ctx):
    forbidden = {"duration", "outtime", "label", "organ_support_eligible", "recovery24_eligible"}
    assert not (set(D.B_NAMES) & forbidden)


def test_dobutamine_is_non_qualifying():
    assert QUALIFYING_VASO_EXCLUDED == {"dobutamine"}


def test_handoff_retains_frozen_v3_and_creates_no_new_final_test():
    h = json.loads((ROOT / "artifacts/performance_v3_1/product_handoff.json").read_text())
    assert h["outcome"] == "V3_RETAINED_AFTER_NEGATIVE_V3_1_DEVELOPMENT" and h["new_final_tests"] == "NONE_CREATED"
    p = ROOT / "artifacts/performance_v3/phase3/selected_models_v3.json"
    assert hashlib.sha256(p.read_bytes()).hexdigest() == h["effective_release"]["manifest_sha256"] == "af9014a33d0a219a3f64b713c6812add46b07f44efcb4d212674ea47b7011961"
    assert not (ROOT / "artifacts/performance_v3_1/cohorts/V31_T").exists()
    state = json.loads((ROOT / "artifacts/performance_v3/governance/v3_final_test_access_state.json").read_text())
    assert state["status"] == "FINAL_RUN_COMPLETED"


def test_frozen_science_inventory_unchanged_except_listed_exceptions():
    inv = ROOT / "artifacts/performance_v3_1/governance/pre_v3_1_hash_inventory_v1.sha256"
    allowed = {"src/data/synthetic/config.py", "docs/evidence/system/system_evidence_manifest_v1.json"}
    bad = []
    for line in inv.read_text().splitlines():
        digest, path = line.split("  ", 1)
        f = ROOT / path
        if path in allowed:
            continue
        if not f.exists() or hashlib.sha256(f.read_bytes()).hexdigest() != digest:
            bad.append(path)
    # files under tests/ and scripts/ may legitimately gain siblings but inventoried files must be intact
    assert not bad, bad[:5]
