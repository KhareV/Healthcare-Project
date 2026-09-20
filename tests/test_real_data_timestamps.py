import json
from datetime import datetime, timedelta

import pytest

from data.real_adapter import RealDataContractError, load_real_bundle
from real_data_helpers import build_real_handoff_fixture, refresh_manifest_hash, write_json


def test_thirteenth_or_post_cap_timestamp_is_blocked(tmp_path):
    manifest = build_real_handoff_fixture(tmp_path)
    path = tmp_path / "artifacts/data/canonical_structural_index_v1.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    extra = dict(payload["rows"][0])
    extra["k"] = 12
    extra["cutoff"] = (
        datetime.fromisoformat(extra["icu_in"]) + timedelta(hours=96)
    ).isoformat()
    payload["rows"].append(extra)
    write_json(path, payload)
    refresh_manifest_hash(tmp_path, "structural_index")
    with pytest.raises(RealDataContractError, match="TIMESTAMP CONTRACT MISMATCH"):
        load_real_bundle(tmp_path, manifest)


def test_outtime_minus_six_equality_is_legal(tmp_path):
    manifest = build_real_handoff_fixture(tmp_path)
    bundle = load_real_bundle(tmp_path, manifest)
    last = max(
        (row for row in bundle.structural_rows if row.stay_id == "STAY-A"),
        key=lambda row: row.prediction_time,
    )
    assert last.prediction_time == last.outtime - timedelta(hours=6)


def test_existing_row_after_outtime_minus_six_is_blocked(tmp_path):
    manifest = build_real_handoff_fixture(tmp_path)
    path = tmp_path / "artifacts/data/canonical_structural_index_v1.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    last = next(
        row for row in payload["rows"]
        if row["icu"] == "STAY-A" and row["k"] == 8
    )
    last["cutoff"] = (
        datetime.fromisoformat(last["icu_out"]) - timedelta(hours=5)
    ).isoformat()
    write_json(path, payload)
    refresh_manifest_hash(tmp_path, "structural_index")
    with pytest.raises(RealDataContractError, match="TIMESTAMP CONTRACT MISMATCH"):
        load_real_bundle(tmp_path, manifest)
