import json
from datetime import datetime, timedelta, timezone

import pytest

from data.real_adapter import (
    RealDataContractError,
    load_real_bundle,
    validate_support_evidence,
)
from real_data_helpers import build_real_handoff_fixture, refresh_manifest_hash, write_json


def _attack(root, field, value):
    path = root / "artifacts/data/feature_dataset_v1.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    row = next(item for item in payload["rows"] if item["rec48_ok"])
    row[field] = value
    write_json(path, payload)
    refresh_manifest_hash(root, "canonical_dataset")


def test_recovery_horizons_are_checked_against_same_baseline(tmp_path):
    manifest = build_real_handoff_fixture(tmp_path)
    _attack(tmp_path, "delta48", 1.0)
    with pytest.raises(RealDataContractError, match="baseline-relative"):
        load_real_bundle(tmp_path, manifest)


def test_recovery24_formula_attack_fails(tmp_path):
    manifest = build_real_handoff_fixture(tmp_path)
    _attack(tmp_path, "delta24", -3.0)
    with pytest.raises(RealDataContractError, match="recovery 24h formula"):
        load_real_bundle(tmp_path, manifest)


def test_icu_remaining_time_formula_attack_fails(tmp_path):
    manifest = build_real_handoff_fixture(tmp_path)
    _attack(tmp_path, "remaining_h", 999.0)
    with pytest.raises(RealDataContractError, match="ICU remaining-time"):
        load_real_bundle(tmp_path, manifest)


def test_support_positive_before_early_exit_remains_positive():
    cutoff = datetime(2026, 1, 1, tzinfo=timezone.utc)
    row = {
        "organ_support_eligible": True,
        "organ_support_target": 1,
        "vasopressor_on_at_t": False,
        "ventilation_on_at_t": True,
        "vasopressor_onset_time": (cutoff + timedelta(hours=2)).isoformat(),
        "ventilation_onset_time": None,
    }
    assert validate_support_evidence(row, cutoff, cutoff + timedelta(hours=4)) == (True, 1)


def test_support_already_on_continuation_is_not_new_initiation():
    cutoff = datetime(2026, 1, 1, tzinfo=timezone.utc)
    row = {
        "organ_support_eligible": False,
        "organ_support_target": None,
        "vasopressor_on_at_t": True,
        "ventilation_on_at_t": True,
        "vasopressor_onset_time": (cutoff + timedelta(hours=2)).isoformat(),
        "ventilation_onset_time": None,
    }
    assert validate_support_evidence(row, cutoff, cutoff + timedelta(hours=30)) == (False, None)
