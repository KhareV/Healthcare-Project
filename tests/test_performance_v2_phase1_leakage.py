"""Performance-v2 Phase 1: feature-leakage adversarial tests (Part 5) and
core governance invariants (v1 immutability, no-TEST-access, cutoff safety).
"""

import copy
from pathlib import Path

import pytest

from performance_v2.data_loading import ALLOWED_SPLITS, assert_no_test_rows, load_dev_rows
from performance_v2.diagnostic_features import Phase1FeatureError, build_all_groups, feature_group_of, GROUP_ORDER

ROOT = Path(__file__).resolve().parents[1]

PROHIBITED_TOKENS = (
    "outtime", "total_episode_duration", "remaining_stay", "future_sofa",
    "delta_sofa", "support_label", "future_support", "censor_reason",
    "split", "test_identity", "latent_state", "trajectory", "outcome_class",
    "final_episode_state", "full_stay_aggregate",
)


def _sample_rows(splits=("validation",), limit=25):
    rows = load_dev_rows(ROOT, splits=splits)
    return rows[:limit]


def test_load_dev_rows_only_ever_returns_train_or_validation():
    rows = load_dev_rows(ROOT)
    assert set(row["split"] for row in rows) <= set(ALLOWED_SPLITS)
    assert_no_test_rows(rows)


def test_load_dev_rows_refuses_a_requested_test_split():
    with pytest.raises(ValueError):
        load_dev_rows(ROOT, splits=("test",))


def test_assert_no_test_rows_detects_a_smuggled_test_row():
    rows = list(load_dev_rows(ROOT, splits=("validation",))[:3])
    tampered = [dict(rows[0])]
    tampered[0]["split"] = "test"
    with pytest.raises(RuntimeError):
        assert_no_test_rows(tampered)


def test_elapsed_time_formula_matches_grid_index_contract():
    rows = _sample_rows()
    for row in rows:
        expected = 24.0 + 6.0 * row["grid_index"]
        assert abs(row["elapsed_episode_hours_at_t"] - expected) < 1e-6
        assert row["cutoff_index"] == row["grid_index"]
        assert abs(row["hours_since_first_eligible_cutoff"] - (expected - 24.0)) < 1e-6


def test_diagnostic_feature_names_carry_no_prohibited_token():
    rows = _sample_rows()
    for row in rows:
        features = build_all_groups(row, baseline_sofa=row.get("baseline_sofa"), sofa_trend={})
        lowered_names = [name.lower() for name in features]
        for token in PROHIBITED_TOKENS:
            assert not any(token in name for name in lowered_names), token


def test_every_feature_belongs_to_a_declared_group():
    rows = _sample_rows(limit=5)
    for row in rows:
        features = build_all_groups(row, baseline_sofa=row.get("baseline_sofa"), sofa_trend={})
        for name in features:
            assert feature_group_of(name) in GROUP_ORDER


def test_appending_an_illegal_extra_bin_is_rejected_not_silently_misread():
    """Part 5: an extra, illegal 9th bin appended after the frozen 8-bin
    (t-48h, t] window must be rejected fail-closed, not silently read as if
    it were the true last-observed value (which would let a future/
    out-of-window row masquerade as "last observed at t")."""

    rows = _sample_rows(limit=5)
    for row in rows:
        mutated = copy.deepcopy(dict(row))
        n_channels = len(mutated["temporal_feature_names"])
        mutated["history_values"] = list(mutated["history_values"]) + [[999999.0] * n_channels]
        mutated["observation_mask"] = list(mutated["observation_mask"]) + [[True] * n_channels]
        mutated["tslo_hours"] = list(mutated["tslo_hours"]) + [[0.0] * n_channels]
        with pytest.raises(Phase1FeatureError):
            build_all_groups(mutated, baseline_sofa=row.get("baseline_sofa"), sofa_trend={})


def test_poisoned_future_outcome_fields_never_change_diagnostic_features():
    """Part 5: mutating any (illegally present) future-outcome field must
    never change a single diagnostic feature, since the builder must never
    read delta_sofa_24/48, icu_time_log1p, or organ_support_label at all."""

    rows = _sample_rows(limit=10)
    for row in rows:
        original = build_all_groups(row, baseline_sofa=row.get("baseline_sofa"), sofa_trend={})

        mutated = copy.deepcopy(dict(row))
        mutated["delta_sofa_24"] = -999.0
        mutated["delta_sofa_48"] = 999.0
        mutated["icu_time_log1p"] = -999.0
        mutated["organ_support_label"] = 1 - int(row.get("organ_support_label") or 0)

        rebuilt = build_all_groups(mutated, baseline_sofa=row.get("baseline_sofa"), sofa_trend={})
        for name in original:
            assert name in rebuilt
            left, right = original[name], rebuilt[name]
            if left != left and right != right:  # both NaN
                continue
            assert left == right, name


def test_group_a_reflects_only_the_frozen_eight_bins():
    rows = _sample_rows(limit=5)
    for row in rows:
        features = build_all_groups(row, baseline_sofa=row.get("baseline_sofa"), sofa_trend={})
        bin_indices = {int(name.split("__")[1][3:]) for name in features if name.startswith("A__bin")}
        assert bin_indices == set(range(8))
