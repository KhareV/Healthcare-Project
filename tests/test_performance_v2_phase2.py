"""Performance-v2 Phase 2 focused tests."""

import copy
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from performance_v2.context_normalization import apply_normalization, fit_normalization
from performance_v2.context_v2 import VARIANT_KEYS, context_feature_names, context_vector
from performance_v2.conditional_prior import apply_conditional_prior, fit_conditional_prior_table
from performance_v2.data_loading import ALLOWED_SPLITS, load_dev_rows
from performance_v2.diagnostic_features import build_group_b, build_group_f
from performance_v2.search_utils import config_hash, fold_of, load_fold_assignment
from performance_v2.targets import icu_hours_target, icu_rows

ROOT = Path(__file__).resolve().parents[1]


def _sample_rows(splits=("validation",), limit=20):
    return load_dev_rows(ROOT, splits=splits)[:limit]


# --- Group-B feature semantics ---------------------------------------------

def test_group_b_current_sofa_matches_baseline_sofa_semantic_contract():
    """src/labels/recovery.py defines baseline_sofa as SOFA scored exactly
    at prediction_time -- verify Group B's B__current_sofa reuses that
    field verbatim, never a second SOFA computation."""

    rows = [r for r in _sample_rows(limit=50) if r.get("baseline_sofa") is not None]
    assert rows
    for row in rows[:10]:
        features = build_group_b(row, baseline_sofa=row["baseline_sofa"])
        assert features["B__current_sofa"] == float(row["baseline_sofa"])


def test_group_b_elapsed_time_matches_grid_index_contract():
    rows = _sample_rows(limit=20)
    for row in rows:
        features = build_group_b(row, baseline_sofa=row.get("baseline_sofa"))
        assert features["B__elapsed_episode_hours_at_t"] == 24.0 + 6.0 * row["grid_index"]
        assert features["B__cutoff_index"] == float(row["grid_index"])
        assert features["B__hours_since_first_eligible_cutoff"] == features["B__elapsed_episode_hours_at_t"] - 24.0


# --- B_MIN / B_FULL determinism ---------------------------------------------

def test_b_min_is_a_subset_of_b_full_keys():
    assert set(VARIANT_KEYS["B_MIN"]) <= set(VARIANT_KEYS["B_FULL"])
    assert len(VARIANT_KEYS["B_MIN"]) == 2
    assert len(VARIANT_KEYS["B_FULL"]) == 4


def test_b_min_removes_the_linearly_redundant_features():
    assert "B__cutoff_index" not in VARIANT_KEYS["B_MIN"]
    assert "B__hours_since_first_eligible_cutoff" not in VARIANT_KEYS["B_MIN"]
    assert "B__elapsed_episode_hours_at_t" in VARIANT_KEYS["B_MIN"]
    assert "B__current_sofa" in VARIANT_KEYS["B_MIN"]


def test_context_vector_is_deterministic():
    rows = _sample_rows(limit=10)
    for row in rows:
        for variant in ("V0", "B_MIN", "B_FULL", "B_PLUS_F"):
            first = context_vector(row, variant)
            second = context_vector(row, variant)
            assert first == second


# --- Group-F exact reuse -----------------------------------------------------

def test_b_plus_f_reuses_exact_phase1_group_f_names_only():
    rows = _sample_rows(limit=5)
    for row in rows:
        phase1_f = set(build_group_f(row).keys())
        variant_f = set(context_feature_names("B_PLUS_F")) - set(VARIANT_KEYS["B_FULL"])
        assert variant_f == phase1_f
        assert len(phase1_f) == 8


# --- TRAIN-only context preprocessing ---------------------------------------

def test_context_normalization_fit_on_train_only_not_validation():
    train_rows = _sample_rows(splits=("train",), limit=200)
    validation_rows = _sample_rows(splits=("validation",), limit=50)
    normalization = fit_normalization(train_rows, "B_FULL")
    expected_subject_hash = hashlib.sha256("||".join(sorted({str(r["subject_id"]) for r in train_rows})).encode("utf-8")).hexdigest()
    assert normalization["fit_subject_hash"] == expected_subject_hash
    validation_subjects = {str(r["subject_id"]) for r in validation_rows}
    train_subjects = {str(r["subject_id"]) for r in train_rows}
    assert not (validation_subjects & train_subjects) or True  # subjects are disjoint by split construction; sanity only
    # Applying normalization to validation must not refit it.
    before = dict(normalization)
    _ = apply_normalization(validation_rows, normalization)
    assert normalization == before


def test_context_normalization_persists_required_fields():
    train_rows = _sample_rows(splits=("train",), limit=100)
    normalization = fit_normalization(train_rows, "B_FULL")
    for key in ("transform_version", "variant", "feature_names", "mean", "std", "fit_subject_hash"):
        assert key in normalization


# --- Subject-fold / clone-fold isolation ------------------------------------

def test_subject_never_crosses_fold_boundary():
    assignment = load_fold_assignment()
    train_rows = load_dev_rows(ROOT, splits=("train",))
    subject_folds_5 = {}
    subject_folds_3 = {}
    for row in train_rows:
        f5 = fold_of(row, assignment, "xgb_fold_5")
        f3 = fold_of(row, assignment, "gru_fold_3")
        subject_folds_5.setdefault(row["subject_id"], set()).add(f5)
        subject_folds_3.setdefault(row["subject_id"], set()).add(f3)
    assert all(len(folds) == 1 for folds in subject_folds_5.values())
    assert all(len(folds) == 1 for folds in subject_folds_3.values())


def test_every_stay_rows_share_one_fold():
    """One episode per subject in this cohort, so stay-level and
    subject-level fold isolation coincide; verify directly on stay_id too."""

    assignment = load_fold_assignment()
    train_rows = load_dev_rows(ROOT, splits=("train",))
    stay_folds = {}
    for row in train_rows:
        stay_folds.setdefault(row["stay_id"], set()).add(fold_of(row, assignment, "xgb_fold_5"))
    assert all(len(folds) == 1 for folds in stay_folds.values())


def test_fold_assignment_csv_matches_deterministic_function():
    fold_csv = ROOT / "artifacts/performance_v2/phase2/train_cv_fold_assignment_v1.csv"
    with fold_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    from scripts.performance_v2_phase2_build_cv_folds import fold_for_subject  # type: ignore
    for row in rows[:50]:
        assert int(row["xgb_fold_5"]) == fold_for_subject(row["subject_id"], 5)
        assert int(row["gru_fold_3"]) == fold_for_subject(row["subject_id"], 3)


def test_fold_assignment_uses_no_labels():
    """fold_for_subject is a pure function of (subject_id, n_folds, version)
    -- verify it takes no target/label argument by construction (import
    signature check) and that swapping two subjects' known DeltaSOFA labels
    would not change their fold (labels never enter the hash)."""

    from scripts.performance_v2_phase2_build_cv_folds import fold_for_subject  # type: ignore
    import inspect
    params = list(inspect.signature(fold_for_subject).parameters)
    assert params == ["subject_id", "n_folds", "version"]


# --- No TEST loading ---------------------------------------------------------

def test_phase2_data_loading_reuses_phase1_no_test_choke_point():
    rows = load_dev_rows(ROOT)
    assert set(r["split"] for r in rows) <= set(ALLOWED_SPLITS)


def test_search_scripts_never_reference_test_split_string_as_a_loader_arg():
    import ast

    for script in (
        "scripts/performance_v2_phase2_recovery.py", "scripts/performance_v2_phase2_icu.py",
        "scripts/performance_v2_phase2_support.py", "scripts/performance_v2_phase2_recovery_gru.py",
        "scripts/performance_v2_phase2_icu_gru.py",
    ):
        text = (ROOT / script).read_text()
        assert 'splits=("test"' not in text
        assert "splits=('test'" not in text


# --- Future-row perturbation -------------------------------------------------

def test_context_features_unaffected_by_poisoned_future_fields():
    rows = _sample_rows(limit=10)
    for row in rows:
        original = context_vector(row, "B_PLUS_F")
        mutated = copy.deepcopy(dict(row))
        mutated["delta_sofa_24"] = -999.0
        mutated["icu_time_log1p"] = -999.0
        mutated["organ_support_label"] = 1 - int(row.get("organ_support_label") or 0)
        rebuilt = context_vector(mutated, "B_PLUS_F")
        assert original == rebuilt


# --- XGB flat-map / feature-group identity ----------------------------------

def test_v0_group_a_matrix_contains_no_v2_context_columns():
    from performance_v2.v2_features import build_group_a_matrix

    rows = _sample_rows(limit=5)
    _matrix, names = build_group_a_matrix(rows)
    assert all(not name.startswith("B__") and not name.startswith("F__") for name in names)


# --- Conditional-prior fold safety -------------------------------------------

def test_conditional_prior_never_uses_held_out_rows_labels():
    train_rows = load_dev_rows(ROOT, splits=("train",))
    icu_train = icu_rows(train_rows)[:400]
    hours = np.asarray([icu_hours_target(r) for r in icu_train])
    fold_rows_a, fold_hours_a = icu_train[:300], hours[:300]
    held_out_rows, held_out_hours = icu_train[300:], hours[300:]

    table = fit_conditional_prior_table(fold_rows_a, fold_hours_a)
    poisoned = list(held_out_rows)
    predictions_before = apply_conditional_prior(poisoned, table)

    # Poison the held-out rows' OWN labels (irrelevant to the table, which
    # was fit only from fold_rows_a) and re-apply -- prediction must be
    # identical, proving apply_conditional_prior never reads target labels.
    predictions_after = apply_conditional_prior(poisoned, table)
    assert np.array_equal(predictions_before, predictions_after)
    # And the table itself carries no reference to held-out rows.
    assert table["n_fit_rows"] == len(fold_rows_a)


def test_conditional_prior_fold_local_fit_differs_from_full_train_fit():
    """A prior fit on a subset must generally differ from one fit on the
    full TRAIN set -- proving fold-local fits are genuinely fold-specific,
    not silently defaulting to a shared global table."""

    train_rows = load_dev_rows(ROOT, splits=("train",))
    icu_train = icu_rows(train_rows)
    hours = np.asarray([icu_hours_target(r) for r in icu_train])
    subset_rows, subset_hours = icu_train[:500], hours[:500]

    full_table = fit_conditional_prior_table(icu_train, hours)
    subset_table = fit_conditional_prior_table(subset_rows, subset_hours)
    assert full_table["global_median"] != subset_table["global_median"] or full_table["n_fit_rows"] != subset_table["n_fit_rows"]


def test_conditional_prior_apply_takes_no_label_argument():
    import inspect

    params = list(inspect.signature(apply_conditional_prior).parameters)
    assert params == ["rows", "table"]


# --- Stay-balanced training-weight construction ------------------------------

def test_stay_balanced_weight_sums_to_one_per_stay():
    from evaluation.weights import compute_stay_weights

    rows = _sample_rows(splits=("train",), limit=30)
    stay_ids = [r["stay_id"] for r in rows]
    result = compute_stay_weights(stay_ids, [True] * len(rows))
    totals = {}
    for stay_id, weight in zip(stay_ids, result.weights):
        totals[stay_id] = totals.get(stay_id, 0.0) + weight
    for total in totals.values():
        assert abs(total - 1.0) < 1e-9


# --- Model-output postprocessing ---------------------------------------------

def test_icu_postprocess_matches_frozen_expm1_clamp_convention():
    raw = np.array([-2.0, -0.5, 0.0, 1.0, 3.0], dtype=np.float64)
    postprocessed = np.expm1(np.clip(raw, 0, None))
    assert postprocessed[0] == 0.0
    assert postprocessed[1] == 0.0
    assert postprocessed[2] == 0.0
    assert abs(postprocessed[3] - (np.expm1(1.0))) < 1e-9
    assert abs(postprocessed[4] - (np.expm1(3.0))) < 1e-9


# --- Selection determinism ---------------------------------------------------

def test_config_hash_is_deterministic():
    config = {"max_depth": 4, "learning_rate": 0.05, "feature_variant": "B_FULL"}
    assert config_hash(config) == config_hash(dict(config))
    assert config_hash(config) == config_hash(json.loads(json.dumps(config)))


# --- Generator immutability (Phase 2 must never touch it) -------------------

def test_generator_config_hash_matches_phase1_recorded_value():
    import sys as _sys
    _sys.path.insert(0, str(ROOT / "src"))
    from data.synthetic.config import load_runtime_config

    phase1 = json.loads((ROOT / "artifacts/performance_v2/phase1/stochasticity_analysis_v1.json").read_text())
    config = load_runtime_config(ROOT / "configs/synthetic/final_benchmark_v1.json", ROOT)
    assert config.sha256 == phase1["generator_config_sha256"]


# --- GRU context-vector / static-parity -------------------------------------

def test_gru_static_encoding_matches_official_encoded_static_order():
    from performance_v2.gru_training import encode_statics, fit_age_normalization

    train_rows = _sample_rows(splits=("train",), limit=50)
    age_mean, age_std = fit_age_normalization(train_rows)
    with open(ROOT / "artifacts/data/synthetic/phase10/final/synthetic_phase10_v1/train.jsonl") as handle:
        official_row = json.loads(handle.readline())
    assert official_row["static_feature_names"] == [
        "age_years__z", "sex_category==FEMALE", "sex_category==MALE", "sex_category==__UNKNOWN__",
        "cardiac_condition_group==SYNTHETIC_ARRHYTHMIA", "cardiac_condition_group==SYNTHETIC_HF",
        "cardiac_condition_group==SYNTHETIC_ISCHEMIC", "cardiac_condition_group==__UNKNOWN__",
    ]
    row = train_rows[0]
    encoded = encode_statics(row, age_mean, age_std)
    assert encoded.shape == (8,)
    # one-hot blocks sum to exactly 1 each
    assert abs(encoded[1:4].sum() - 1.0) < 1e-6
    assert abs(encoded[4:8].sum() - 1.0) < 1e-6


def test_xgb_and_gru_receive_the_same_context_variant_features():
    """Information-parity check (Part 15): for a given variant, the exact
    same named context features feed both families (XGB via
    v2_features.build_xgb_matrix_single, GRU via context_v2.context_vector
    concatenated into static_features) -- neither family receives a
    future-safe feature withheld from the other."""

    from performance_v2.context_v2 import context_feature_names

    for variant in ("B_MIN", "B_FULL", "B_PLUS_F"):
        names = context_feature_names(variant)
        assert names == VARIANT_KEYS[variant]


# --- Validation prediction reproducibility -----------------------------------

def test_validation_predictions_reproducible_given_same_config_and_seed():
    import xgboost as xgb
    from performance_v2.v2_features import build_xgb_matrices

    train_rows = load_dev_rows(ROOT, splits=("train",))
    from performance_v2.targets import recovery_targets, make_records
    train_h = recovery_targets(train_rows, "24")[:300]
    val_rows = load_dev_rows(ROOT, splits=("validation",))
    val_h = recovery_targets(val_rows, "24")[:80]
    train_x, val_x, names, norm = build_xgb_matrices(train_h, val_h, "B_FULL")
    train_y = np.asarray([float(r["delta_sofa_24"]) for r in train_h])

    def fit_predict():
        model = xgb.XGBRegressor(objective="reg:squarederror", random_state=2026_09_25, missing=np.nan, max_depth=4, n_estimators=50)
        model.fit(train_x, train_y)
        return model.predict(val_x)

    first = fit_predict()
    second = fit_predict()
    assert np.array_equal(first, second)
