"""Tests for the V3 learning-curve library (synthetic data; never reads a real cohort)."""

from pathlib import Path

import numpy as np
import pytest

from performance_v3 import features as v3feat
from performance_v3 import learning_curve as lc

ROOT = Path(__file__).resolve().parents[1]


def _ids(prefix, n):
    return [f"{prefix}-S-{i:05d}" for i in range(n)]


V1 = _ids("SYN", 1700)
POOL = V1 + _ids("SYN-V3-D1", 2100) + _ids("SYN-V3-D2", 2100) + _ids("SYN-V3-D3", 2100)


def test_subsets_nested_within_repeat_and_small_sizes_from_v1_only():
    subsets = lc.training_subsets(V1, POOL, 0)
    ordered = [subsets[s] for s in lc.SIZES]
    for small, large in zip(ordered[:3], ordered[1:3]):
        assert large[: len(small)] == small
    for size in (425, 850, 1700):
        assert set(subsets[size]) <= set(V1)
    for small, large in zip(ordered[3:], ordered[4:]):
        assert large[: len(small)] == small
    assert set(subsets[8000]) == set(POOL)
    assert len(subsets[3800]) == 3800 and len(set(subsets[3800])) == 3800


def test_repeats_independent_but_deterministic():
    a0, a0b = lc.training_subsets(V1, POOL, 0), lc.training_subsets(V1, POOL, 0)
    a1 = lc.training_subsets(V1, POOL, 1)
    assert a0 == a0b
    assert a0[425] != a1[425]
    assert a0[3800] != a1[3800]
    assert set(a0[1700]) == set(a1[1700]) == set(V1)


def test_disjointness_assertion_refuses_overlap():
    lc.assert_disjoint(POOL[:10], _ids("SYN-V3-LCE", 5))
    with pytest.raises(lc.LearningCurveError):
        lc.assert_disjoint(POOL[:10], [POOL[3]])


def test_trend_labels_follow_protocol_thresholds():
    # lower-is-better: 4% per doubling -> strong; 1% -> marginal; 0.1% -> plateau; -2% -> degrading
    sizes = [1700, 3400, 6800]
    for rate, label in ((0.04, "strong"), (0.01, "marginal"), (0.001, "plateau"), (-0.02, "degrading")):
        vals = [1.0 * (1 - rate) ** i for i in range(3)]
        assert lc.classify_trend("recovery24", sizes, vals)["overall_from_1700"]["label"] == label
    vals_up = [0.6 * (1.03) ** i for i in range(3)]
    assert lc.classify_trend("organ_support", sizes, vals_up)["overall_from_1700"]["label"] == "strong"


def test_v2_recipes_match_frozen_selection():
    recipes = lc.load_v2_recipes(ROOT)
    assert {t: r["variant"] for t, r in recipes.items()} == {
        "recovery24": "B_MIN", "recovery48": "B_MIN", "icu_stay_time": "B_PLUS_F", "organ_support": "B_FULL"}
    assert {t: r["n_estimators"] for t, r in recipes.items()} == {
        "recovery24": 169, "recovery48": 316, "icu_stay_time": 428, "organ_support": 56}
    assert recipes["icu_stay_time"]["objective"] == "reg:absoluteerror"
    assert recipes["recovery24"]["weight_mode"] == "uniform" and recipes["recovery48"]["weight_mode"] == "stay_balanced"
    assert recipes["organ_support"]["kind"] == "classification"


def _toy_cache(n=60, seed=0):
    rng = np.random.default_rng(seed)
    names = ("A__bin0__hr__latest__value", "A__bin0__hr__latest__mask", "B__elapsed_episode_hours_at_t", "B__current_sofa")
    matrix = rng.normal(size=(n, len(names))).astype(np.float32)
    matrix[:, 1] = (matrix[:, 1] > 0)
    subj = np.asarray([f"S{i // 3}" for i in range(n)])
    meta = {
        "subject_id": subj, "stay_id": subj.copy(), "prediction_time": np.asarray([f"t{i}" for i in range(n)]),
        "cohort_id": np.asarray(["X"] * n), "grid_index": np.arange(n) % 3,
        "delta_sofa_24": rng.normal(size=n), "delta_sofa_48": rng.normal(size=n),
        "icu_time_log1p": rng.uniform(1, 4, size=n), "organ_support_label": (rng.uniform(size=n) > 0.5).astype(float),
        "baseline_sofa": rng.uniform(0, 10, size=n),
        "recovery24_eligible": np.ones(n, bool), "recovery48_eligible": np.ones(n, bool),
        "icu_time_eligible": np.ones(n, bool), "organ_support_eligible": np.ones(n, bool),
    }
    return v3feat.FeatureCache(names=names, matrix=matrix, meta=meta)


def test_design_matrix_row_order_follows_requested_index():
    cache = _toy_cache()
    idx = np.asarray([10, 3, 25, 7])
    norm = v3feat.fit_context_normalization(cache, "B_MIN", row_mask=np.isin(np.arange(cache.n_rows), idx))
    X = lc.design_matrix(cache, idx, "B_MIN", norm)
    assert X.shape == (4, 4)  # 2 Group-A columns + 2 normalised B_MIN context columns
    a_cols = lc.group_a_columns(cache.names)
    assert np.array_equal(X[:, : len(a_cols)], cache.matrix[idx][:, a_cols])
    # normalised context of row idx[i] must equal row-wise application, independent of order
    X_sorted = lc.design_matrix(cache, np.sort(idx), "B_MIN", norm)
    order = np.argsort(idx)
    assert np.allclose(X[order], X_sorted)


def test_train_and_evaluate_wiring_deterministic():
    cache = _toy_cache(n=120)
    recipes = {
        "recovery24": dict(task="recovery24", kind="regression", objective="reg:pseudohubererror", weight_mode="stay_balanced",
                           params=dict(max_depth=3, min_child_weight=1, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.0, reg_lambda=1.0, gamma=0.0),
                           n_estimators=10, variant="B_MIN"),
        "organ_support": dict(task="organ_support", kind="classification", objective=None, weight_mode="uniform",
                              params=dict(max_depth=3, min_child_weight=1, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.0, reg_lambda=1.0, gamma=0.0),
                              n_estimators=10, variant="B_MIN"),
        "icu_stay_time": dict(task="icu_stay_time", kind="regression", objective="reg:absoluteerror", weight_mode="stay_balanced",
                              params=dict(max_depth=3, min_child_weight=1, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.0, reg_lambda=1.0, gamma=0.0),
                              n_estimators=10, variant="B_MIN"),
    }
    train_idx = np.arange(0, 90)
    ev_idx = np.arange(90, 120)
    for task, recipe in recipes.items():
        m1, p1 = lc.fit_and_score(recipe, cache, train_idx, cache, ev_idx, seed=11, n_jobs=2)
        m2, p2 = lc.fit_and_score(recipe, cache, train_idx, cache, ev_idx, seed=11, n_jobs=2)
        assert np.allclose(p1, p2)
        assert m1[lc.PRIMARY_NAME[task]] == m2[lc.PRIMARY_NAME[task]]
        assert m1["n_stays"] == len(set(cache.meta["stay_id"][ev_idx]))
        assert m1["n_train_rows"] == 90


def test_eval_metrics_match_task_metrics_wrappers():
    from performance_v3.task_metrics import recovery_metrics

    y = np.asarray([0.0, 1.0, -1.0, 2.0]); pred = np.asarray([0.5, 0.5, 0.0, 1.0])
    stays = ["a", "a", "b", "c"]; times = ["1", "2", "1", "1"]
    got = lc.evaluate("recovery24", y, pred, stays, times)
    want = recovery_metrics(y, pred, stays, times, horizon="24h")
    assert got["mae"] == want["mae"] and got["n_stays"] == 3


def test_loaders_refuse_sealed_test_cohorts():
    from performance_v3 import data as v3data

    with pytest.raises(v3data.CohortAccessError):
        v3data.load_cohort_rows(ROOT, ROOT / "artifacts/performance_v3/cohorts/T", cohort_id="SYN-V3-T", role="dev")
    with pytest.raises(v3data.CohortAccessError):
        v3data.load_cohort_rows(ROOT, ROOT / "artifacts/performance_v3/cohorts/LCE", cohort_id="LCE", role="test")
