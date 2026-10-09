import numpy as np
import pytest

from performance_v3.models import (
    FAMILIES,
    HYPERPARAM_SPACES,
    ModelError,
    fit_predict,
    postprocess_icu,
    sample_family_configs,
)


def _toy(n=240, seed=3, classification=False):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 6))
    X[rng.random(X.shape) < 0.1] = np.nan
    base = np.nan_to_num(X[:, 0]) * 1.5 - np.nan_to_num(X[:, 1])
    if classification:
        y = (base + rng.normal(scale=0.5, size=n) > 0).astype(float)
    else:
        y = base + rng.normal(scale=0.3, size=n)
        y[:5] += 25.0  # outliers: separate squared vs absolute objectives
    w = rng.uniform(0.2, 1.5, size=n)
    return X, y, w


FAST = {
    "xgboost": {"max_depth": 3, "learning_rate": 0.2, "n_estimators": 25},
    "lightgbm": {"num_leaves": 8, "learning_rate": 0.2, "n_estimators": 25, "min_data_in_leaf": 5},
    "catboost": {"depth": 4, "learning_rate": 0.2, "n_estimators": 25},
    "ridge": {"alpha": 1.0},
    "logistic": {"C": 1.0},
}


@pytest.mark.parametrize("family", ["xgboost", "lightgbm", "catboost", "ridge"])
def test_regression_deterministic(family):
    X, y, w = _toy()
    cfg = dict(FAST[family], objective="squarederror", weight_mode="stay_balanced")
    p1, i1 = fit_predict(family, cfg, X[:180], y[:180], w[:180], X[180:], task="recovery24", nthread=2, seed=7)
    p2, _ = fit_predict(family, cfg, X[:180], y[:180], w[:180], X[180:], task="recovery24", nthread=2, seed=7)
    assert np.array_equal(p1, p2)
    assert p1.shape == (60,) and np.all(np.isfinite(p1))
    assert set(("best_iteration", "fit_seconds", "peak_rss_mb")) <= set(i1)
    assert i1["fit_seconds"] >= 0 and i1["peak_rss_mb"] > 0


@pytest.mark.parametrize("family", ["xgboost", "lightgbm", "catboost", "logistic"])
def test_classification_probabilities(family):
    X, y, w = _toy(classification=True)
    cfg = dict(FAST[family], objective="binary", weight_mode="none", class_weight="scale_pos_weight")
    p, _ = fit_predict(family, cfg, X[:180], y[:180], w[:180], X[180:], task="organ_support", nthread=2, seed=1)
    assert np.all((p >= 0) & (p <= 1))
    assert np.mean((p > 0.5) == (y[180:] > 0.5)) > 0.7


def test_weight_mode_wiring():
    X, y, w = _toy()
    base = dict(FAST["xgboost"], objective="squarederror")
    p_none_w, _ = fit_predict("xgboost", dict(base, weight_mode="none"), X[:180], y[:180], w[:180], X[180:], task="recovery24", nthread=1)
    p_none_nw, _ = fit_predict("xgboost", dict(base, weight_mode="none"), X[:180], y[:180], None, X[180:], task="recovery24", nthread=1)
    p_bal, _ = fit_predict("xgboost", dict(base, weight_mode="stay_balanced"), X[:180], y[:180], w[:180], X[180:], task="recovery24", nthread=1)
    assert np.array_equal(p_none_w, p_none_nw)  # weight_mode none ignores supplied weights
    assert not np.array_equal(p_none_w, p_bal)


@pytest.mark.parametrize("family", ["xgboost", "lightgbm", "catboost"])
def test_objective_wiring(family):
    X, y, w = _toy()
    outs = {}
    for objective in ("squarederror", "absoluteerror", "pseudohubererror"):
        cfg = dict(FAST[family], objective=objective, weight_mode="none")
        outs[objective], _ = fit_predict(family, cfg, X[:180], y[:180], w[:180], X[180:], task="icu_stay_time", nthread=1, seed=2)
    assert not np.array_equal(outs["squarederror"], outs["absoluteerror"])
    assert not np.array_equal(outs["squarederror"], outs["pseudohubererror"])


def test_early_stopping_uses_inner_split():
    X, y, w = _toy()
    cfg = dict(FAST["xgboost"], n_estimators=400, learning_rate=0.3, objective="squarederror", weight_mode="none", early_stopping_rounds=5)
    _, info = fit_predict("xgboost", cfg, X[:150], y[:150], w[:150], X[180:], task="recovery48", es=(X[150:180], y[150:180], w[150:180]), nthread=1)
    assert 1 <= info["best_iteration"] < 400


def test_invalid_combinations():
    X, y, w = _toy()
    with pytest.raises(ModelError):
        fit_predict("nope", {}, X, y, w, X, task="recovery24")
    with pytest.raises(ModelError):
        fit_predict("ridge", {}, X, y, w, X, task="organ_support")
    with pytest.raises(ModelError):
        fit_predict("logistic", {}, X, y, w, X, task="recovery24")


def test_postprocess_icu():
    out = postprocess_icu([-1.0, 0.0, np.log1p(10.0)])
    assert np.allclose(out, [0.0, 0.0, 10.0])


def test_sample_family_configs_deterministic_and_axes():
    a = sample_family_configs("xgboost", 12, 5, "icu_stay_time")
    b = sample_family_configs("xgboost", 12, 5, "icu_stay_time")
    assert a == b and len({c["config_hash"] for c in a}) == 12
    assert {c["objective"] for c in a} == {"squarederror", "pseudohubererror", "absoluteerror"}
    assert {c["weight_mode"] for c in a} == {"none", "stay_balanced"}
    assert all(3 <= c["max_depth"] <= 8 and 0.01 <= c["learning_rate"] <= 0.2 and c["n_estimators"] == 1500 for c in a)
    s = sample_family_configs("lightgbm", 6, 1, "organ_support")
    assert {c["class_weight"] for c in s} == {"none", "scale_pos_weight"} and all(c["objective"] == "binary" for c in s)
    assert set(HYPERPARAM_SPACES) == set(FAMILIES)
