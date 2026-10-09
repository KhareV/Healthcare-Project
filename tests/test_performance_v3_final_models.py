import json
from pathlib import Path

import numpy as np
import pytest

from performance_v3 import final_models as fm
from performance_v3 import models as M


def toy(task="recovery24", n=400, d=8, seed=3):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    X[rng.random((n, d)) < 0.1] = np.nan
    base = np.nan_to_num(X[:, 0]) + 0.5 * np.nan_to_num(X[:, 1])
    if task == "organ_support":
        y = (base + rng.normal(scale=0.5, size=n) > 0).astype(float)
    else:
        y = base + rng.normal(scale=0.3, size=n)
    w = rng.uniform(0.2, 1.0, size=n)
    return X, y, w, [f"f{i}" for i in range(d)]


CONFIGS = {
    "xgboost": {"objective": "squarederror", "weight_mode": "stay_balanced", "max_depth": 3, "learning_rate": 0.1},
    "lightgbm": {"objective": "squarederror", "weight_mode": "stay_balanced", "num_leaves": 8, "learning_rate": 0.1, "min_data_in_leaf": 5},
    "catboost": {"objective": "squarederror", "weight_mode": "none", "depth": 4, "learning_rate": 0.1},
    "ridge": {"objective": "squarederror", "weight_mode": "stay_balanced", "alpha": 2.0},
}


@pytest.mark.parametrize("family", ["xgboost", "lightgbm", "catboost", "ridge"])
def test_regression_roundtrip_is_bit_identical(family, tmp_path):
    X, y, w, names = toy()
    model = fm.fit_final(family, CONFIGS[family], X, y, w, task="recovery24", feature_columns=names, n_estimators=30, nthread=2)
    before = model.predict(X, names)
    meta = model.save(tmp_path / family)
    loaded = fm.load_fitted_model(tmp_path / family / fm.META_NAME)
    after = loaded.predict(X, names)
    np.testing.assert_array_equal(before, after)
    assert meta["family"] == family and meta["names_sha256"] == fm.names_sha256(names)
    assert loaded.feature_columns == tuple(names)
    assert float(np.mean(np.abs(before - y))) < float(np.mean(np.abs(y - np.median(y))))  # learned something


@pytest.mark.parametrize("family", ["xgboost", "lightgbm", "catboost", "logistic"])
def test_classification_roundtrip_probabilities(family, tmp_path):
    X, y, w, names = toy("organ_support")
    cfg = {"objective": "binary", "weight_mode": "none", "class_weight": "none", "max_depth": 3, "num_leaves": 8,
           "min_data_in_leaf": 5, "depth": 4, "learning_rate": 0.1, "C": 1.0}
    model = fm.fit_final(family, cfg, X, y, w, task="organ_support", feature_columns=names, n_estimators=25, nthread=2)
    p = model.predict(X)
    assert p.min() >= 0.0 and p.max() <= 1.0
    model.save(tmp_path / family)
    np.testing.assert_array_equal(p, fm.load_fitted_model(tmp_path / family).predict(X))


def test_final_model_matches_models_fit_predict_exactly():
    X, y, w, names = toy()
    cfg = dict(CONFIGS["xgboost"], n_estimators=40)
    ref, _ = M.fit_predict("xgboost", cfg, X, y, w, X, task="recovery24", es=None, nthread=2, seed=7)
    got = fm.fit_final("xgboost", cfg, X, y, w, task="recovery24", feature_columns=names, n_estimators=40, nthread=2, seed=7).predict(X)
    np.testing.assert_allclose(got, ref, rtol=0, atol=1e-6)  # xgb float32 DMatrix; same helper, same params
    ref_l, _ = M.fit_predict("ridge", CONFIGS["ridge"], X, y, w, X, task="recovery24")
    got_l = fm.fit_final("ridge", CONFIGS["ridge"], X, y, w, task="recovery24", feature_columns=names).predict(X)
    np.testing.assert_allclose(got_l, ref_l, rtol=0, atol=1e-9)


def test_training_capture_restores_library_functions():
    import xgboost as xgb

    original = xgb.train
    X, y, w, names = toy()
    fm.fit_final("xgboost", CONFIGS["xgboost"], X, y, w, task="recovery24", feature_columns=names, n_estimators=5, nthread=1)
    assert xgb.train is original


def test_tamper_fails_closed(tmp_path):
    X, y, w, names = toy()
    model = fm.fit_final("xgboost", CONFIGS["xgboost"], X, y, w, task="recovery24", feature_columns=names, n_estimators=10, nthread=1)
    model.save(tmp_path)
    (tmp_path / "model.json").write_bytes((tmp_path / "model.json").read_bytes() + b" ")
    with pytest.raises(fm.HashMismatchError):
        fm.load_fitted_model(tmp_path)
    # tampered feature columns in the meta
    model.save(tmp_path)
    meta = json.loads((tmp_path / fm.META_NAME).read_text())
    meta["feature_columns"] = list(reversed(meta["feature_columns"]))
    (tmp_path / fm.META_NAME).write_text(json.dumps(meta))
    with pytest.raises(fm.HashMismatchError):
        fm.load_fitted_model(tmp_path)


def test_column_order_and_width_mismatch_fail_closed(tmp_path):
    X, y, w, names = toy()
    model = fm.fit_final("lightgbm", CONFIGS["lightgbm"], X, y, w, task="recovery24", feature_columns=names, n_estimators=10, nthread=1)
    with pytest.raises(fm.ColumnMismatchError):
        model.predict(X, list(reversed(names)))
    with pytest.raises(fm.ColumnMismatchError):
        model.predict(X[:, :-1])


def test_n_estimators_rule_and_input_validation():
    assert fm.n_estimators_rule([100, 120, 110]) == 121  # ceil(1.10 * 110)
    assert fm.n_estimators_rule([56]) == 62
    with pytest.raises(fm.FinalModelError):
        fm.n_estimators_rule([])
    X, y, w, names = toy("organ_support")
    with pytest.raises(fm.FinalModelError):
        fm.fit_final("ridge", {}, X, y, w, task="organ_support", feature_columns=names)
    with pytest.raises(fm.FinalModelError):
        fm.fit_final("xgboost", {"objective": "binary"}, X, y + 0.5, w, task="organ_support", feature_columns=names, n_estimators=3)
