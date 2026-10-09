import json
from pathlib import Path

import numpy as np
import pytest

from performance_v3 import search
from performance_v3.features import FeatureCache, META_BOOL


def toy_cache(n_stays=150, cutoffs=4, seed=5):
    rng = np.random.default_rng(seed)
    n = n_stays * cutoffs
    names = tuple(
        [f"A__c{i}" for i in range(6)]
        + ["B__elapsed_episode_hours_at_t", "B__cutoff_index", "B__hours_since_first_eligible_cutoff", "B__current_sofa"]
    )
    X = rng.normal(size=(n, len(names))).astype(np.float32)
    stay = np.repeat([f"E{i:05d}" for i in range(n_stays)], cutoffs)
    subj = np.repeat([f"S{i:05d}" for i in range(n_stays)], cutoffs)
    signal = X[:, 0] + 0.5 * X[:, 9]
    meta = {
        "subject_id": subj, "stay_id": stay,
        "prediction_time": np.asarray([f"2108-01-01T{k % 24:02d}:00:00.000000Z" for k in range(n)]),
        "cohort_id": np.asarray(["SYN"] * n),
        "delta_sofa_24": signal + rng.normal(scale=0.3, size=n),
        "delta_sofa_48": signal + rng.normal(scale=0.5, size=n),
        "icu_time_log1p": np.log1p(np.abs(5 + 3 * X[:, 1] + rng.normal(scale=0.5, size=n)) + 1),
        "organ_support_label": (signal + rng.normal(scale=0.7, size=n) > 0).astype(np.float64),
        "baseline_sofa": np.abs(X[:, 9]),
    }
    for col in META_BOOL:
        meta[col] = np.ones(n, dtype=bool)
    return FeatureCache(names=names, matrix=X, meta=meta)


def test_prepare_is_subject_grouped_and_columns_selected():
    cache = toy_cache()
    td = search.prepare_task_data(cache, "recovery24", "V3_B")
    assert td.X.shape[1] == len(td.names)
    assert set(np.unique(td.fold)) == {0, 1, 2, 3, 4}
    for subject in np.unique(td.subject_id)[:20]:
        assert len(set(td.fold[td.subject_id == subject])) == 1
    assert np.isclose(td.w.sum(), td.n_stays)  # each stay totals weight 1


@pytest.mark.parametrize("family", ["xgboost", "ridge"])
def test_cv_trial_regression_runs_and_beats_zero_baseline(family):
    td = search.prepare_task_data(toy_cache(), "recovery24", "V3_B")
    cfg = {"objective": "squarederror", "weight_mode": "none", "alpha": 1.0, "max_depth": 3, "learning_rate": 0.1,
           "n_estimators": 60, "early_stopping_rounds": 10, "subsample": 1.0, "colsample_bytree": 1.0,
           "min_child_weight": 1, "reg_alpha": 0.0, "reg_lambda": 1.0, "gamma": 0.0}
    out = search.run_cv_trial(td, family, cfg, folds=[0, 1], seed=1, nthread=2, keep_oof=True)
    zero_mae = float(np.mean(np.abs(td.y_eval[np.isin(td.fold, [0, 1])])))
    assert out["primary_mean"] < zero_mae
    oof = out["oof"]
    assert np.isfinite(oof[np.isin(td.fold, [0, 1])]).all()
    assert np.isnan(oof[~np.isin(td.fold, [0, 1])]).all()  # outer folds never predicted outside requested folds


def test_support_trial_uses_raw_auprc_and_icu_uses_hours():
    cache = toy_cache()
    td = search.prepare_task_data(cache, "organ_support", "V3_B")
    cfg = {"objective": "binary", "weight_mode": "none", "class_weight": "none", "C": 1.0}
    out = search.run_cv_trial(td, "logistic", cfg, folds=[0], seed=1, nthread=2)
    assert out["primary_metric"] == "auprc" and 0.5 < out["primary_mean"] <= 1.0
    td_icu = search.prepare_task_data(cache, "icu_stay_time", "V3_B")
    assert np.allclose(np.log1p(td_icu.y_eval), td_icu.y)
    cfg_icu = {"objective": "squarederror", "weight_mode": "stay_balanced", "alpha": 1.0}
    out_icu = search.run_cv_trial(td_icu, "ridge", cfg_icu, folds=[0], seed=1)
    assert out_icu["primary_metric"] == "weighted_median_ae_hours"


def test_ledger_resumes_and_records_failures(tmp_path):
    td = search.prepare_task_data(toy_cache(), "recovery24", "V3_B")
    ledger = search.TrialLedger(tmp_path / "ledger.jsonl")
    ok = search.run_trial_logged(ledger, td, "S1", "ridge", {"objective": "squarederror", "weight_mode": "none", "alpha": 1.0, "config_hash": "abc"}, folds=[0], seed=1)
    assert ok["status"] == "ok"
    bad = search.run_trial_logged(ledger, td, "S1", "xgboost", {"objective": "nonsense", "weight_mode": "none", "config_hash": "bad"}, folds=[0], seed=1)
    assert bad["status"] == "failed" and "error" in bad
    again = search.TrialLedger(tmp_path / "ledger.jsonl")
    assert again.has(ok["trial_id"]) and again.has(bad["trial_id"])
    best = search.best_trial(again.records(), "recovery24")
    assert best["trial_id"] == ok["trial_id"]
    assert len((tmp_path / "ledger.jsonl").read_text().splitlines()) == 2


def test_early_stopping_split_is_training_subjects_only_and_deterministic():
    td = search.prepare_task_data(toy_cache(), "recovery24", "V3_B")
    m1 = search._es_mask(td.subject_id, 0, "v")
    m2 = search._es_mask(td.subject_id, 0, "v")
    assert (m1 == m2).all() and 0 < m1.mean() < 0.3
    # every row of a subject shares the early-stopping decision
    for subject in np.unique(td.subject_id)[:25]:
        assert len(set(m1[td.subject_id == subject])) == 1
