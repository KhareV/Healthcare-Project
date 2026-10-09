import copy

import numpy as np
import pytest
import torch

from performance_v2 import gru_training as V2
from performance_v2.context_normalization import apply_normalization, fit_normalization
from performance_v3 import gru_search as G
from performance_v3.search import TrialLedger, _es_mask, assign_folds
from test_performance_v3_features import synthetic_rows


def rows60():
    rows = synthetic_rows(n_stays=60, cutoffs=4, seed=9)
    for i, r in enumerate(rows):
        age, sex, group = r["static_features"]
        r["age_years"], r["sex_category"], r["cardiac_condition_group"] = float(age), sex, group
        if i == 3:
            r["cardiac_condition_group"] = "SOMETHING_NEW"  # unmatched category -> all-zero one-hot like V2
    return rows


@pytest.fixture(scope="module")
def pool_rows():
    rows = rows60()
    return rows, G.pool_from_rows(rows)


def test_context_variant_mapping():
    assert G.context_variant_for("V3_CTRL", "recovery24") == "B_MIN"
    assert G.context_variant_for("V3_CTRL", "icu_stay_time") == "B_PLUS_F"
    assert G.context_variant_for("V3_CTRL", "organ_support") == "B_FULL"
    assert G.context_variant_for("V3_BC", "recovery24") == "B_FULL"
    assert G.context_variant_for("V3_FULL", "organ_support") == "B_PLUS_F"
    with pytest.raises(ValueError):
        G.context_variant_for("V9", "recovery24")


def test_batch_and_context_parity_with_v2(pool_rows):
    rows, pool = pool_rows
    idx = np.arange(0, 40, 3)
    sub = [rows[i] for i in idx]
    variant = "B_PLUS_F"
    norm = fit_normalization(sub, variant)
    v2_ctx = apply_normalization(sub, norm)
    cols, mean, std = G._norm_params(pool, idx, variant)
    ctx = G._context(pool, idx, cols, mean, std)
    assert np.allclose(ctx, v2_ctx, atol=1e-6)
    age_mean, age_std = V2.fit_age_normalization(sub)
    targets = np.arange(len(idx), dtype=np.float32)
    ref = V2.build_canonical_batch(sub, targets, v2_ctx, torch.device("cpu"), age_mean, age_std)
    mine = G.make_batch(pool, idx, targets, ctx, age_mean, age_std)
    assert torch.equal(ref.sequence, mine.sequence)
    assert torch.equal(ref.observation_mask, mine.observation_mask)
    assert torch.equal(ref.padding_mask, mine.padding_mask)
    assert torch.allclose(ref.tslo, mine.tslo)
    assert torch.allclose(ref.static_features, mine.static_features, atol=1e-6)


def test_train_loop_matches_v2_train_gru(pool_rows):
    rows, pool = pool_rows
    tr = np.arange(0, 120)
    es = np.arange(120, 160)
    variant = "B_MIN"
    cols, mean, std = G._norm_params(pool, tr, variant)
    tr_ctx, es_ctx = G._context(pool, tr, cols, mean, std), G._context(pool, es, cols, mean, std)
    y = np.asarray([rows[i]["icu_time_log1p"] for i in range(160)], dtype=np.float64)
    age_mean, age_std = V2.fit_age_normalization([rows[i] for i in tr])
    t_mean, t_std = V2.standardize_targets(y[tr])
    base = V2.SingleOutputGRU(V2.make_encoder_config(variant, 16 if False else 64, 1, 0.1))
    m_ref, m_mine = copy.deepcopy(base), copy.deepcopy(base)
    kw = dict(loss_name="huber", lr=1e-3, weight_decay=1e-5, batch_size=64, max_epochs=2, patience=6, seed=5,
              target_mean=t_mean, target_std=t_std, age_mean=age_mean, age_std=age_std)
    m_ref, _ = V2.train_gru(m_ref, [rows[i] for i in tr], y[tr], tr_ctx, [rows[i] for i in es], y[es], es_ctx, device=torch.device("cpu"), **kw)
    m_mine, best, run = G.train_arrays(m_mine, pool, tr, y[tr], tr_ctx, es, y[es], es_ctx, **kw)
    p_ref = V2.predict_gru(m_ref, [rows[i] for i in es], es_ctx, torch.device("cpu"), age_mean, age_std, t_mean, t_std)
    p_mine = G.predict_arrays(m_mine, pool, es, es_ctx, age_mean, age_std, t_mean, t_std, probability=False)
    assert run == 2 and best in (1, 2)
    assert np.allclose(p_ref, p_mine, atol=1e-4)


def test_sample_configs_deterministic_and_in_space():
    a = G.sample_gru_configs(8, 123, "recovery24")
    b = G.sample_gru_configs(8, 123, "recovery24")
    assert [c["config_hash"] for c in a] == [c["config_hash"] for c in b]
    assert len({c["config_hash"] for c in a}) == 8
    for c in a:
        assert c["hidden_dim"] in G.HIDDEN_DIMS and c["num_layers"] in G.NUM_LAYERS and c["batch_size"] in G.BATCH_SIZES
        assert 0.10 <= c["dropout"] <= 0.40 and 1e-4 <= c["lr"] <= 3e-3 and 1e-6 <= c["weight_decay"] <= 1e-3
        assert c["loss_name"] in ("huber", "l1") and c["max_epochs"] == 40 and c["patience"] == 6
    assert {c["loss_name"] for c in G.sample_gru_configs(8, 5, "organ_support")} == {"bce"}


def test_cv_uses_search_folds_and_logs_records(pool_rows, tmp_path):
    rows, pool = pool_rows
    gt = G.prepare_gru_task(pool, "organ_support")
    expected = assign_folds(gt.subject_id)
    assert all(int(f) == expected[str(s)] for f, s in zip(gt.fold, gt.subject_id))
    cfg = G.sample_gru_configs(1, 11, "organ_support")[0]
    cfg.update(hidden_dim=64, num_layers=1, batch_size=128)
    out = G.run_gru_cv(pool, gt, "V3_CTRL", cfg, folds=[0, 1], seed=11, max_epochs=1)
    ev = np.isin(gt.fold, [0, 1])
    assert np.isfinite(out["oof"][ev]).all() and np.isnan(out["oof"][~ev]).all()
    assert (out["oof"][ev] > 0).all() and (out["oof"][ev] < 1).all()
    ledger = TrialLedger(tmp_path / "ledger.jsonl")
    rec = G.run_gru_trial_logged(ledger, pool, gt, "V3_CTRL", cfg, folds=[0], seed=11, oof_dir=tmp_path / "oof", max_epochs=1)
    assert rec["status"] == "ok" and rec["stage"] == "S3" and rec["family"] == "gru" and rec["primary_metric"] == "auprc"
    assert (tmp_path / "oof").exists() and rec["oof_path"].endswith(".npz")
    again = G.run_gru_trial_logged(TrialLedger(tmp_path / "ledger.jsonl"), pool, gt, "V3_CTRL", cfg, folds=[0], seed=11, max_epochs=1)
    assert again["trial_id"] == rec["trial_id"]
    for key in ("trial_id", "stage", "task", "family", "variant", "config", "config_hash", "seed", "primary_mean", "primary_sd",
                "per_fold_primary", "secondary_means", "best_iterations", "fit_seconds", "folds", "status"):
        assert key in rec


def test_icu_and_recovery_regression_paths(pool_rows):
    rows, pool = pool_rows
    for task, metric in (("icu_stay_time", "weighted_median_ae_hours"), ("recovery48", "mae")):
        gt = G.prepare_gru_task(pool, task)
        cfg = G.sample_gru_configs(1, 3, task)[0]
        cfg.update(hidden_dim=64, num_layers=1, batch_size=128, loss_name="huber")
        out = G.run_gru_cv(pool, gt, "V3_CTRL", cfg, folds=[0], seed=11, max_epochs=1, keep_oof=False)
        assert out["primary_metric"] == metric and np.isfinite(out["primary_mean"])
