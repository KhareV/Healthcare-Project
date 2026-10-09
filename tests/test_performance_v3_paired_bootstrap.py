import numpy as np
import pytest

from evaluation.bootstrap import grouped_bootstrap
from performance_v3.paired_bootstrap import (
    PairedBootstrapError,
    align_paired_records,
    ci_excludes_zero,
    paired_stay_cluster_bootstrap,
)
from performance_v3.task_metrics import make_records, record_metric_fn


def _recs(n_stays=40, per=3, seed=0, noise=1.0, cls=False):
    rng = np.random.default_rng(seed)
    stay = np.repeat(np.arange(n_stays), per).astype(str)
    times = [f"t{j}" for _ in range(n_stays) for j in range(per)]
    if cls:
        y = rng.integers(0, 2, n_stays * per).astype(float)
        a = np.clip(y * 0.5 + rng.random(len(y)) * 0.5, 0, 1)
        b = np.clip(y * 0.3 + rng.random(len(y)) * 0.7, 0, 1)
    else:
        y = rng.normal(size=n_stays * per)
        a = y + rng.normal(scale=noise, size=len(y))
        b = y + rng.normal(scale=noise * 0.5, size=len(y))
    return make_records(y, a, stay, times), make_records(y, b, stay, times)


def test_paired_equals_independent_bootstraps_under_forced_draws():
    ra, rb = _recs()
    fn = record_metric_fn("recovery24", "mae")
    res = paired_stay_cluster_bootstrap(ra, rb, fn, n=60, seed=11)
    ga = grouped_bootstrap(ra, metric_fn=fn, metric_name="m", task="t", n_bootstrap=60, seed=11)
    gb = grouped_bootstrap(rb, metric_fn=fn, metric_name="m", task="t", n_bootstrap=60, seed=999, forced_draws=ga.sampled_cluster_sequences)
    assert res["n_valid_joint"] == 60
    assert np.allclose(res["per_replicate_a"], ga.bootstrap_distribution)
    assert np.allclose(res["per_replicate_b"], gb.bootstrap_distribution)
    assert np.allclose(res["per_replicate_diff"], np.array(gb.bootstrap_distribution) - np.array(ga.bootstrap_distribution))
    assert res["diff_point"] == pytest.approx(gb.point_estimate - ga.point_estimate)
    assert res["difference_definition"] == "metric(B) - metric(A)"


def test_better_model_gives_negative_diff_excluding_zero():
    ra, rb = _recs(n_stays=80, noise=1.5)
    res = paired_stay_cluster_bootstrap(ra, rb, record_metric_fn("recovery24", "mae"), n=200, seed=3)
    assert res["diff_point"] < 0
    assert ci_excludes_zero(res, 0.95) and set(res["ci"]) == {"0.95", "0.9875"}
    lo95, hi95 = res["ci"]["0.95"]; lo99, hi99 = res["ci"]["0.9875"]
    assert lo99 <= lo95 <= hi95 <= hi99


def test_row_set_mismatch_raises_and_reorder_aligns():
    ra, rb = _recs(n_stays=10)
    with pytest.raises(PairedBootstrapError):
        paired_stay_cluster_bootstrap(ra, rb[:-1], record_metric_fn("recovery24", "mae"), n=5, seed=1)
    shuffled = list(rb); shuffled.reverse()
    a2, b2 = align_paired_records(ra, shuffled)
    assert [(r.stay_id, r.prediction_time) for r in a2] == [(r.stay_id, r.prediction_time) for r in b2]
    res = paired_stay_cluster_bootstrap(ra, shuffled, record_metric_fn("recovery24", "mae"), n=20, seed=1)
    res_ordered = paired_stay_cluster_bootstrap(ra, rb, record_metric_fn("recovery24", "mae"), n=20, seed=1)
    assert np.allclose(res["per_replicate_diff"], res_ordered["per_replicate_diff"])


def test_class_diversity_skip_is_joint():
    # 3 stays; stay "x" is the only positive stay. Replicates drawing no "x" have no
    # positives -> AUROC undefined for BOTH models; both must drop the same replicates.
    stay = ["x", "x", "y", "y", "z", "z"]
    times = ["t0", "t1"] * 3
    y = [1, 0, 0, 0, 0, 0]
    a = make_records(y, [0.9, 0.2, 0.4, 0.1, 0.3, 0.2], stay, times)
    b = make_records(y, [0.8, 0.3, 0.5, 0.2, 0.1, 0.1], stay, times)
    res = paired_stay_cluster_bootstrap(a, b, record_metric_fn("organ_support", "auroc"), n=80, seed=5)
    assert 0 < res["n_valid_joint"] < 80
    assert np.array_equal(np.isfinite(res["per_replicate_all_a"]), np.isfinite(res["per_replicate_all_b"]))
    assert res["n_invalid"] == 80 - res["n_valid_joint"]
    assert len(res["per_replicate_diff"]) == res["n_valid_joint"]


def test_duplicate_keys_rejected():
    ra, rb = _recs(n_stays=3)
    with pytest.raises(PairedBootstrapError):
        align_paired_records(list(ra) + [ra[0]], rb)
