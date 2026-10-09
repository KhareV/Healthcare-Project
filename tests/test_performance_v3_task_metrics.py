import math

import numpy as np
import pytest

from evaluation.metrics import _weighted_auroc, _weighted_average_precision
from performance_v3.task_metrics import (
    fast_weighted_auprc,
    fast_weighted_auroc,
    icu_metrics,
    is_better,
    make_records,
    primary_metric,
    record_metric_fn,
    recovery_metrics,
    stay_weights,
    support_metrics,
)

STAYS = ["a", "a", "a", "b", "c", "c"]


def test_recovery_stay_balanced_hand_computed():
    y = [0, 0, 0, 0, 0, 0]
    pred = [3, 3, 3, 1, 2, 2]  # stay means of |err|: a=3, b=1, c=2
    m = recovery_metrics(y, pred, STAYS, horizon="24h")
    assert m["mae"] == pytest.approx((3 + 1 + 2) / 3)  # not the row mean (14/6)
    assert m["median_ae"] == 2.0
    assert m["n_examples"] == 6 and m["n_stays"] == 3
    assert primary_metric("recovery48", y, pred, STAYS) == pytest.approx(2.0)


def test_icu_postprocess_and_hours():
    hours = [10.0, 10.0, 10.0, 20.0, 30.0, 30.0]
    raw = np.log1p(np.array(hours)) + 0.0
    raw[0] = -3.0  # clamps to 0h before metric
    m = icu_metrics(hours, raw, STAYS)
    # errors: stay a -> [10,0,0] mean ... median uses 1/n weights
    assert m["weighted_median_ae_hours"] == pytest.approx(0.0, abs=1e-6)
    assert m["mae"] == pytest.approx((10 / 3 + 0 + 0) / 3, rel=1e-6)


def test_support_metrics_and_threshold():
    y = [1, 0, 1, 0, 1, 0]
    s = [0.9, 0.2, 0.6, 0.4, 0.3, 0.1]
    m = support_metrics(y, s, STAYS, threshold=0.5)
    assert 0 <= m["auprc"] <= 1 and 0 <= m["auroc"] <= 1
    assert set(m["confusion"]) == {"TP", "TN", "FP", "FN"}
    assert m["log_loss"] > 0 and m["n_positive_examples"] == 3
    assert primary_metric("organ_support", y, s, STAYS) == pytest.approx(m["auprc"])


def test_fast_auc_matches_reference_with_ties():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    s = np.round(rng.random(200), 1)  # heavy ties
    w = rng.random(200) + 0.1
    assert fast_weighted_auroc(y, s, w) == pytest.approx(_weighted_auroc(tuple(y), tuple(s), tuple(w)))
    assert fast_weighted_auprc(y, s, w) == pytest.approx(_weighted_average_precision(tuple(y), tuple(s), tuple(w)))


def test_record_metric_fn_matches_wrappers():
    y = np.array([1, 0, 1, 0, 1, 0]); s = np.array([0.9, 0.2, 0.6, 0.4, 0.3, 0.1])
    recs = make_records(y, s, STAYS)
    full = support_metrics(y, s, STAYS, threshold=0.5)
    assert record_metric_fn("organ_support", "auprc")(recs) == pytest.approx(full["auprc"])
    assert record_metric_fn("organ_support", "f1", threshold=0.5)(recs) == pytest.approx(full["f1"])
    rec_y = np.array([0.0, 1.0, 2.0, 0.0, 1.0, 5.0]); rec_p = np.zeros(6)
    r = make_records(rec_y, rec_p, STAYS)
    assert record_metric_fn("recovery24", "mae")(r) == pytest.approx(recovery_metrics(rec_y, rec_p, STAYS)["mae"])
    assert record_metric_fn("recovery24", "median_ae")(r) == pytest.approx(recovery_metrics(rec_y, rec_p, STAYS)["median_ae"])


def test_is_better_direction():
    assert is_better("recovery24", 1.0, 1.1)
    assert is_better("icu_stay_time", 5.0, 6.0)
    assert is_better("organ_support", 0.8, 0.7)
    assert not is_better("organ_support", 0.7, 0.8)
    with pytest.raises(ValueError):
        is_better("x", 1, 2)


def test_stay_weights_total_one_per_stay():
    w = stay_weights(STAYS)
    assert math.isclose(w.sum(), 3.0)
