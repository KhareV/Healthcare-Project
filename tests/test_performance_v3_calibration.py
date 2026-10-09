import numpy as np
import pytest

from performance_v3 import calibration as cal


def _data(n=600, seed=0):
    rng = np.random.default_rng(seed)
    subj = np.repeat(np.arange(n // 3), 3)
    true_logit = rng.normal(size=n)
    y = (rng.random(n) < 1 / (1 + np.exp(-true_logit))).astype(float)
    score = 1 / (1 + np.exp(-(0.4 * true_logit + 0.8)))  # miscalibrated, monotone in truth
    w = 1.0 / 3.0 * np.ones(n)
    folds = (subj % 5).astype(int)
    return score, y, w, subj, folds


@pytest.mark.parametrize("method", ["none", "platt", "isotonic", "beta"])
def test_round_trip_json_and_monotone(method):
    import json

    s, y, w, _, _ = _data()
    frozen = cal.fit_calibrator(method, s, y, w)
    again = json.loads(json.dumps(frozen))
    grid = np.linspace(0.0, 1.0, 201)
    out = cal.apply_calibrator(again, grid)
    assert np.all(np.diff(out) >= -1e-12)
    assert np.all((out >= 0) & (out <= 1))
    assert np.array_equal(out, cal.apply_calibrator(frozen, grid))


@pytest.mark.parametrize("method", ["platt", "beta"])
def test_strictly_monotone_preserves_ranking_and_auprc(method):
    s, y, w, _, _ = _data()
    frozen = cal.fit_calibrator(method, s, y, w)
    out = cal.apply_calibrator(frozen, s)
    assert np.array_equal(np.argsort(s, kind="stable"), np.argsort(out, kind="stable"))
    m_raw = cal.calibration_metrics(s, y, w)
    m_out = cal.calibration_metrics(out, y, w)
    assert m_out["auprc"] == pytest.approx(m_raw["auprc"])


def test_isotonic_creates_ties():
    s, y, w, _, _ = _data()
    out = cal.apply_calibrator(cal.fit_calibrator("isotonic", s, y, w), s)
    assert len(np.unique(out)) < len(np.unique(s))


def test_platt_improves_log_loss_on_miscalibrated_scores():
    s, y, w, _, _ = _data(n=3000, seed=1)
    raw = cal.calibration_metrics(s, y, w)
    platt = cal.calibration_metrics(cal.apply_calibrator(cal.fit_calibrator("platt", s, y, w), s), y, w)
    assert platt["log_loss"] < raw["log_loss"]
    assert 0.0 <= platt["ece_10_equal_mass"] < raw["ece_10_equal_mass"] + 1e-9


def test_nested_crossfit_no_overlap_and_selection():
    s, y, w, subj, folds = _data(n=900, seed=2)
    res = cal.nested_crossfit_calibration(s, y, w, subj, folds)
    for a in res["_audit"]:
        assert a["row_overlap"] == 0 and a["subject_overlap"] == 0
        assert a["n_fit"] + a["n_eval"] == len(s)
    for m in cal.METHODS:
        assert np.all(np.isfinite(res[m]["heldout_prob"]))
    # held-out value of fold k must be independent of its own labels: flip fold-0 labels,
    # fold-0 held-out output must not change (fit never saw fold-0 rows)
    y2 = y.copy(); y2[folds == 0] = 1 - y2[folds == 0]
    res2 = cal.nested_crossfit_calibration(s, y2, w, subj, folds)
    assert np.array_equal(res["platt"]["heldout_prob"][folds == 0], res2["platt"]["heldout_prob"][folds == 0])
    chosen = cal.select_calibrator(res)
    assert chosen in cal.METHODS


def test_nested_rejects_subject_spanning_folds():
    s, y, w, subj, folds = _data(n=90)
    bad = folds.copy(); bad[0] = (bad[0] + 1) % 5
    with pytest.raises(cal.CalibrationError):
        cal.nested_crossfit_calibration(s, y, w, subj, bad)


def test_select_calibrator_tie_order_and_tolerance():
    def r(none, platt, beta, iso):
        return {k: {"metrics": {"log_loss": v}} for k, v in zip(("none", "platt", "beta", "isotonic"), (none, platt, beta, iso))}
    assert cal.select_calibrator(r(0.60, 0.60, 0.60, 0.60)) == "none"
    assert cal.select_calibrator(r(0.70, 0.601, 0.600, 0.599)) == "platt"  # within 0.002 of best, simpler first
    assert cal.select_calibrator(r(0.70, 0.70, 0.70, 0.60)) == "isotonic"
    assert cal.select_calibrator(r(0.70, 0.70, 0.5995, 0.60)) == "beta"


def test_threshold_weighted_f1_lowest_on_tie_and_value():
    p = np.array([0.1, 0.4, 0.4, 0.8])
    y = np.array([0, 1, 1, 1])
    w = np.ones(4)
    res = cal.select_threshold_weighted_f1(p, y, w)
    assert res["f1"] == pytest.approx(1.0)
    assert res["threshold"] == pytest.approx(0.4)  # also 0.8 is not better; lowest max
    # V2 rule parity
    pred = (p >= res["threshold"]).astype(int)
    assert cal._weighted_f1(y, pred, w) == pytest.approx(res["f1"])
    # all-tied F1 plateau chooses the lowest threshold
    p2 = np.array([0.2, 0.3, 0.9]); y2 = np.array([1, 1, 1])
    assert cal.select_threshold_weighted_f1(p2, y2, np.ones(3))["threshold"] == 0.0
