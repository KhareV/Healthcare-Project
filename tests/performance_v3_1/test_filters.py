import numpy as np
import pytest

from performance_v3_1 import filters as F


def make_names():
    names = []
    for b in range(8):
        for ch in F.CONT_CHANNELS:
            base = f"A__bin{b}__{ch}__latest"
            names += [base + "__mask", base + "__tslo", base + "__value"]
    return tuple(names)


NAMES = make_names()
IDX = {n: i for i, n in enumerate(NAMES)}


def row(ch, values):
    """values: list of 8 (value or None) for one channel; others unobserved."""
    x = np.zeros(len(NAMES), dtype=np.float32)
    for b, v in enumerate(values):
        x[IDX[f"A__bin{b}__{ch}__latest__mask"]] = 0.0 if v is None else 1.0
        x[IDX[f"A__bin{b}__{ch}__latest__value"]] = 123.0 if v is None else v  # forward-fill-like junk under a 0 mask
    return x


def test_channel_columns_cover_all_continuous_channels():
    cols = F.channel_columns(NAMES)
    assert set(cols) == set(F.CONT_CHANNELS) and all(len(v[0]) == 8 and len(v[1]) == 8 for v in cols.values())


def test_masked_values_are_ignored_and_ewma_matches_manual():
    vals = [None, None, 10.0, None, 20.0, None, None, 30.0]
    X = np.stack([row("heart_rate", vals)])
    params = {ch: {"q": 1.0, "r": 1.0, "p0": 1.0} for ch in F.CONT_CHANNELS}
    M, names = F.build_filter_features(X, NAMES, params)
    j = names.index("K__heart_rate__ewma12h")
    w = lambda b: 0.5 ** ((7 - b) * 6 / 12)
    expected = (10 * w(2) + 20 * w(4) + 30 * w(7)) / (w(2) + w(4) + w(7))
    assert M[0, j] == pytest.approx(expected, rel=1e-5)


def test_no_observation_gives_nan_and_row_independence():
    params = {ch: {"q": 1.0, "r": 1.0, "p0": 1.0} for ch in F.CONT_CHANNELS}
    a = row("lactate", [None] * 8); b = row("lactate", [1, 2, 3, 4, 5, 6, 7, 8])
    M, names = F.build_filter_features(np.stack([a, b]), NAMES, params)
    j = names.index("K__lactate__kal_mean")
    assert np.isnan(M[0, j]) and np.isfinite(M[1, j])
    M2, _ = F.build_filter_features(np.stack([b]), NAMES, params)
    assert M2[0, j] == M[1, j]


def test_kalman_variance_grows_with_gap_and_single_observation_exact():
    m, P = F.kalman_filter(np.array([[5.0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]]), q=0.5, r=1.0, p0=3.0)
    assert m[0] == 5.0 and P[0] == pytest.approx(1.0 + 7 * 0.5)
    m2, P2 = F.kalman_filter(np.array([[5.0] + [np.nan] * 6 + [5.0]]), q=0.5, r=1.0, p0=3.0)
    assert P2[0] < P[0]


def test_filter_uses_only_forward_information():
    """Changing a LATER bin changes the final state, but the final state of a prefix is unchanged by it."""
    base = np.array([[1.0, 2.0, 3.0, 4.0, np.nan, np.nan, np.nan, np.nan]])
    changed = base.copy(); changed[0, 7] = 99.0
    pre_a = F.kalman_filter(base[:, :4], 0.3, 1.0, 2.0)
    pre_b = F.kalman_filter(changed[:, :4], 0.3, 1.0, 2.0)
    assert pre_a[0][0] == pre_b[0][0]


def test_parameter_estimation_positive():
    rng = np.random.default_rng(0)
    X = np.zeros((400, len(NAMES)), dtype=np.float32)
    for ch in F.CONT_CHANNELS:
        walk = np.cumsum(rng.normal(0, 0.5, (400, 8)), axis=1) + rng.normal(0, 1.0, (400, 8))
        for b in range(8):
            X[:, IDX[f"A__bin{b}__{ch}__latest__mask"]] = 1.0
            X[:, IDX[f"A__bin{b}__{ch}__latest__value"]] = walk[:, b]
    p = F.fit_kalman_params(X, NAMES)
    assert all(v["q"] > 0 and v["r"] > 0 for v in p.values())
    assert 0.1 < p["heart_rate"]["q"] < 0.6  # true q = 0.25
