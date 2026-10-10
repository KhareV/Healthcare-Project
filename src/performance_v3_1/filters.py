"""Forward-only causal filters over the canonical 8-bin window (observed data only).

Inputs are the cache's per-bin latest value + genuine-observation mask columns; nothing outside
the approved (t-48h, t] window and nothing latent is used. Kalman parameters are label-free
method-of-moments estimates from development rows.
"""
from __future__ import annotations

from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np

CONT_CHANNELS = ("pao2", "fio2", "platelet_count", "bilirubin_total", "mean_arterial_pressure", "glasgow_coma_scale", "creatinine",
                 "urine_output_volume", "heart_rate", "systolic_blood_pressure", "diastolic_blood_pressure", "respiratory_rate",
                 "oxygen_saturation", "temperature", "lactate")
HALF_LIVES = (6.0, 12.0, 24.0)
BIN_HOURS = 6.0
FILTERS_VERSION = "performance_v3_1_causal_filters_v1"


def channel_columns(names: Sequence[str]) -> Dict[str, Tuple[List[int], List[int]]]:
    idx = {n: i for i, n in enumerate(names)}
    out: Dict[str, Tuple[List[int], List[int]]] = {}
    for ch in CONT_CHANNELS:
        vals, masks = [], []
        for b in range(8):
            base = [n for n in names if n.startswith(f"A__bin{b}__{ch}__") and n.endswith("__value")]
            if len(base) != 1:
                raise ValueError(f"channel {ch} bin {b}: expected one value column, found {base}")
            v = base[0]
            m = v[: -len("__value")] + "__mask"
            if m not in idx:
                raise ValueError(f"missing mask column {m}")
            vals.append(idx[v]); masks.append(idx[m])
        out[ch] = (vals, masks)
    return out


def observed_values(X: np.ndarray, cols: Tuple[List[int], List[int]]) -> np.ndarray:
    vals, masks = cols
    v = X[:, vals].astype(np.float64)
    m = X[:, masks] > 0.5
    return np.where(m & np.isfinite(v), v, np.nan)


def fit_kalman_params(X: np.ndarray, names: Sequence[str]) -> Dict[str, Dict[str, float]]:
    params = {}
    for ch, cols in channel_columns(names).items():
        V = observed_values(X, cols)
        d1 = (V[:, 1:] - V[:, :-1]).ravel(); d1 = d1[np.isfinite(d1)]
        d2 = (V[:, 2:] - V[:, :-2]).ravel(); d2 = d2[np.isfinite(d2)]
        v1, v2 = (float(np.var(d1)), float(np.var(d2))) if len(d1) > 50 and len(d2) > 50 else (1.0, 2.0)
        q = max(v2 - v1, 1e-9)
        r = max((v1 - q) / 2.0, 1e-9 + 0.01 * v1)
        params[ch] = {"q": q, "r": r, "p0": float(np.nanvar(V)) if np.isfinite(np.nanvar(V)) else 1.0}
    return params


def kalman_filter(V: np.ndarray, q: float, r: float, p0: float) -> Tuple[np.ndarray, np.ndarray]:
    n = V.shape[0]
    m = np.full(n, np.nan); P = np.full(n, p0)
    for b in range(V.shape[1]):
        started = np.isfinite(m)
        P = np.where(started, P + q, P)
        obs = np.isfinite(V[:, b])
        first = obs & ~started
        upd = obs & started
        m = np.where(first, V[:, b], m)
        P = np.where(first, r, P)
        K = P / (P + r)
        m = np.where(upd, m + K * (V[:, b] - m), m)
        P = np.where(upd, (1 - K) * P, P)
    return m, np.where(np.isfinite(m), P, np.nan)


def build_filter_features(X: np.ndarray, names: Sequence[str], params: Mapping[str, Mapping[str, float]]) -> Tuple[np.ndarray, Tuple[str, ...]]:
    cols_all = channel_columns(names)
    out, out_names = [], []
    age = (7 - np.arange(8)) * BIN_HOURS
    for ch, cols in cols_all.items():
        V = observed_values(X, cols)
        M = np.isfinite(V).astype(np.float64)
        Z = np.nan_to_num(V, nan=0.0)
        ew = {}
        for h in HALF_LIVES:
            w = 0.5 ** (age / h)
            num = (Z * M * w).sum(axis=1); den = (M * w).sum(axis=1)
            ew[h] = np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)
            out.append(ew[h]); out_names.append(f"K__{ch}__ewma{int(h)}h")
        out.append(ew[6.0] - ew[24.0]); out_names.append(f"K__{ch}__momentum")
        p = params[ch]
        m, P = kalman_filter(V, p["q"], p["r"], p["p0"])
        out.append(m); out_names.append(f"K__{ch}__kal_mean")
        out.append(P); out_names.append(f"K__{ch}__kal_var")
    return np.column_stack(out).astype(np.float32), tuple(out_names)


def family_columns(filter_names: Sequence[str], family: str) -> List[str]:
    if family == "F0":
        return []
    keep = []
    for n in filter_names:
        ew = ("ewma" in n) or n.endswith("__momentum")
        kal = "__kal_" in n
        if family == "F1" and ew or family == "F2" and kal or family == "F3" and (ew or kal):
            keep.append(n)
    return keep
