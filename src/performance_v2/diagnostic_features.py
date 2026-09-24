"""Lawful, cutoff-safe DIAGNOSTIC feature generator for Performance-v2 Phase 1.

DEVELOPMENT_DIAGNOSTIC_ONLY. This is NOT the production canonical builder
(src/data/xgb_canonical.py, src/preprocess/synthetic_phase10.py) and never
replaces or modifies it. Every feature here is built ONLY from information
already at-or-before the row's own prediction cutoff t:

  - history_values / observation_mask / tslo_hours: the row's own 8x21
    canonical bins, each bin covering (t-48h, t], already cutoff-safe by
    construction (this is exactly what the frozen preprocessor feeds the
    served models).
  - elapsed_episode_hours_at_t, cutoff_index: derived only from
    prediction_time and intime (performance_v2.data_loading), both known
    at t.
  - baseline_sofa and past-cutoff SOFA trend: computed by the frozen,
    unmodified data.synthetic.sofa.sofa_at() at cutoffs <= t only (t, t-6h,
    t-12h, t-24h, t-48h, clamped to intime). sofa_at() itself only reads
    history strictly at-or-before its cutoff argument, so calling it at a
    past cutoff cannot see anything beyond that past cutoff.

Never uses: outtime, any post-t event, delta_sofa_24/48, icu_time_log1p,
organ_support_label, or any other prohibited field (see
configs/synthetic/feature_schema_v2.json#prohibited_fields).

Missingness is preserved as NaN (XGBoost's native missing-value handling),
never silently converted to a fake physiologic zero.
"""

from __future__ import annotations

import math
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np

NAN = float("nan")

# ---------------------------------------------------------------------------
# Feature groups, matching Part 15's ablation spec exactly:
#   A: existing canonical features only (value + mask + tslo + statics)
#   B: + elapsed/cutoff structural features (+ current baseline SOFA, a
#      single "where am I now" scalar, grouped with structural facts)
#   C: + trend/delta features (per-channel deltas/slopes + SOFA trend)
#   D: + rolling summary features (per-channel windowed mean/min/max/std)
#   E: + missingness/TSLO summaries (per-channel/overall observation density)
#   F: + support-history features (duration-in-state, transitions)
# ---------------------------------------------------------------------------
GROUP_ORDER = ("A", "B", "C", "D", "E", "F")


def _slope(x: Sequence[float], y: Sequence[float]) -> float:
    xs = np.asarray(x, dtype=float)
    ys = np.asarray(y, dtype=float)
    mask = ~np.isnan(ys)
    if mask.sum() < 2:
        return NAN
    xs, ys = xs[mask], ys[mask]
    if np.allclose(xs, xs[0]):
        return NAN
    slope, _ = np.polyfit(xs, ys, 1)
    return float(slope)


def _channel_series(history_values: Sequence[Sequence[float]], channel_index: int) -> List[float]:
    return [row[channel_index] if row[channel_index] is not None else NAN for row in history_values]


def build_group_a(row: Mapping[str, object]) -> Dict[str, float]:
    names = row["temporal_feature_names"]
    values = row["history_values"]
    mask = row["observation_mask"]
    tslo = row["tslo_hours"]
    features: Dict[str, float] = {}
    for bin_index in range(8):
        for channel_index, name in enumerate(names):
            value = values[bin_index][channel_index]
            features[f"A__bin{bin_index}__{name}__value"] = NAN if value is None else float(value)
            features[f"A__bin{bin_index}__{name}__mask"] = float(bool(mask[bin_index][channel_index]))
            features[f"A__bin{bin_index}__{name}__tslo"] = float(tslo[bin_index][channel_index])
    # The pre-split scientific package stores RAW statics (age_years,
    # sex_category, cardiac_condition_group), not the production
    # preprocessor's already-encoded (z-scored age + one-hot categoricals)
    # vector. For a tree-based diagnostic model, raw age needs no scaling;
    # categoricals are one-hot encoded here (diagnostic-only encoding, not
    # the frozen production preprocessor).
    for static_name, static_value in zip(row["static_feature_names"], row["static_features"]):
        if isinstance(static_value, str):
            features[f"A__static__{static_name}=={static_value}"] = 1.0
        else:
            features[f"A__static__{static_name}"] = float(static_value)
    return features


def build_group_b(row: Mapping[str, object], *, baseline_sofa) -> Dict[str, float]:
    return {
        "B__elapsed_episode_hours_at_t": float(row["elapsed_episode_hours_at_t"]),
        "B__cutoff_index": float(row["cutoff_index"]),
        "B__hours_since_first_eligible_cutoff": float(row["hours_since_first_eligible_cutoff"]),
        "B__current_sofa": NAN if baseline_sofa is None else float(baseline_sofa),
    }


def build_group_c(row: Mapping[str, object], *, sofa_trend: Mapping[str, object]) -> Dict[str, float]:
    names = row["temporal_feature_names"]
    values = row["history_values"]
    features: Dict[str, float] = {}
    lag_bins = {"6h": 1, "12h": 2, "24h": 4, "48h": 7}
    for channel_index, name in enumerate(names):
        series = _channel_series(values, channel_index)
        last = series[-1]
        for label, lag in lag_bins.items():
            past_index = len(series) - 1 - lag
            past = series[past_index] if past_index >= 0 else NAN
            features[f"C__{name}__diff_last_minus_{label}"] = (
                NAN if (np.isnan(last) or np.isnan(past)) else float(last - past)
            )
        for label, span in (("12h", 2), ("24h", 4), ("48h", 8)):
            window = series[-span:]
            features[f"C__{name}__slope_{label}"] = _slope(range(len(window)), window)
    current = sofa_trend.get("t")
    for lag_label in ("t_minus_6h", "t_minus_12h", "t_minus_24h", "t_minus_48h"):
        past = sofa_trend.get(lag_label)
        features[f"C__sofa__diff_{lag_label}"] = (
            NAN if (current is None or past is None) else float(current - past)
        )
    known_points = [(lag, sofa_trend.get(key)) for lag, key in ((0, "t"), (-6, "t_minus_6h"), (-12, "t_minus_12h"), (-24, "t_minus_24h"), (-48, "t_minus_48h"))]
    xs = [lag for lag, value in known_points if value is not None]
    ys = [value for lag, value in known_points if value is not None]
    features["C__sofa__slope"] = _slope(xs, ys) if len(xs) >= 2 else NAN
    return features


def build_group_d(row: Mapping[str, object]) -> Dict[str, float]:
    names = row["temporal_feature_names"]
    values = row["history_values"]
    features: Dict[str, float] = {}
    for channel_index, name in enumerate(names):
        series = np.asarray(_channel_series(values, channel_index), dtype=float)
        for label, span in (("12h", 2), ("24h", 4), ("48h", 8)):
            window = series[-span:]
            observed = window[~np.isnan(window)]
            if observed.size:
                features[f"D__{name}__mean_{label}"] = float(observed.mean())
                features[f"D__{name}__min_{label}"] = float(observed.min())
                features[f"D__{name}__max_{label}"] = float(observed.max())
                features[f"D__{name}__std_{label}"] = float(observed.std(ddof=0)) if observed.size > 1 else 0.0
            else:
                features[f"D__{name}__mean_{label}"] = NAN
                features[f"D__{name}__min_{label}"] = NAN
                features[f"D__{name}__max_{label}"] = NAN
                features[f"D__{name}__std_{label}"] = NAN
    return features


def build_group_e(row: Mapping[str, object]) -> Dict[str, float]:
    names = row["temporal_feature_names"]
    mask = row["observation_mask"]
    features: Dict[str, float] = {}
    total_observed = 0
    total_cells = 0
    for channel_index, name in enumerate(names):
        column = [mask[bin_index][channel_index] for bin_index in range(8)]
        observed_count = sum(1 for value in column if value)
        features[f"E__{name}__observed_count_8bins"] = float(observed_count)
        features[f"E__{name}__observed_fraction_8bins"] = float(observed_count) / 8.0
        total_observed += observed_count
        total_cells += 8
    features["E__overall_observed_fraction"] = float(total_observed) / float(total_cells)
    return features


def build_group_f(row: Mapping[str, object]) -> Dict[str, float]:
    names = row["temporal_feature_names"]
    values = row["history_values"]
    v_idx = names.index("vasopressor_on")
    i_idx = names.index("invasive_ventilation_on")
    features: Dict[str, float] = {}
    for label, channel_index in (("vasopressor", v_idx), ("invasive_ventilation", i_idx)):
        column = [bool(values[b][channel_index]) for b in range(8)]
        current = column[-1]
        duration = 0
        for state in reversed(column):
            if state == current:
                duration += 1
            else:
                break
        transitions = sum(1 for i in range(1, 8) if column[i] != column[i - 1])
        time_since_last_transition = next((i for i in range(1, 8) if column[-1 - i] != current), 8)
        features[f"F__{label}__current_state"] = float(current)
        features[f"F__{label}__duration_in_state_bins"] = float(duration)
        features[f"F__{label}__transitions_in_window"] = float(transitions)
        features[f"F__{label}__bins_since_last_transition"] = float(time_since_last_transition)
    return features


def feature_group_of(name: str) -> str:
    return name.split("__", 1)[0]


class Phase1FeatureError(ValueError):
    """Raised when a row's canonical bin arrays are not the frozen 8-bin
    shape -- fail closed rather than silently reading a malformed or
    tampered extra bin as if it were the true last-observed value."""


def _validate_bin_shape(row: Mapping[str, object]) -> None:
    n_channels = len(row["temporal_feature_names"])
    for field in ("history_values", "observation_mask", "tslo_hours"):
        values = row[field]
        if len(values) != 8:
            raise Phase1FeatureError(f"{field} must have exactly 8 bins, got {len(values)}")
        for bin_row in values:
            if len(bin_row) != n_channels:
                raise Phase1FeatureError(f"{field} bin must have exactly {n_channels} channels, got {len(bin_row)}")


def build_all_groups(row: Mapping[str, object], *, baseline_sofa, sofa_trend: Mapping[str, object]) -> Dict[str, float]:
    _validate_bin_shape(row)
    features: Dict[str, float] = {}
    features.update(build_group_a(row))
    features.update(build_group_b(row, baseline_sofa=baseline_sofa))
    features.update(build_group_c(row, sofa_trend=sofa_trend))
    features.update(build_group_d(row))
    features.update(build_group_e(row))
    features.update(build_group_f(row))
    return features


def select_groups(features: Mapping[str, float], groups: Sequence[str]) -> Dict[str, float]:
    allowed = set(groups)
    return {name: value for name, value in features.items() if feature_group_of(name) in allowed}


def cumulative_groups(up_to: str) -> Tuple[str, ...]:
    index = GROUP_ORDER.index(up_to)
    return GROUP_ORDER[: index + 1]
