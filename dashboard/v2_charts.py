"""Matplotlib-rendered chart images (base64 PNG data URIs) for the V2
dashboard. Dependency-free at the browser level (no JS charting library),
consistent with the existing dashboard's server-rendered HTML/CSS philosophy.
Every chart here is built only from already-known replay/history/metric
data passed in by the caller -- never triggers new inference itself.
"""

from __future__ import annotations

import base64
import io
from typing import Mapping, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_PALETTE = {
    "observed": "#2563eb",
    "forecast": "#dc2626",
    "naive": "#9ca3af",
    "grid": "#e5e7eb",
    "accent": "#0f766e",
}


def _to_data_uri(fig) -> str:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def recovery_trajectory_chart(
    *, history_times: Sequence[str], history_sofa: Sequence[float],
    cutoff_time: str, current_sofa: float,
    forecast_24h_time: str, forecast_24h_sofa: float,
    forecast_48h_time: str, forecast_48h_sofa: float,
) -> str:
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    if history_times:
        ax.plot(range(len(history_times)), history_sofa, marker="o", color=_PALETTE["observed"], label="Observed SOFA (<= cutoff)")
    cutoff_x = len(history_times) - 1 if history_times else 0
    ax.scatter([cutoff_x], [current_sofa], color=_PALETTE["observed"], zorder=5, s=60, edgecolor="white")
    forecast_x = [cutoff_x, cutoff_x + 4, cutoff_x + 8]
    forecast_y = [current_sofa, forecast_24h_sofa, forecast_48h_sofa]
    ax.plot(forecast_x, forecast_y, marker="s", linestyle="--", color=_PALETTE["forecast"], label="Forecast (+24h / +48h)")
    ax.axvline(cutoff_x, color="black", linewidth=0.8, linestyle=":")
    ax.axvspan(cutoff_x, cutoff_x + 8, color=_PALETTE["forecast"], alpha=0.06)
    ax.text(cutoff_x, ax.get_ylim()[1] if ax.get_ylim()[1] else 24, " cutoff", fontsize=8, va="top")
    ax.set_ylabel("SOFA score")
    ax.set_xticks([])
    ax.set_ylim(0, 24)
    ax.set_title("Recovery trajectory: observed vs. forecast")
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    ax.grid(axis="y", color=_PALETTE["grid"], linewidth=0.6)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    return _to_data_uri(fig)


def trend_chart(*, x_labels: Sequence[str], y_values: Sequence[float], title: str, y_label: str, threshold: Optional[float] = None) -> str:
    fig, ax = plt.subplots(figsize=(5.2, 2.6))
    ax.plot(range(len(y_values)), y_values, marker="o", color=_PALETTE["accent"])
    if threshold is not None:
        ax.axhline(threshold, color=_PALETTE["forecast"], linestyle="--", linewidth=1.0, label="Frozen threshold")
        ax.legend(loc="best", fontsize=8, frameon=False)
    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels([label[-8:] for label in x_labels], rotation=45, ha="right", fontsize=7)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(axis="y", color=_PALETTE["grid"], linewidth=0.6)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    return _to_data_uri(fig)


def bar_comparison_chart(*, labels: Sequence[str], values: Sequence[float], errors: Optional[Sequence[float]] = None, title: str, y_label: str) -> str:
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    colors = [_PALETTE["accent"], _PALETTE["naive"]] if len(labels) == 2 else [_PALETTE["accent"]] * len(labels)
    ax.bar(labels, values, color=colors, yerr=errors, capsize=4)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(axis="y", color=_PALETTE["grid"], linewidth=0.6)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    return _to_data_uri(fig)


def temporal_bin_heatmap(*, channel_names: Sequence[str], observation_mask: Sequence[Sequence[bool]], padding_mask: Sequence[bool]) -> str:
    import numpy as np

    n_bins = len(observation_mask)
    n_channels = len(channel_names)
    grid = np.zeros((n_channels, n_bins))
    for bin_index in range(n_bins):
        for channel_index in range(n_channels):
            if padding_mask[bin_index]:
                grid[channel_index, bin_index] = -1
            elif observation_mask[bin_index][channel_index]:
                grid[channel_index, bin_index] = 1
    fig, ax = plt.subplots(figsize=(6.0, max(3.0, 0.22 * n_channels)))
    cmap = matplotlib.colors.ListedColormap(["#d1d5db", "#fee2e2", "#16a34a"])
    ax.imshow(grid, aspect="auto", cmap=cmap, vmin=-1, vmax=1)
    ax.set_xticks(range(n_bins))
    ax.set_xticklabels([f"bin {i}\n(t-{48-6*i}h,t-{42-6*i}h]" for i in range(n_bins)], fontsize=6)
    ax.set_yticks(range(n_channels))
    ax.set_yticklabels(channel_names, fontsize=6)
    ax.set_title("48h lookback: 8 x 6h bins — observation availability")
    return _to_data_uri(fig)


def reliability_chart(*, table_raw: Sequence[Mapping[str, object]], table_calibrated: Sequence[Mapping[str, object]]) -> str:
    fig, ax = plt.subplots(figsize=(4.6, 4.6))
    for table, label, color in ((table_raw, "raw", "#9ca3af"), (table_calibrated, "calibrated", _PALETTE["accent"])):
        xs = [b["mean_predicted"] for b in table if b["n"]]
        ys = [b["observed_frequency"] for b in table if b["n"]]
        ax.plot(xs, ys, marker="o", label=label, color=color)
    ax.plot([0, 1], [0, 1], color="black", linewidth=0.6, linestyle="--")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title("Support: reliability diagram (frozen fresh-test)")
    ax.legend(fontsize=8, frameon=False)
    return _to_data_uri(fig)
