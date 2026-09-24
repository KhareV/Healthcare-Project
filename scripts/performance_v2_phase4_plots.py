"""Performance-v2 Phase 4, Part 21: publication-ready plots generated only
from saved prediction/metric artifacts. No new model inference.
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vedant_infra.hashing import sha256_file  # noqa: E402

PRED_DIR = ROOT / "artifacts/performance_v2/phase4/predictions"
PLOTS_DIR = ROOT / "artifacts/performance_v2/phase4/plots"


def _load_jsonl(path):
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _save(fig, name):
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    path = PLOTS_DIR / name
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path, sha256_file(path), path.stat().st_size, "bytes")


def scatter_and_error(target, prediction, title, unit, name_prefix):
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.scatter(target, prediction, s=4, alpha=0.25, linewidths=0)
    lo, hi = min(min(target), min(prediction)), max(max(target), max(prediction))
    ax.plot([lo, hi], [lo, hi], color="black", linewidth=0.8)
    ax.set_xlabel(f"True {unit}")
    ax.set_ylabel(f"Predicted {unit}")
    ax.set_title(title)
    _save(fig, f"{name_prefix}_scatter.png")

    errors = np.abs(np.asarray(prediction) - np.asarray(target))
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    ax.hist(errors, bins=40, color="steelblue")
    ax.set_xlabel(f"Absolute error ({unit})")
    ax.set_ylabel("Row count")
    ax.set_title(f"{title}: |error| distribution")
    _save(fig, f"{name_prefix}_abs_error_hist.png")


def main() -> None:
    rec24 = _load_jsonl(PRED_DIR / "recovery24_predictions_v1.jsonl")
    rec48 = _load_jsonl(PRED_DIR / "recovery48_predictions_v1.jsonl")
    icu = _load_jsonl(PRED_DIR / "icu_stay_time_predictions_v1.jsonl")
    support = _load_jsonl(PRED_DIR / "organ_support_predictions_v1.jsonl")

    scatter_and_error([r["target_delta24"] for r in rec24], [r["prediction_delta24"] for r in rec24], "Recovery +24h (fresh V2 test)", "ΔSOFA24", "recovery24")
    scatter_and_error([r["target_delta48"] for r in rec48], [r["prediction_delta48"] for r in rec48], "Recovery +48h (fresh V2 test)", "ΔSOFA48", "recovery48")
    scatter_and_error([r["true_remaining_hours"] for r in icu], [r["prediction_hours"] for r in icu], "ICU remaining time (fresh V2 test)", "hours", "icu_stay_time")

    labels = np.asarray([r["label"] for r in support], dtype=float)
    raw = np.asarray([r["raw_probability"] for r in support], dtype=float)
    calibrated = np.asarray([r["calibrated_probability"] for r in support], dtype=float)

    def roc_points(y, p):
        order = np.argsort(-p)
        y_sorted = y[order]
        tp = np.cumsum(y_sorted)
        fp = np.cumsum(1 - y_sorted)
        tpr = tp / tp[-1]
        fpr = fp / fp[-1]
        return np.concatenate([[0], fpr]), np.concatenate([[0], tpr])

    def pr_points(y, p):
        order = np.argsort(-p)
        y_sorted = y[order]
        tp = np.cumsum(y_sorted)
        precision = tp / np.arange(1, len(y_sorted) + 1)
        recall = tp / tp[-1]
        return recall, precision

    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    for name, probs in (("raw", raw), ("calibrated", calibrated)):
        fpr, tpr = roc_points(labels, probs)
        ax.plot(fpr, tpr, label=name)
    ax.plot([0, 1], [0, 1], color="gray", linewidth=0.6, linestyle="--")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Organ support: ROC (fresh V2 test)")
    ax.legend()
    _save(fig, "organ_support_roc.png")

    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    for name, probs in (("raw", raw), ("calibrated", calibrated)):
        recall, precision = pr_points(labels, probs)
        ax.plot(recall, precision, label=name)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Organ support: precision-recall (fresh V2 test)")
    ax.legend()
    _save(fig, "organ_support_precision_recall.png")

    calibration_evidence = json.loads((ROOT / "artifacts/performance_v2/phase4/metrics/calibration_evidence_v2.json").read_text())
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    for key, label in (("raw_reliability_table", "raw"), ("calibrated_reliability_table", "calibrated")):
        table = calibration_evidence[key]
        xs = [b["mean_predicted"] for b in table if b["n"] > 0]
        ys = [b["observed_frequency"] for b in table if b["n"] > 0]
        ax.plot(xs, ys, marker="o", label=label)
    ax.plot([0, 1], [0, 1], color="gray", linewidth=0.6, linestyle="--")
    ax.set_xlabel("Mean predicted probability (bin)")
    ax.set_ylabel("Observed frequency (bin)")
    ax.set_title("Organ support: reliability diagram (fresh V2 test)")
    ax.legend()
    _save(fig, "organ_support_reliability.png")

    print("PLOTS_COMPLETE")


if __name__ == "__main__":
    main()
