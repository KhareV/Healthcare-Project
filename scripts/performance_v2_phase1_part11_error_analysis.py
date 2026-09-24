"""Performance-v2 Phase 1, Part 11: validation-only error analysis for the
CURRENT selected v1 models, using their already-frozen validation
prediction files (no retraining, no re-inference).

DEVELOPMENT_DIAGNOSTIC_ONLY. VALIDATION only, never TEST.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.synthetic.config import canonical_json_bytes  # noqa: E402
from experiments.stage5_final_test import _observation_density  # noqa: E402
from performance_v2.data_loading import load_dev_rows  # noqa: E402
from vedant_infra.hashing import sha256_file  # noqa: E402

OUT_DIR = ROOT / "artifacts/performance_v2/phase1"


def _load_jsonl(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _severity_band(sofa):
    if sofa is None:
        return "unknown"
    return "low" if sofa <= 3 else ("mid" if sofa <= 7 else "high")


def _slice_summary(joined, error_fn, slice_fn):
    buckets = defaultdict(list)
    for item in joined:
        buckets[slice_fn(item)].append(error_fn(item))
    return {
        str(key): {"n": len(errors), "mean_abs_error": float(np.mean(np.abs(errors)))}
        for key, errors in buckets.items()
    }


def _join(predictions_rows, dev_rows_by_key, extra_fields_fn):
    joined = []
    for pred in predictions_rows:
        key = (pred["stay_id"], pred["prediction_time"])
        dev_row = dev_rows_by_key.get(key)
        if dev_row is None:
            continue
        item = dict(pred)
        item.update(extra_fields_fn(dev_row))
        joined.append(item)
    return joined


def main() -> None:
    validation_rows = load_dev_rows(ROOT, splits=("validation",))
    dev_by_key = {(row["stay_id"], row["prediction_time"]): row for row in validation_rows}

    def extra(row):
        return {
            "cutoff_index": row["cutoff_index"],
            "elapsed_episode_hours_at_t": row["elapsed_episode_hours_at_t"],
            "cardiac_condition_group": row["cardiac_condition_group"],
            "baseline_sofa": row.get("baseline_sofa"),
            "observation_density": _observation_density(row),
        }

    result = {"status": "DEVELOPMENT_DIAGNOSTIC_ONLY", "artifact_version": "performance_v2_phase1_error_analysis_v1", "scope": "validation_partition_only"}

    # Recovery 24h
    for horizon, path in (
        ("24", "artifacts/search/xgb/phase12/recovery/xgb-recovery-014/attempt-1/validation_predictions_24h.jsonl"),
        ("48", "artifacts/search/xgb/phase12/recovery/xgb-recovery-014/attempt-1/validation_predictions_48h.jsonl"),
    ):
        rows = [r for r in _load_jsonl(ROOT / path) if r.get("eligible")]
        joined = _join(rows, dev_by_key, extra)

        def error(item):
            return item["prediction"] - item["true_target"]

        signed = [item["true_target"] for item in joined]
        entry = {
            "n": len(joined),
            "by_cutoff_index": _slice_summary(joined, error, lambda item: item["cutoff_index"]),
            "by_elapsed_hours_quartile": _slice_summary(
                joined, error,
                lambda item, edges=np.percentile([j["elapsed_episode_hours_at_t"] for j in joined], [25, 50, 75]):
                    int(np.searchsorted(edges, item["elapsed_episode_hours_at_t"])),
            ),
            "by_cardiac_subtype": _slice_summary(joined, error, lambda item: item["cardiac_condition_group"]),
            "by_baseline_sofa_band": _slice_summary(joined, error, lambda item: _severity_band(item["baseline_sofa"])),
            "by_true_target_sign": _slice_summary(
                joined, error,
                lambda item: "negative" if item["true_target"] < 0 else ("zero" if item["true_target"] == 0 else "positive"),
            ),
        }
        result["recovery" + horizon] = entry

    # ICU
    icu_rows_raw = [r for r in _load_jsonl(ROOT / "artifacts/search/gru/final_v2/icu_time/gru-icu-time-026/attempt-1/validation_predictions.jsonl") if r.get("eligible")]
    joined = _join(icu_rows_raw, dev_by_key, extra)

    def icu_error(item):
        return item["prediction"] - item["true_target"]

    remaining_quartile_edges = np.percentile([j["true_target"] for j in joined], [25, 50, 75])
    elapsed_quartile_edges = np.percentile([j["elapsed_episode_hours_at_t"] for j in joined], [25, 50, 75])
    result["icu_stay_time"] = {
        "n": len(joined),
        "by_cutoff_index": _slice_summary(joined, icu_error, lambda item: item["cutoff_index"]),
        "by_remaining_time_quartile": _slice_summary(joined, icu_error, lambda item: int(np.searchsorted(remaining_quartile_edges, item["true_target"]))),
        "by_elapsed_time_quartile": _slice_summary(joined, icu_error, lambda item: int(np.searchsorted(elapsed_quartile_edges, item["elapsed_episode_hours_at_t"]))),
        "by_cardiac_subtype": _slice_summary(joined, icu_error, lambda item: item["cardiac_condition_group"]),
    }

    # Support (descriptive FP/FN pattern at the frozen-equivalent threshold
    # is not meaningful here since this raw-probability GRU sensitivity
    # model was never calibrated/thresholded; report raw-probability error
    # magnitude by slice instead, consistent with the other tasks).
    support_rows_raw = [r for r in _load_jsonl(ROOT / "artifacts/search/xgb/phase12/organ_support/xgb-support-024/attempt-1/validation_predictions.jsonl") if r.get("eligible")]
    joined = _join(support_rows_raw, dev_by_key, extra)

    def support_error(item):
        return item["prediction"] - item["true_target"]

    result["organ_support"] = {
        "n": len(joined),
        "by_cutoff_index": _slice_summary(joined, support_error, lambda item: item["cutoff_index"]),
        "by_cardiac_subtype": _slice_summary(joined, support_error, lambda item: item["cardiac_condition_group"]),
        "by_baseline_sofa_band": _slice_summary(joined, support_error, lambda item: _severity_band(item["baseline_sofa"])),
        "by_true_label": _slice_summary(joined, support_error, lambda item: "positive" if item["true_target"] == 1 else "negative"),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "validation_error_analysis_v1.json"
    out_path.write_bytes(canonical_json_bytes(result))
    print("wrote", out_path, sha256_file(out_path))


if __name__ == "__main__":
    main()
