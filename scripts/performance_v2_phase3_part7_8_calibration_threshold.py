"""Performance-v2 Phase 3, Parts 7-8: final V2 support isotonic calibrator
+ operating threshold, fit ONLY from the cross-fitted DEV OOF raw
probabilities/labels (Part 5). No fresh test. No method comparison -- this
is the final, frozen v2 calibration method per the Phase-3 spec.
"""

import json
import sys
from pathlib import Path

import numpy as np
import sklearn
from sklearn.isotonic import IsotonicRegression

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.synthetic.config import canonical_json_bytes  # noqa: E402
from evaluation.weights import compute_stay_weights  # noqa: E402
from vedant_infra.hashing import sha256_file  # noqa: E402

CAL_DIR = ROOT / "artifacts/performance_v2/phase3/calibration"
THRESH_DIR = ROOT / "artifacts/performance_v2/phase3/thresholds"


def _load_oof():
    rows = []
    with (CAL_DIR / "support_dev_oof_raw_predictions_v1.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _weighted_f1(labels, predictions, weights):
    labels = np.asarray(labels)
    predictions = np.asarray(predictions)
    weights = np.asarray(weights)
    tp = float(np.sum(weights[(labels == 1) & (predictions == 1)]))
    fp = float(np.sum(weights[(labels == 0) & (predictions == 1)]))
    fn = float(np.sum(weights[(labels == 1) & (predictions == 0)]))
    if tp + fp == 0 or tp + fn == 0:
        return 0.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def main() -> None:
    oof_rows = _load_oof()
    oof_path = CAL_DIR / "support_dev_oof_raw_predictions_v1.jsonl"
    oof_sha256 = sha256_file(oof_path)

    labels = np.asarray([row["label"] for row in oof_rows], dtype=np.float64)
    raw = np.asarray([row["raw_probability"] for row in oof_rows], dtype=np.float64)
    stay_ids = [row["stay_id"] for row in oof_rows]
    weights_result = compute_stay_weights(stay_ids, [True] * len(oof_rows))
    weights = np.asarray(weights_result.weights, dtype=np.float64)

    n_stays = len({row["stay_id"] for row in oof_rows})
    n_positive = int(labels.sum())
    n_positive_stays = len({row["stay_id"] for row in oof_rows if row["label"] == 1})
    weighted_positive = float(weights[labels == 1].sum())
    weighted_negative = float(weights[labels == 0].sum())

    calibrator = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
    calibrator.fit(raw, labels, sample_weight=weights)
    calibrated = calibrator.predict(raw)

    CAL_DIR.mkdir(parents=True, exist_ok=True)
    calibrated_path = CAL_DIR / "support_dev_oof_calibrated_predictions_v1.jsonl"
    with calibrated_path.open("w", encoding="utf-8") as handle:
        for row, cal in zip(oof_rows, calibrated):
            enriched = dict(row)
            enriched["calibrated_probability"] = float(cal)
            handle.write(json.dumps(enriched, sort_keys=True) + "\n")
    calibrated_sha256 = sha256_file(calibrated_path)

    x_knots = calibrator.X_thresholds_.tolist() if hasattr(calibrator, "X_thresholds_") else calibrator.f_.x.tolist()
    y_knots = calibrator.y_thresholds_.tolist() if hasattr(calibrator, "y_thresholds_") else calibrator.f_.y.tolist()

    calibrator_payload = {
        "status": "V2_SUPPORT_CALIBRATOR_FROZEN",
        "artifact_version": "isotonic_support_v2",
        "method": "sklearn.isotonic.IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip')",
        "sklearn_version": sklearn.__version__,
        "source_oof_artifact": "artifacts/performance_v2/phase3/calibration/support_dev_oof_raw_predictions_v1.jsonl",
        "source_oof_sha256": oof_sha256,
        "n_eligible_rows": len(oof_rows), "n_eligible_stays": n_stays,
        "n_positive_examples": n_positive, "n_positive_stays": n_positive_stays,
        "weighted_class_counts": {"weighted_positive": weighted_positive, "weighted_negative": weighted_negative},
        "fit_weights": "stay_balanced (each eligible stay contributes total weight 1 -- evaluation.weights.compute_stay_weights)",
        "calibration_knots": {"x_thresholds": x_knots, "y_thresholds": y_knots},
        "calibrated_output_artifact": "artifacts/performance_v2/phase3/calibration/support_dev_oof_calibrated_predictions_v1.jsonl",
        "calibrated_output_sha256": calibrated_sha256,
    }
    calibrator_path = CAL_DIR / "isotonic_support_v2.json"
    calibrator_path.write_bytes(canonical_json_bytes(calibrator_payload))
    print("wrote", calibrator_path, sha256_file(calibrator_path))
    print("wrote", calibrated_path, calibrated_sha256)

    # --- Part 8: threshold ---
    candidates = sorted(set(calibrated.tolist()) | {0.0, 1.0})
    best = None
    for threshold in candidates:
        predictions = (calibrated >= threshold).astype(int)
        f1 = _weighted_f1(labels, predictions, weights)
        candidate = {"threshold": threshold, "f1": f1, "distance_from_half": abs(threshold - 0.5)}
        if best is None:
            best = candidate
        else:
            if f1 > best["f1"] + 1e-15:
                best = candidate
            elif abs(f1 - best["f1"]) <= 1e-15:
                if candidate["distance_from_half"] < best["distance_from_half"] - 1e-15:
                    best = candidate
                elif abs(candidate["distance_from_half"] - best["distance_from_half"]) <= 1e-15 and candidate["threshold"] > best["threshold"]:
                    best = candidate

    predictions_at_best = (calibrated >= best["threshold"]).astype(int)
    tp = float(np.sum(weights[(labels == 1) & (predictions_at_best == 1)]))
    fp = float(np.sum(weights[(labels == 0) & (predictions_at_best == 1)]))
    tn = float(np.sum(weights[(labels == 0) & (predictions_at_best == 0)]))
    fn = float(np.sum(weights[(labels == 1) & (predictions_at_best == 0)]))
    threshold_payload = {
        "status": "V2_SUPPORT_THRESHOLD_FROZEN",
        "artifact_version": "support_threshold_v2",
        "criterion": "maximize stay-balanced DEV F1 on cross-fitted OOF calibrated probabilities",
        "candidate_set": "sorted unique calibrated OOF probabilities plus {0.0, 1.0}",
        "tie_break": ["highest full-precision weighted F1", "closest to 0.5", "higher threshold"],
        "threshold_value": best["threshold"], "comparator": ">=",
        "dev_weighted_f1": best["f1"],
        "weighted_confusion": {"TP": tp, "FP": fp, "TN": tn, "FN": fn},
        "source_calibrated_oof_artifact": "artifacts/performance_v2/phase3/calibration/support_dev_oof_calibrated_predictions_v1.jsonl",
        "source_calibrated_oof_sha256": calibrated_sha256,
        "source_calibrator_artifact": "artifacts/performance_v2/phase3/calibration/isotonic_support_v2.json",
        "source_calibrator_sha256": sha256_file(calibrator_path),
    }
    THRESH_DIR.mkdir(parents=True, exist_ok=True)
    threshold_path = THRESH_DIR / "support_threshold_v2.json"
    threshold_path.write_bytes(canonical_json_bytes(threshold_payload))
    print("wrote", threshold_path, sha256_file(threshold_path))
    print("threshold", best["threshold"], "dev_weighted_f1", best["f1"])


if __name__ == "__main__":
    main()
