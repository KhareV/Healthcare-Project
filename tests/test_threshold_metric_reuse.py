from evaluation.metrics import weighted_confusion_metrics
from evaluation.threshold import choose_support_threshold
from evaluation.weights import compute_stay_weights
from calibration_helpers import fitted, identity_calibrator, persist_calibrator, threshold_policy
from test_support_threshold import weighted_fixture_rows


def test_threshold_metrics_equal_authoritative_phase9_function(tmp_path):
    rows = weighted_fixture_rows()
    _, selected, _, calibrator = fitted(tmp_path, rows=rows)
    calibrator = identity_calibrator(calibrator)
    saved = persist_calibrator(tmp_path, calibrator)
    result = choose_support_threshold(
        selected,
        calibrator,
        calibrator_ref=saved["artifact_ref"],
        calibrator_sha256=saved["artifact_sha256"],
        policy=threshold_policy(),
    )
    labels = tuple(row["label"] for row in rows)
    probabilities = tuple(row["raw_probability"] for row in rows)
    weights = compute_stay_weights(tuple(row["stay_id"] for row in rows), (True,) * len(rows))
    expected = weighted_confusion_metrics(labels, probabilities, weights.weights, result.threshold)
    assert result.validation_metrics == expected
    assert result.metric_implementation_version == "stay_balanced_metrics_v1"


def test_stay_balanced_optimum_differs_from_ordinary_row_optimum(tmp_path):
    rows = weighted_fixture_rows()
    _, selected, _, calibrator = fitted(tmp_path, rows=rows)
    calibrator = identity_calibrator(calibrator)
    saved = persist_calibrator(tmp_path, calibrator)
    result = choose_support_threshold(
        selected,
        calibrator,
        calibrator_ref=saved["artifact_ref"],
        calibrator_sha256=saved["artifact_sha256"],
        policy=threshold_policy(),
    )
    labels = tuple(row["label"] for row in rows)
    probabilities = tuple(row["raw_probability"] for row in rows)
    row_scores = {
        threshold: weighted_confusion_metrics(
            labels, probabilities, (1.0,) * len(rows), threshold
        )["f1"]
        for threshold in set(probabilities)
    }
    ordinary_best = max(row_scores, key=row_scores.get)
    assert ordinary_best == 0.9
    assert result.threshold == 0.6
