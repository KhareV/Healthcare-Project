from evaluation.threshold import (
    apply_support_threshold,
    choose_support_threshold,
    load_threshold_artifact,
    save_threshold_artifact,
)
from evaluation.calibration_validation import CalibrationValidationError
import pytest
from calibration_helpers import (
    fitted,
    identity_calibrator,
    persist_calibrator,
    threshold_policy,
)


def weighted_fixture_rows():
    rows = [
        {"row_id": "a1", "stay_id": "A", "organ_support_eligible": True, "raw_probability": 0.6, "label": 1},
        {"row_id": "bpos", "stay_id": "B", "organ_support_eligible": True, "raw_probability": 0.9, "label": 1},
    ]
    rows.extend(
        {"row_id": "bn{}".format(index), "stay_id": "B", "organ_support_eligible": True, "raw_probability": 0.7, "label": 0}
        for index in range(9)
    )
    return rows


def test_threshold_maximizes_stay_balanced_validation_f1(tmp_path):
    _, selected, _, calibrator = fitted(tmp_path, rows=weighted_fixture_rows())
    calibrator = identity_calibrator(calibrator)
    saved = persist_calibrator(tmp_path, calibrator)
    result = choose_support_threshold(
        selected,
        calibrator,
        calibrator_ref=saved["artifact_ref"],
        calibrator_sha256=saved["artifact_sha256"],
        policy=threshold_policy(),
    )
    assert result.threshold == 0.6
    assert result.validation_metrics["f1"] > 0.7


def test_threshold_comparator_is_greater_than_or_equal(tmp_path):
    assert apply_support_threshold((0.49, 0.5, 0.51), 0.5) == (0, 1, 1)


def test_threshold_artifact_roundtrip(tmp_path):
    _, selected, _, calibrator = fitted(tmp_path, rows=weighted_fixture_rows())
    calibrator = identity_calibrator(calibrator)
    calibrator_saved = persist_calibrator(tmp_path, calibrator)
    result = choose_support_threshold(
        selected,
        calibrator,
        calibrator_ref=calibrator_saved["artifact_ref"],
        calibrator_sha256=calibrator_saved["artifact_sha256"],
        policy=threshold_policy(),
    )
    saved = save_threshold_artifact(tmp_path / "threshold_support_v1.json", result)
    loaded = load_threshold_artifact(
        saved["artifact_ref"], expected_sha256=saved["artifact_sha256"]
    )
    assert loaded.threshold == result.threshold
    assert apply_support_threshold((0.4, 0.6, 0.8), loaded.threshold) == (0, 1, 1)


def test_unfrozen_tie_policy_blocks_an_exact_f1_tie(tmp_path):
    rows = [
        {"row_id": "p", "stay_id": "A", "organ_support_eligible": True, "raw_probability": 0.2, "label": 1},
        {"row_id": "n", "stay_id": "B", "organ_support_eligible": True, "raw_probability": 0.8, "label": 0},
    ]
    _, selected, _, calibrator = fitted(tmp_path, rows=rows)
    calibrator = identity_calibrator(calibrator)
    saved = persist_calibrator(tmp_path, calibrator)
    with pytest.raises(CalibrationValidationError, match="TIE POLICY REQUIRED"):
        choose_support_threshold(
            selected,
            calibrator,
            calibrator_ref=saved["artifact_ref"],
            calibrator_sha256=saved["artifact_sha256"],
            policy=threshold_policy(
                candidate_method="unique_calibrated_probabilities_plus_endpoints",
                tie_policy="block",
            ),
        )
