from evaluation.final_test import apply_frozen_support_postprocessing
from final_test_helpers import fake_calibrator, record


def test_support_preserves_raw_and_uses_transform_only_global_threshold():
    rows = (record("A", 1, 0, 0.2), record("B", 2, 1, 0.8))
    raw_before = tuple(row.prediction for row in rows)
    calibrated, alerts = apply_frozen_support_postprocessing(
        rows, fake_calibrator(), threshold=0.5
    )
    assert tuple(row.prediction for row in rows) == raw_before
    assert tuple(row.prediction for row in calibrated) == raw_before
    assert alerts == (False, True)
