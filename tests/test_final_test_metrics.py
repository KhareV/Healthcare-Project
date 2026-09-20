import math

import pytest

from evaluation.final_test import (
    FinalPredictionBundle,
    ICULogPredictionRecord,
    evaluate_frozen_predictions,
)
from final_test_helpers import fake_calibrator, record


def test_raw_recovery_and_common_icu_hours_path_are_reused():
    bundle = FinalPredictionBundle(
        recovery24=(record("A", 1, -2, -1), record("B", 2, 2, 1)),
        recovery48=(record("A", 1, -3, -1), record("B", 2, 3, 2)),
        icu_time_log=(
            ICULogPredictionRecord("A", "t1", 1.0, -4.0, True),
            ICULogPredictionRecord("B", "t2", 3.0, math.log1p(3.0), True),
        ),
        support_raw=(record("A", 1, 0, 0.2), record("B", 2, 1, 0.8)),
    )
    result = evaluate_frozen_predictions(
        bundle,
        support_calibrator=fake_calibrator(),
        support_threshold=0.5,
        threshold_identifier="frozen-threshold",
    )
    assert result.recovery["24h"].metadata["evaluation_domain"] == "raw_sofa_delta_unclipped"
    assert [row.prediction for row in result.icu_time_hours_records] == pytest.approx([0.0, 3.0])
    assert result.icu_time.metrics["median_absolute_error"] == pytest.approx(0.0)
