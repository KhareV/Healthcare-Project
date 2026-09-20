import math
import unittest

import torch

from evaluation.metrics import PredictionRecord, evaluate_icu_time, evaluate_recovery
from models.icu_time_postprocess import remaining_icu_hours_from_log_prediction


def record(stay, time, target, prediction, eligible=True):
    return PredictionRecord(stay, time, target, prediction, eligible)


class RegressionMetricTests(unittest.TestCase):
    def test_hostile_unequal_snapshot_mae_is_five_not_row_mae(self):
        rows = [
            record("A", "t0", 0.0, 10.0),
            record("B", "t0", 0.0, 0.0),
            record("B", "t1", 0.0, 0.0),
            record("B", "t2", 0.0, 0.0),
        ]
        result = evaluate_recovery(rows, rows)["24h"]
        self.assertEqual(result.metrics["mae"], 5.0)
        self.assertNotEqual(result.metrics["mae"], 2.5)
        self.assertEqual(result.counts, {"N_examples": 4, "N_ICU_stays": 2})

    def test_recovery_horizons_are_separate_and_raw_delta_is_unclipped(self):
        rows24 = [record("A", "t0", 2.0, 5.0)]
        rows48 = [record("A", "t0", -2.0, -3.0)]
        before = evaluate_recovery(rows24, rows48)
        changed48 = [record("A", "t0", -2.0, 100.0)]
        after = evaluate_recovery(rows24, changed48)
        self.assertEqual(before["24h"].metrics, after["24h"].metrics)
        self.assertEqual(before["24h"].metrics["mae"], 3.0)
        self.assertEqual(before["24h"].metadata["evaluation_domain"], "raw_sofa_delta_unclipped")

    def test_direction_is_computed_for_nonzero_and_blocked_for_zero(self):
        nonzero = [
            record("A", "t0", 2.0, 1.0),
            record("B", "t0", -2.0, 1.0),
        ]
        result = evaluate_recovery(nonzero, nonzero)["24h"]
        self.assertEqual(result.metrics["directional_agreement"], 0.5)
        zero = [record("A", "t0", 0.0, 1.0)]
        result = evaluate_recovery(zero, zero)["24h"]
        self.assertIsNone(result.metrics["directional_agreement"])
        self.assertIn("ZERO/TIE", result.metadata["blockers"][0])

    def test_icu_time_evaluates_postprocessed_hours(self):
        raw_log = torch.tensor([[-2.0], [math.log1p(9.0)]])
        predicted_hours = remaining_icu_hours_from_log_prediction(raw_log).flatten().tolist()
        rows = [
            record("A", "t0", 2.0, predicted_hours[0]),
            record("B", "t0", 10.0, predicted_hours[1]),
        ]
        result = evaluate_icu_time(rows, percentile_levels=(0.5, 1.0))
        self.assertAlmostEqual(result.metrics["mae"], 1.5)
        self.assertAlmostEqual(result.metrics["median_absolute_error"], 1.0)
        self.assertEqual(result.metadata["evaluation_domain"], "remaining_current_icu_hours")


if __name__ == "__main__":
    unittest.main()
