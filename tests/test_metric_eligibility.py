import unittest

from evaluation.metrics import (
    EmptyEligibleSetError,
    MetricInputError,
    PredictionRecord,
    evaluate_organ_support,
    evaluate_recovery,
)


class MetricEligibilityTests(unittest.TestCase):
    def test_ineligible_attack_cannot_change_regression_metrics_or_counts(self):
        clean = [
            PredictionRecord("A", "t0", 1.0, 2.0, True),
            PredictionRecord("A", "t1", float("nan"), float("nan"), False),
        ]
        attacked = [clean[0], PredictionRecord("A", "t1", 1_000_000, -1_000_000, False)]
        self.assertEqual(
            evaluate_recovery(clean, clean)["24h"],
            evaluate_recovery(attacked, attacked)["24h"],
        )

    def test_ineligible_attack_cannot_change_classification_metrics(self):
        clean = [
            PredictionRecord("A", "t0", 1.0, 0.8, True),
            PredictionRecord("B", "t0", 0.0, 0.2, True),
            PredictionRecord("B", "t1", float("nan"), float("nan"), False),
        ]
        attacked = clean[:2] + [PredictionRecord("B", "t1", 999999, -999999, False)]
        self.assertEqual(evaluate_organ_support(clean), evaluate_organ_support(attacked))

    def test_empty_eligible_set_fails_explicitly(self):
        rows = [PredictionRecord("A", "t0", float("nan"), float("nan"), False)]
        with self.assertRaises(EmptyEligibleSetError):
            evaluate_recovery(rows, rows)

    def test_eligible_nan_infinity_and_duplicate_key_fail(self):
        for value in (float("nan"), float("inf")):
            rows = [PredictionRecord("A", "t0", value, 0.0, True)]
            with self.assertRaises(MetricInputError):
                evaluate_recovery(rows, rows)
        duplicates = [
            PredictionRecord("A", "t0", 0.0, 0.0, True),
            PredictionRecord("A", "t0", 0.0, 0.0, True),
        ]
        with self.assertRaises(MetricInputError):
            evaluate_recovery(duplicates, duplicates)


if __name__ == "__main__":
    unittest.main()
