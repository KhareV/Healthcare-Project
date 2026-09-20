import math
import unittest

from evaluation.metrics import (
    MetricInputError,
    PredictionRecord,
    evaluate_organ_support,
    weighted_confusion_metrics,
)


def support_record(stay, time, label, probability, eligible=True):
    return PredictionRecord(stay, time, label, probability, eligible)


class ClassificationMetricTests(unittest.TestCase):
    def setUp(self):
        self.rows = [
            support_record("A", "t0", 1.0, 0.8),
            support_record("B", "t0", 0.0, 0.9),
            support_record("B", "t1", 0.0, 0.7),
            support_record("B", "t2", 0.0, 0.1),
        ]

    def test_weighted_auprc_auroc_brier_and_positive_counts(self):
        result = evaluate_organ_support(self.rows)
        self.assertAlmostEqual(result.metrics["auprc"], 0.75)
        self.assertAlmostEqual(result.metrics["auroc"], 2.0 / 3.0)
        self.assertAlmostEqual(result.metrics["brier"], (0.04 + 1.31 / 3.0) / 2.0)
        self.assertEqual(
            result.counts,
            {
                "N_examples": 4,
                "N_ICU_stays": 2,
                "N_positive_examples": 1,
                "N_positive_stays": 1,
            },
        )
        self.assertFalse(result.metadata["class_weight_used_for_evaluation"])

    def test_test_only_fixed_threshold_confusion_metrics(self):
        result = evaluate_organ_support(
            self.rows, fixed_threshold=0.5, threshold_identifier="TEST_THRESHOLD_0_5"
        )
        self.assertAlmostEqual(result.metrics["weighted_TP"], 1.0)
        self.assertAlmostEqual(result.metrics["weighted_FP"], 2.0 / 3.0)
        self.assertAlmostEqual(result.metrics["weighted_TN"], 1.0 / 3.0)
        self.assertAlmostEqual(result.metrics["weighted_FN"], 0.0)
        self.assertAlmostEqual(result.metrics["sensitivity"], 1.0)
        self.assertAlmostEqual(result.metrics["specificity"], 1.0 / 3.0)
        self.assertAlmostEqual(result.metrics["precision"], 0.6)
        self.assertAlmostEqual(result.metrics["f1"], 0.75)
        self.assertFalse(result.metadata["threshold_selected"])

    def test_one_class_auroc_and_zero_denominators_are_explicit_nan(self):
        rows = [support_record("A", "t0", 1.0, 0.8)]
        result = evaluate_organ_support(
            rows, fixed_threshold=0.5, threshold_identifier="TEST_ONLY"
        )
        self.assertTrue(math.isnan(result.metrics["auroc"]))
        self.assertTrue(math.isnan(result.metrics["specificity"]))
        self.assertIn("auroc", result.metadata["undefined_metrics"])

    def test_no_positive_auprc_is_nan_and_bad_probabilities_fail(self):
        no_positive = [support_record("A", "t0", 0.0, 0.2)]
        result = evaluate_organ_support(no_positive)
        self.assertTrue(math.isnan(result.metrics["auprc"]))
        for probability in (-0.1, 1.1, float("nan"), float("inf")):
            with self.assertRaises(MetricInputError):
                evaluate_organ_support([support_record("A", "t0", 1.0, probability)])

    def test_threshold_is_never_inferred(self):
        result = evaluate_organ_support(self.rows)
        self.assertNotIn("f1", result.metrics)
        with self.assertRaises(MetricInputError):
            evaluate_organ_support(self.rows, fixed_threshold=0.5)

    def test_duplicate_negative_snapshots_do_not_multiply_stay_influence(self):
        one_negative = [
            support_record("A", "t0", 1.0, 0.8),
            support_record("B", "t0", 0.0, 0.9),
        ]
        three_negatives = [one_negative[0]] + [
            support_record("B", "t{}".format(index), 0.0, 0.9)
            for index in range(3)
        ]
        first = evaluate_organ_support(one_negative)
        second = evaluate_organ_support(three_negatives)
        self.assertEqual(first.metrics, second.metrics)

    def test_public_confusion_helper_rejects_malformed_inputs(self):
        bad_arguments = (
            ([1], [0.5, 0.6], [1.0]),
            ([2], [0.5], [1.0]),
            ([1], [1.5], [1.0]),
            ([1], [0.5], [-1.0]),
            ([1], [0.5], [0.0]),
        )
        for labels, probabilities, weights in bad_arguments:
            with self.assertRaises(MetricInputError):
                weighted_confusion_metrics(labels, probabilities, weights, 0.5)


if __name__ == "__main__":
    unittest.main()
