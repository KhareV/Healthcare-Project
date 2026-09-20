import unittest

from bootstrap_helpers import TEST_BOOTSTRAP_SEED, support_record
from evaluation.bootstrap import bootstrap_organ_support


class BootstrapClassificationTests(unittest.TestCase):
    def setUp(self):
        self.records = [
            support_record("P", "t0", 1, 0.9),
            support_record("N", "t0", 0, 0.1),
        ]

    def test_single_class_auroc_replicates_are_skipped_and_counted(self):
        forced = [
            ["P", "N"],
            ["N", "P"],
            ["P", "N"],
            ["N", "P"],
            ["P", "N"],
            ["N", "P"],
            ["P", "N"],
            ["P", "P"],
            ["N", "N"],
            ["P", "P"],
        ]
        result = bootstrap_organ_support(
            self.records,
            n_bootstrap=10,
            seed=TEST_BOOTSTRAP_SEED,
            forced_draws=forced,
        )["auroc"]
        self.assertEqual(result.n_requested_replicates, 10)
        self.assertEqual(result.n_valid_replicates, 7)
        self.assertEqual(result.n_invalid_replicates, 3)
        self.assertEqual(result.bootstrap_distribution, (1.0,) * 7)
        self.assertEqual((result.ci_lower, result.ci_upper), (1.0, 1.0))

    def test_zero_valid_replicates_produce_no_fabricated_bounds(self):
        forced = [["P", "P"], ["N", "N"]]
        result = bootstrap_organ_support(
            self.records,
            n_bootstrap=2,
            seed=TEST_BOOTSTRAP_SEED,
            forced_draws=forced,
        )["auroc"]
        self.assertEqual(result.n_valid_replicates, 0)
        self.assertEqual(result.n_invalid_replicates, 2)
        self.assertIsNone(result.ci_lower)
        self.assertIsNone(result.ci_upper)

    def test_threshold_is_supplied_and_never_reselected_or_calibrator_refit(self):
        forced = [["P", "N"], ["N", "P"]]
        results = bootstrap_organ_support(
            self.records,
            n_bootstrap=2,
            seed=TEST_BOOTSTRAP_SEED,
            fixed_threshold=0.5,
            threshold_identifier="TEST_THRESHOLD_ONLY",
            forced_draws=forced,
        )
        f1 = results["f1"]
        self.assertEqual(f1.bootstrap_distribution, (1.0, 1.0))
        self.assertFalse(f1.metadata["calibrator_refit"])
        self.assertFalse(f1.metadata["threshold_reselected"])


if __name__ == "__main__":
    unittest.main()
