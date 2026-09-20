import unittest

from evaluation.weighted_stats import (
    WeightedStatisticError,
    weighted_median,
    weighted_percentile,
)


class WeightedMedianTests(unittest.TestCase):
    def test_weighted_median_differs_from_ordinary_median(self):
        values = [10.0, 20.0, 0.0]
        weights = [0.5, 0.5, 1.0]
        self.assertEqual(weighted_median(values, weights), 0.0)
        self.assertEqual(sorted(values)[1], 10.0)

    def test_unsorted_repeated_single_and_percentiles(self):
        self.assertEqual(weighted_median([5.0], [2.0]), 5.0)
        self.assertEqual(weighted_median([3.0, 1.0, 3.0], [1.0, 1.0, 1.0]), 3.0)
        self.assertEqual(weighted_percentile([30.0, 10.0, 20.0], [1.0, 1.0, 1.0], 0), 10.0)
        self.assertEqual(weighted_percentile([30.0, 10.0, 20.0], [1.0, 1.0, 1.0], 1), 30.0)

    def test_exact_half_uses_frozen_lower_boundary(self):
        self.assertEqual(weighted_median([10.0, 20.0], [1.0, 1.0]), 10.0)

    def test_invalid_weights_values_and_quantiles_fail(self):
        cases = (
            ([1.0], [-1.0], 0.5),
            ([1.0], [0.0], 0.5),
            ([float("nan")], [1.0], 0.5),
            ([1.0], [1.0], -0.1),
            ([1.0], [1.0], 1.1),
        )
        for values, weights, quantile in cases:
            with self.assertRaises(WeightedStatisticError):
                weighted_percentile(values, weights, quantile)


if __name__ == "__main__":
    unittest.main()
