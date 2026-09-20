import unittest

from bootstrap_helpers import (
    TEST_BOOTSTRAP_SEED,
    icu_median_ae,
    recovery_mae,
    regression_record,
)
from evaluation.bootstrap import bootstrap_icu_time, bootstrap_recovery, grouped_bootstrap
from evaluation.metrics import METRIC_IMPLEMENTATION_VERSION


class BootstrapMetricReuseTests(unittest.TestCase):
    def test_two_stay_forced_draws_match_hand_calculation(self):
        records = [
            regression_record("A", "t0", 10),
            regression_record("B", "t0", 0),
        ]
        forced = [["A", "A"], ["A", "B"], ["B", "A"], ["B", "B"]]
        result = grouped_bootstrap(
            records,
            metric_fn=recovery_mae,
            metric_name="mae",
            task="recovery",
            n_bootstrap=4,
            seed=TEST_BOOTSTRAP_SEED,
            horizon="24h",
            forced_draws=forced,
        )
        self.assertEqual(result.point_estimate, 5.0)
        self.assertEqual(result.bootstrap_distribution, (10.0, 5.0, 5.0, 0.0))
        self.assertAlmostEqual(result.ci_lower, 0.375)
        self.assertAlmostEqual(result.ci_upper, 9.625)

    def test_same_callback_is_used_for_point_and_every_replicate(self):
        records = [regression_record("A", "t0", 1), regression_record("B", "t0", 3)]
        calls = []

        def instrumented(rows):
            calls.append(tuple(row.stay_id for row in rows))
            return recovery_mae(rows)

        result = grouped_bootstrap(
            records,
            metric_fn=instrumented,
            metric_name="mae",
            task="recovery",
            n_bootstrap=3,
            seed=TEST_BOOTSTRAP_SEED,
        )
        self.assertEqual(len(calls), 4)
        self.assertEqual(result.metric_implementation_version, METRIC_IMPLEMENTATION_VERSION)

    def test_original_point_is_not_bootstrap_mean(self):
        records = [regression_record("A", "t0", 0), regression_record("B", "t0", 10)]
        forced = [["A", "A"], ["A", "B"], ["B", "A"], ["B", "B"]]
        result = grouped_bootstrap(
            records,
            metric_fn=icu_median_ae,
            metric_name="median_absolute_error",
            task="icu_stay_time",
            n_bootstrap=4,
            seed=TEST_BOOTSTRAP_SEED,
            forced_draws=forced,
        )
        bootstrap_mean = sum(result.bootstrap_distribution) / 4
        self.assertEqual(result.point_estimate, 0.0)
        self.assertEqual(bootstrap_mean, 2.5)

    def test_model_family_metadata_cannot_change_bootstrap_math(self):
        records = [regression_record("A", "t0", 2), regression_record("B", "t0", 8)]
        forced = [["A", "A"], ["A", "B"], ["B", "A"], ["B", "B"]]
        common = dict(
            records=records,
            metric_fn=recovery_mae,
            metric_name="mae",
            task="recovery",
            n_bootstrap=4,
            seed=TEST_BOOTSTRAP_SEED,
            forced_draws=forced,
        )
        gru = grouped_bootstrap(**common, metadata={"model_family": "gru"})
        xgb = grouped_bootstrap(**common, metadata={"model_family": "xgboost"})
        self.assertEqual(gru.point_estimate, xgb.point_estimate)
        self.assertEqual(gru.bootstrap_distribution, xgb.bootstrap_distribution)
        self.assertEqual((gru.ci_lower, gru.ci_upper), (xgb.ci_lower, xgb.ci_upper))

    def test_recovery_horizons_use_independent_eligible_cluster_sets(self):
        recovery24 = [
            regression_record("A", "t0", 1),
            regression_record("A", "t1", 1),
            regression_record("B", "t0", 3),
            regression_record("B", "t1", 3),
        ]
        recovery48 = [
            regression_record("A", "t0", 1),
            regression_record("A", "t1", 999, eligible=False),
            regression_record("B", "t0", 3),
            regression_record("B", "t1", 999, eligible=False),
        ]
        results = bootstrap_recovery(
            recovery24,
            recovery48,
            n_bootstrap=2,
            seed=TEST_BOOTSTRAP_SEED,
            forced_draws_24=[["A", "B"], ["B", "A"]],
            forced_draws_48=[["A", "B"], ["B", "A"]],
        )
        self.assertEqual(results["24h"]["mae"].n_original_examples, 4)
        self.assertEqual(results["48h"]["mae"].n_original_examples, 2)
        self.assertNotIn("directional_agreement", results["24h"])

    def test_icu_wrapper_bootstraps_hours_domain_phase9_estimator(self):
        records = [regression_record("A", "t0", 2), regression_record("B", "t0", 8)]
        result = bootstrap_icu_time(
            records,
            n_bootstrap=2,
            seed=TEST_BOOTSTRAP_SEED,
            forced_draws=[["A", "B"], ["B", "A"]],
        )["median_absolute_error"]
        self.assertEqual(result.point_estimate, 2.0)
        self.assertEqual(result.bootstrap_distribution, (2.0, 2.0))


if __name__ == "__main__":
    unittest.main()
