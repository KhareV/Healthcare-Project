import unittest

from bootstrap_helpers import TEST_BOOTSTRAP_SEED, recovery_mae, regression_record
from evaluation.bootstrap import BootstrapError, grouped_bootstrap


class BootstrapReproducibilityTests(unittest.TestCase):
    def setUp(self):
        self.records = [
            regression_record("A", "t0", 10),
            regression_record("B", "t0", 0),
            regression_record("C", "t0", 4),
        ]

    def run_bootstrap(self, seed):
        return grouped_bootstrap(
            self.records,
            metric_fn=recovery_mae,
            metric_name="mae",
            task="recovery",
            n_bootstrap=25,
            seed=seed,
        )

    def test_same_seed_reproduces_draws_distribution_and_ci(self):
        self.assertEqual(
            self.run_bootstrap(TEST_BOOTSTRAP_SEED),
            self.run_bootstrap(TEST_BOOTSTRAP_SEED),
        )

    def test_different_seed_changes_sampling_sequence(self):
        first = self.run_bootstrap(TEST_BOOTSTRAP_SEED)
        second = self.run_bootstrap(TEST_BOOTSTRAP_SEED + 1)
        self.assertNotEqual(first.sampled_cluster_sequences, second.sampled_cluster_sequences)

    def test_bootstrap_count_and_seed_must_be_explicit_valid_integers(self):
        for count, seed in ((0, TEST_BOOTSTRAP_SEED), (True, TEST_BOOTSTRAP_SEED), (10, True)):
            with self.assertRaises(BootstrapError):
                grouped_bootstrap(
                    self.records,
                    metric_fn=recovery_mae,
                    metric_name="mae",
                    task="recovery",
                    n_bootstrap=count,
                    seed=seed,
                )

    def test_ineligible_extreme_attack_does_not_change_draws_distribution_or_ci(self):
        clean = self.records + [regression_record("A", "future", 999999, eligible=False)]
        attacked = self.records + [regression_record("A", "future", -999999, eligible=False)]
        common = dict(
            metric_fn=recovery_mae,
            metric_name="mae",
            task="recovery",
            n_bootstrap=25,
            seed=TEST_BOOTSTRAP_SEED,
        )
        first = grouped_bootstrap(clean, **common)
        second = grouped_bootstrap(attacked, **common)
        self.assertEqual(first.sampled_cluster_sequences, second.sampled_cluster_sequences)
        self.assertEqual(first.bootstrap_distribution, second.bootstrap_distribution)
        self.assertEqual((first.ci_lower, first.ci_upper), (second.ci_lower, second.ci_upper))


if __name__ == "__main__":
    unittest.main()
