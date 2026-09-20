import unittest
import json
from pathlib import Path

from bootstrap_helpers import TEST_BOOTSTRAP_SEED, recovery_mae, regression_record
from evaluation.bootstrap import grouped_bootstrap, materialize_stay_copy_replicate, percentile


class GroupedBootstrapTests(unittest.TestCase):
    def test_all_rows_from_each_sampled_stay_move_together(self):
        records = [
            regression_record("A", "A1", 1),
            regression_record("A", "A2", 2),
            regression_record("A", "A3", 3),
            regression_record("B", "B1", 4),
            regression_record("B", "B2", 5),
        ]
        replicate = materialize_stay_copy_replicate(
            records, ["B", "A"], replicate_index=0
        )
        self.assertEqual(
            [
                (row.original_stay_id, row.source_record.prediction_time)
                for row in replicate.copied_rows
            ],
            [("B", "B1"), ("B", "B2"), ("A", "A1"), ("A", "A2"), ("A", "A3")],
        )

    def test_draw_count_equals_original_eligible_stay_count(self):
        records = [regression_record(stay, "t0", index) for index, stay in enumerate("ABCDE")]
        result = grouped_bootstrap(
            records,
            metric_fn=recovery_mae,
            metric_name="mae",
            task="recovery",
            n_bootstrap=10,
            seed=TEST_BOOTSTRAP_SEED,
        )
        self.assertEqual(result.n_original_stays, 5)
        self.assertTrue(all(len(sequence) == 5 for sequence in result.sampled_cluster_sequences))

    def test_one_stay_produces_degenerate_but_valid_interval(self):
        records = [
            regression_record("A", "t0", 1),
            regression_record("A", "t1", 3),
        ]
        result = grouped_bootstrap(
            records,
            metric_fn=recovery_mae,
            metric_name="mae",
            task="recovery",
            n_bootstrap=10,
            seed=TEST_BOOTSTRAP_SEED,
        )
        self.assertEqual(result.point_estimate, 2.0)
        self.assertEqual(result.ci_lower, 2.0)
        self.assertEqual(result.ci_upper, 2.0)
        self.assertEqual(result.n_valid_replicates, 10)

    def test_percentile_ci_uses_declared_linear_convention(self):
        self.assertAlmostEqual(percentile([0.0, 5.0, 5.0, 10.0], 0.025), 0.375)
        self.assertAlmostEqual(percentile([0.0, 5.0, 5.0, 10.0], 0.975), 9.625)

    def test_versioned_config_does_not_invent_scientific_count_or_seed(self):
        path = Path(__file__).resolve().parents[1] / "configs/evaluation/bootstrap_v1.json"
        config = json.loads(path.read_text(encoding="utf-8"))
        self.assertIsNone(config["n_bootstrap"])
        self.assertIsNone(config["seed"])
        self.assertFalse(config["final_test_access_allowed"])


if __name__ == "__main__":
    unittest.main()
