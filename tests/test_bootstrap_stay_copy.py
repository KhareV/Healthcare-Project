import unittest

from bootstrap_helpers import regression_record
from evaluation.bootstrap import materialize_stay_copy_replicate
from evaluation.weights import compute_stay_weights


class BootstrapStayCopyTests(unittest.TestCase):
    def test_duplicate_sampled_stay_becomes_two_cluster_copies(self):
        records = [
            regression_record("A", "A1", 1),
            regression_record("A", "A2", 3),
            regression_record("B", "B1", 9),
        ]
        replicate = materialize_stay_copy_replicate(
            records, ["A", "A"], replicate_index=4
        )
        self.assertEqual(len(replicate.copied_rows), 4)
        self.assertEqual(len(replicate.bootstrap_cluster_ids), 2)
        self.assertNotEqual(replicate.bootstrap_cluster_ids[0], replicate.bootstrap_cluster_ids[1])
        copied = replicate.prediction_records
        weights = compute_stay_weights(
            [row.stay_id for row in copied], [row.eligible for row in copied]
        ).weights
        for cluster_id in replicate.bootstrap_cluster_ids:
            total = sum(
                weight for row, weight in zip(copied, weights) if row.stay_id == cluster_id
            )
            self.assertEqual(total, 1.0)

    def test_long_stay_has_same_total_weight_as_one_row_stay(self):
        records = [regression_record("A", "t0", 10)] + [
            regression_record("B", "t{}".format(index), 0) for index in range(10)
        ]
        replicate = materialize_stay_copy_replicate(
            records, ["A", "B"], replicate_index=0
        )
        copied = replicate.prediction_records
        weights = compute_stay_weights(
            [row.stay_id for row in copied], [True] * len(copied)
        ).weights
        totals = {
            cluster_id: sum(
                weight for row, weight in zip(copied, weights) if row.stay_id == cluster_id
            )
            for cluster_id in replicate.bootstrap_cluster_ids
        }
        for total in totals.values():
            self.assertAlmostEqual(total, 1.0)

    def test_only_task_eligible_rows_are_copied(self):
        recovery24 = [
            regression_record("A", "t{}".format(index), index, eligible=True)
            for index in range(4)
        ]
        recovery48 = [
            regression_record("A", "t0", 0, eligible=True),
            regression_record("A", "t1", 1, eligible=True),
            regression_record("A", "t2", 999999, eligible=False),
            regression_record("A", "t3", 999999, eligible=False),
        ]
        copy24 = materialize_stay_copy_replicate(recovery24, ["A"], replicate_index=0)
        copy48 = materialize_stay_copy_replicate(recovery48, ["A"], replicate_index=0)
        self.assertEqual(len(copy24.copied_rows), 4)
        self.assertEqual(len(copy48.copied_rows), 2)


if __name__ == "__main__":
    unittest.main()
