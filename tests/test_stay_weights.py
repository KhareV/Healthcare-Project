import unittest

from evaluation.weights import StayWeightError, compute_stay_weights


class StayWeightTests(unittest.TestCase):
    def test_every_eligible_stay_has_total_weight_one(self):
        stay_ids = []
        for stay_id, count in (("one", 1), ("two", 2), ("seven", 7), ("twelve", 12)):
            stay_ids.extend([stay_id] * count)
        result = compute_stay_weights(stay_ids, [True] * len(stay_ids))
        for stay_id in set(stay_ids):
            total = sum(
                weight
                for row_stay, weight in zip(stay_ids, result.weights)
                if row_stay == stay_id
            )
            self.assertAlmostEqual(total, 1.0)

    def test_weights_are_horizon_specific(self):
        stay_ids = ["A"] * 4
        recovery24 = compute_stay_weights(stay_ids, [True, True, True, True])
        recovery48 = compute_stay_weights(stay_ids, [True, True, False, False])
        self.assertEqual(recovery24.weights, (0.25, 0.25, 0.25, 0.25))
        self.assertEqual(recovery48.weights, (0.5, 0.5, 0.0, 0.0))

    def test_missing_stay_id_length_mismatch_and_nonboolean_fail(self):
        with self.assertRaises(StayWeightError):
            compute_stay_weights([None], [True])
        with self.assertRaises(StayWeightError):
            compute_stay_weights(["A"], [])
        with self.assertRaises(StayWeightError):
            compute_stay_weights(["A"], [1])


if __name__ == "__main__":
    unittest.main()
