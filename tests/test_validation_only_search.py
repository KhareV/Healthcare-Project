import unittest

from experiments.search_governance import (
    CandidateResult,
    GovernanceError,
    ValidationOnlyAccess,
    choose_best_within_family,
    execute_synthetic_candidate,
)


class ValidationOnlySearchTests(unittest.TestCase):
    def test_test_partition_is_hard_rejected(self):
        self.assertEqual(ValidationOnlyAccess.require("train"), "train")
        self.assertEqual(ValidationOnlyAccess.require("validation"), "validation")
        with self.assertRaises(GovernanceError):
            ValidationOnlyAccess.require("test")

    def test_test_sentinel_never_enters_allowed_partition_payloads(self):
        partitions = {
            "train": ("train-a",),
            "validation": ("val-a",),
            "test": ("DO_NOT_TOUCH_TEST_SENTINEL",),
        }
        observed = []
        for name in ("train", "validation"):
            observed.extend(partitions[ValidationOnlyAccess.require(name)])
        self.assertNotIn("DO_NOT_TOUCH_TEST_SENTINEL", observed)

    def test_dry_runner_never_requests_test_provider(self):
        requested = []
        payloads = {
            "train": ("train-a",),
            "validation": ("val-a",),
            "test": ("DO_NOT_TOUCH_TEST_SENTINEL",),
        }

        def provider(partition):
            requested.append(partition)
            return payloads[partition]

        def runner(config, train_data, validation_data):
            self.assertEqual(train_data, ("train-a",))
            self.assertEqual(validation_data, ("val-a",))
            return {"validation_recovery24_mae": 1.0, "test_accessed": False}

        result = execute_synthetic_candidate(
            {"candidate_id": "dryrun_gru_recovery_01", "config": {}},
            partition_provider=provider,
            runner=runner,
        )
        self.assertEqual(requested, ["train", "validation"])
        self.assertFalse(result["test_accessed"])

    def test_frozen_task_objectives_choose_only_within_family(self):
        recovery = [
            CandidateResult("A", "rA", "COMPLETE", {"validation_recovery24_mae": 2, "mae48": 1}),
            CandidateResult("B", "rB", "COMPLETE", {"validation_recovery24_mae": 1, "mae48": 100}),
        ]
        icu = [
            CandidateResult("A", "iA", "COMPLETE", {"validation_median_absolute_error_hours": 10, "mae": 5}),
            CandidateResult("B", "iB", "COMPLETE", {"validation_median_absolute_error_hours": 8, "mae": 100}),
        ]
        support = [
            CandidateResult("A", "sA", "COMPLETE", {"validation_auprc": 0.3}, "raw_uncalibrated"),
            CandidateResult("B", "sB", "COMPLETE", {"validation_auprc": 0.4}, "raw_uncalibrated"),
        ]
        self.assertEqual(choose_best_within_family(recovery, "recovery").candidate_id, "B")
        self.assertEqual(choose_best_within_family(icu, "icu_stay_time").candidate_id, "B")
        self.assertEqual(choose_best_within_family(support, "organ_support").candidate_id, "B")

    def test_tie_and_calibrated_support_ranking_are_rejected(self):
        tied = [
            CandidateResult("A", "a", "COMPLETE", {"validation_recovery24_mae": 1.0}),
            CandidateResult("B", "b", "COMPLETE", {"validation_recovery24_mae": 1.0}),
        ]
        with self.assertRaises(GovernanceError):
            choose_best_within_family(tied, "recovery")
        calibrated = [
            CandidateResult("A", "a", "COMPLETE", {"validation_auprc": 0.9}, "calibrated")
        ]
        with self.assertRaises(GovernanceError):
            choose_best_within_family(calibrated, "organ_support")


if __name__ == "__main__":
    unittest.main()
