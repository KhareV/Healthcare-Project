import copy
import unittest

from experiments.search_governance import GovernanceError, assert_immutable_transition
from search_helpers import load_manifest, manifest_paths


class SearchFreezeTests(unittest.TestCase):
    def test_started_search_rejects_result_adaptive_mutation(self):
        manifest, _, _ = load_manifest(manifest_paths()[0])
        before = {
            **manifest,
            "status": "RUNNING",
            "feature_version": manifest["feature_version"],
        }
        after = copy.deepcopy(before)
        after["search_space_hash"] = "f" * 64
        with self.assertRaises(GovernanceError):
            assert_immutable_transition(before, after)

    def test_started_search_rejects_objective_or_candidate_list_change(self):
        manifest, _, _ = load_manifest(manifest_paths()[0])
        before = {**manifest, "status": "COMPLETE"}
        for field in ("validation_objective", "candidate_list_hash"):
            after = copy.deepcopy(before)
            after[field] = "changed"
            with self.assertRaises(GovernanceError):
                assert_immutable_transition(before, after)


if __name__ == "__main__":
    unittest.main()
