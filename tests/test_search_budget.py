import copy
import unittest

from experiments.search_governance import (
    GovernanceError,
    canonical_sha256,
    materialize_candidates,
    validate_candidate_list,
)
from search_helpers import load_manifest, manifest_paths


class SearchBudgetTests(unittest.TestCase):
    def test_six_dry_manifests_each_materialize_exactly_30_unique_configs(self):
        paths = manifest_paths()
        self.assertEqual(len(paths), 6)
        for path in paths:
            manifest, _, candidates = load_manifest(path)
            self.assertEqual(len(candidates), 30)
            self.assertEqual(len({item["config_hash"] for item in candidates}), 30)
            self.assertEqual(canonical_sha256(candidates), manifest["candidate_list_hash"])
            self.assertEqual(manifest["scientific_candidate_slots_consumed"], 0)

    def test_29_and_31_fail_while_30_passes(self):
        _, space, candidates = load_manifest(manifest_paths()[0])
        validate_candidate_list(candidates)
        with self.assertRaises(GovernanceError):
            validate_candidate_list(candidates[:29])
        extra = copy.deepcopy(candidates[0])
        extra["candidate_id"] = "candidate_31"
        extra["config"] = {**extra["config"], "dropout": 0.123456789}
        extra["config_hash"] = canonical_sha256(extra["config"])
        with self.assertRaises(GovernanceError):
            validate_candidate_list(candidates + (extra,))
        with self.assertRaises(GovernanceError):
            materialize_candidates(
                space, task="recovery", seed=123, run_type="scientific"
            )


if __name__ == "__main__":
    unittest.main()
