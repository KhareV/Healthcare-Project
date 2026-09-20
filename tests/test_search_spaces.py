import copy
import unittest

from experiments.search_governance import (
    GovernanceError,
    canonical_sha256,
    load_search_space,
    validate_search_space,
)
from search_helpers import ROOT


class SearchSpaceTests(unittest.TestCase):
    def test_gru_and_xgb_spaces_match_authoritative_ranges(self):
        gru, _ = load_search_space(ROOT / "configs/search_space_gru_v1.json")
        xgb, _ = load_search_space(ROOT / "configs/search_space_xgb_v1.json")
        self.assertEqual(gru["budget_per_task"], 30)
        self.assertEqual(gru["parameters"]["hidden_dim"]["values"], [64, 128, 192])
        self.assertEqual(gru["training"]["maximum_epochs"], 60)
        self.assertEqual(gru["training"]["early_stopping_patience"], 8)
        self.assertEqual(xgb["parameters"]["max_depth"]["minimum"], 3)
        self.assertEqual(xgb["parameters"]["max_depth"]["maximum"], 8)
        self.assertEqual(xgb["training"]["maximum_estimators"], 1000)

    def test_canonical_hash_ignores_mapping_order_but_detects_change(self):
        first = {"b": 2, "a": {"y": 1, "x": 0}}
        second = {"a": {"x": 0, "y": 1}, "b": 2}
        self.assertEqual(canonical_sha256(first), canonical_sha256(second))
        changed = copy.deepcopy(first)
        changed["a"]["x"] = 9
        self.assertNotEqual(canonical_sha256(first), canonical_sha256(changed))

    def test_range_mutation_is_scientific_configuration_conflict(self):
        gru, _ = load_search_space(ROOT / "configs/search_space_gru_v1.json")
        changed = copy.deepcopy(gru)
        changed["parameters"]["dropout"]["maximum"] = 0.5
        with self.assertRaises(GovernanceError):
            validate_search_space(changed)


if __name__ == "__main__":
    unittest.main()
