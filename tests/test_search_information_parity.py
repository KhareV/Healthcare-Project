import unittest

from experiments.search_governance import (
    GovernanceError,
    InformationManifest,
    validate_information_parity,
)


def manifest(family, representation, **changes):
    values = dict(
        family=family,
        row_keys_hash="a" * 64,
        feature_schema_version="synthetic_feature_schema_v1",
        temporal_bins=8,
        dynamic_features=("feature_0", "feature_1"),
        observation_masks=("feature_0_mask", "feature_1_mask"),
        tslo_features=(),
        static_features=(),
        preprocessing_hash="b" * 64,
        split_hash="c" * 64,
        eligibility_hash="d" * 64,
        representation=representation,
    )
    values.update(changes)
    return InformationManifest(**values)


class SearchInformationParityTests(unittest.TestCase):
    def test_only_sequence_versus_flattened_representation_may_differ(self):
        gru = manifest("gru", "sequence_8xf")
        xgb = manifest("xgboost", "flattened_8xf")
        validate_information_parity(gru, xgb)

    def test_extra_feature_or_different_eligibility_fails(self):
        gru = manifest("gru", "sequence_8xf")
        for changed in (
            {"dynamic_features": ("feature_0", "feature_1", "future_feature")},
            {"eligibility_hash": "e" * 64},
            {"temporal_bins": 9},
        ):
            with self.assertRaises(GovernanceError):
                validate_information_parity(gru, manifest("xgboost", "flattened_8xf", **changed))


if __name__ == "__main__":
    unittest.main()
