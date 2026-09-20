import unittest
from dataclasses import replace

import torch

from data.dataset import (
    CanonicalTensorDataset,
    DatasetContractError,
    SyntheticTensorizationPolicy,
    subjects_in_dataset,
)
from data.schema import TensorContractError
from phase4_helpers import (
    CANONICAL_FIXTURE,
    SYNTHETIC_SPLIT,
    canonical_dataset,
    feature_schema,
    tensor_dataset,
)


class CanonicalDatasetTests(unittest.TestCase):
    def test_partition_lengths_and_identifiers(self):
        train = tensor_dataset("train")
        validation = tensor_dataset("validation")
        self.assertEqual(len(train), 2)
        self.assertEqual(len(validation), 3)
        first = train[0]
        self.assertEqual(first["identifiers"]["subject_id"], "SYNTH-A")
        self.assertEqual(first["identifiers"]["split"], "train")

    def test_sequence_feature_order_and_masks_are_preserved(self):
        dataset = tensor_dataset("train")
        item = dataset[0]
        source = dataset.examples[0]
        self.assertEqual(tuple(item["sequence"].shape), (8, 2))
        self.assertEqual(item["sequence"].dtype, torch.float32)
        self.assertEqual(item["feature_names"], ("feature_0", "feature_1"))
        self.assertTrue(
            torch.equal(item["padding_mask"], torch.tensor(source.padding_mask))
        )
        self.assertTrue(
            torch.equal(
                item["observation_mask"], torch.tensor(source.observation_mask)
            )
        )

    def test_synthetic_fill_does_not_change_observation_or_padding_masks(self):
        dataset = tensor_dataset("train")
        item = dataset[0]
        source = dataset.examples[0]
        for bin_index, row in enumerate(source.history_values):
            for feature_index, value in enumerate(row):
                if value is None:
                    self.assertEqual(item["sequence"][bin_index, feature_index].item(), 0.0)
                    self.assertFalse(item["observation_mask"][bin_index, feature_index])

    def test_tslo_and_static_features_remain_blocked_and_unmodified(self):
        item = tensor_dataset("train")[0]
        self.assertIsNone(item["tslo"])
        self.assertIsNone(item["static_features"])

    def test_targets_and_independent_eligibility_are_preserved(self):
        dataset = tensor_dataset("train")
        item = next(
            dataset[index]
            for index, example in enumerate(dataset.examples)
            if not example.eligibility.recovery48_eligible
        )
        self.assertTrue(item["eligibility"]["recovery24"])
        self.assertFalse(item["eligibility"]["recovery48"])
        self.assertIsNone(item["targets"]["recovery48"])
        self.assertIsNone(item["targets"]["organ_support"])

    def test_multiple_stay_snapshot_counts_survive_partitioning(self):
        train = tensor_dataset("train")
        validation = tensor_dataset("validation")
        self.assertEqual(len({item.stay_id for item in train.examples}), 1)
        self.assertEqual(len(train), 2)
        self.assertEqual(len(validation), 3)

    def test_invalid_contract_or_feature_version_is_rejected(self):
        dataset = canonical_dataset()
        schema = feature_schema()
        split_map = {"SYNTH-A": "train", "SYNTH-B": "validation", "SYNTH-C": "test"}
        with self.assertRaises(TensorContractError):
            CanonicalTensorDataset(
                replace(dataset, tensor_contract_version="wrong"),
                schema,
                "train",
                SyntheticTensorizationPolicy(0.0),
                split_map,
            )
        with self.assertRaises(TensorContractError):
            CanonicalTensorDataset(
                dataset,
                replace(schema, version="wrong"),
                "train",
                SyntheticTensorizationPolicy(0.0),
                split_map,
            )

    def test_duplicate_canonical_key_is_rejected_upstream(self):
        dataset = canonical_dataset()
        schema = feature_schema()
        split_map = {"SYNTH-A": "train", "SYNTH-B": "validation", "SYNTH-C": "test"}
        duplicate = replace(dataset, examples=dataset.examples + (dataset.examples[0],))
        with self.assertRaises(TensorContractError):
            CanonicalTensorDataset(
                duplicate,
                schema,
                "train",
                SyntheticTensorizationPolicy(0.0),
                split_map,
            )

    def test_empty_partition_is_explicit(self):
        dataset = canonical_dataset()
        train_only = replace(
            dataset,
            examples=tuple(item for item in dataset.examples if item.split == "train"),
        )
        split_map = {"SYNTH-A": "train"}
        with self.assertRaises(DatasetContractError):
            CanonicalTensorDataset(
                train_only,
                feature_schema(),
                "validation",
                SyntheticTensorizationPolicy(0.0),
                split_map,
            )

    def test_split_artifact_conflict_is_rejected(self):
        dataset = canonical_dataset()
        wrong = {"SYNTH-A": "validation", "SYNTH-B": "validation", "SYNTH-C": "test"}
        with self.assertRaises(DatasetContractError):
            CanonicalTensorDataset(
                dataset,
                feature_schema(),
                "train",
                SyntheticTensorizationPolicy(0.0),
                wrong,
            )

    def test_train_validation_subjects_are_disjoint(self):
        self.assertFalse(
            subjects_in_dataset(tensor_dataset("train"))
            & subjects_in_dataset(tensor_dataset("validation"))
        )


if __name__ == "__main__":
    unittest.main()
