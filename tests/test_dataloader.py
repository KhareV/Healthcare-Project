import math
import unittest

import torch

from data.collate import CanonicalBatch, create_dataloader
from data.dataset import DatasetContractError, subjects_in_dataset
from phase4_helpers import tensor_dataset


def identifier_order(loader):
    return [
        (subject, prediction)
        for batch in loader
        for subject, prediction in zip(
            batch.identifiers["subject_id"], batch.identifiers["prediction_time"]
        )
    ]


class DataLoaderTests(unittest.TestCase):
    def test_batch_contract_shapes_and_masks(self):
        loader = create_dataloader(
            tensor_dataset("train"), batch_size=2, shuffle=False, seed=17
        )
        batch = next(iter(loader))
        self.assertIsInstance(batch, CanonicalBatch)
        self.assertEqual(tuple(batch.sequence.shape), (2, 8, 2))
        self.assertEqual(tuple(batch.padding_mask.shape), (2, 8))
        self.assertEqual(tuple(batch.observation_mask.shape), (2, 8, 2))
        self.assertIsNone(batch.tslo)
        self.assertIsNone(batch.static_features)

    def test_task_masks_batch_independently(self):
        batch = next(
            iter(
                create_dataloader(
                    tensor_dataset("train"), batch_size=2, shuffle=False, seed=17
                )
            )
        )
        self.assertEqual(batch.eligibility["recovery24"].tolist(), [True, True])
        self.assertEqual(batch.eligibility["recovery48"].tolist(), [True, False])
        self.assertEqual(batch.eligibility["icu_time"].tolist(), [True, True])
        self.assertEqual(batch.eligibility["organ_support"].tolist(), [True, False])

    def test_censored_support_target_is_nan_not_negative(self):
        batch = next(
            iter(
                create_dataloader(
                    tensor_dataset("train"), batch_size=2, shuffle=False, seed=17
                )
            )
        )
        self.assertEqual(batch.targets["organ_support"][0].item(), 1.0)
        self.assertTrue(math.isnan(batch.targets["organ_support"][1].item()))
        self.assertFalse(batch.eligibility["organ_support"][1])

    def test_training_shuffle_is_reproducible_for_same_seed(self):
        dataset = tensor_dataset("validation")
        loader_one = create_dataloader(
            dataset, batch_size=1, shuffle=False, seed=123
        )
        loader_two = create_dataloader(
            dataset, batch_size=1, shuffle=False, seed=123
        )
        self.assertEqual(identifier_order(loader_one), identifier_order(loader_two))

        train = tensor_dataset("train")
        shuffled_one = create_dataloader(
            train, batch_size=1, shuffle=True, seed=123
        )
        shuffled_two = create_dataloader(
            train, batch_size=1, shuffle=True, seed=123
        )
        self.assertEqual(identifier_order(shuffled_one), identifier_order(shuffled_two))

    def test_validation_order_is_stable_and_shuffle_is_rejected(self):
        validation = tensor_dataset("validation")
        first = identifier_order(
            create_dataloader(validation, batch_size=2, shuffle=False, seed=1)
        )
        second = identifier_order(
            create_dataloader(validation, batch_size=2, shuffle=False, seed=999)
        )
        self.assertEqual(first, second)
        with self.assertRaises(DatasetContractError):
            create_dataloader(validation, batch_size=2, shuffle=True, seed=1)

    def test_train_validation_loaders_have_no_subject_overlap(self):
        train = tensor_dataset("train")
        validation = tensor_dataset("validation")
        self.assertFalse(subjects_in_dataset(train) & subjects_in_dataset(validation))

    def test_batch_device_transfer_preserves_identifiers_and_shapes(self):
        batch = next(
            iter(
                create_dataloader(
                    tensor_dataset("train"), batch_size=2, shuffle=False, seed=1
                )
            )
        )
        moved = batch.to(torch.device("cpu"))
        self.assertEqual(moved.identifiers, batch.identifiers)
        self.assertEqual(tuple(moved.sequence.shape), (2, 8, 2))


if __name__ == "__main__":
    unittest.main()
