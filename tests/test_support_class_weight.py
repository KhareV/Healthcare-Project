import tempfile
import unittest
from pathlib import Path

import torch

from data.collate import CanonicalBatch, create_dataloader
from phase4_helpers import SYNTHETIC_SPLIT
from phase7_helpers import support_contract, support_dataset
from training.class_weights import (
    ScientificFeasibilityError,
    SupportClassWeight,
    fit_support_class_weight,
)
from vedant_infra.hashing import sha256_file


class _Partition:
    partition = "train"


class _Loader:
    dataset = _Partition()

    def __init__(self, labels, eligibility):
        size = len(labels)
        self.batch = CanonicalBatch(
            identifiers={
                "subject_id": tuple("S{}".format(i) for i in range(size)),
                "stay_id": tuple("ST{}".format(i) for i in range(size)),
                "prediction_time": tuple("T{}".format(i) for i in range(size)),
                "grid_index": tuple(0 for _ in range(size)),
                "split": tuple("train" for _ in range(size)),
            },
            sequence=torch.zeros((size, 8, 1)),
            padding_mask=torch.zeros((size, 8), dtype=torch.bool),
            observation_mask=torch.zeros((size, 8, 1), dtype=torch.bool),
            tslo=None,
            static_features=None,
            targets={"organ_support": torch.tensor(labels, dtype=torch.float32)},
            eligibility={"organ_support": torch.tensor(eligibility, dtype=torch.bool)},
            versions={},
            feature_names=("f",),
        )

    def __iter__(self):
        yield self.batch


def fit(loader):
    contract = support_contract()
    return fit_support_class_weight(
        loader,
        split_hash=sha256_file(SYNTHETIC_SPLIT),
        label_contract_version=contract.contract_version,
        event_dictionary_version=contract.event_dictionary_version,
        source_config_hash="a" * 64,
    )


class SupportClassWeightTests(unittest.TestCase):
    def test_eligible_train_only_formula_ignores_censored_placeholders(self):
        eligibility = [True] * 6 + [False] * 3
        first = fit(_Loader([1, 1, 0, 0, 0, 0, 999, -5, 1], eligibility))
        second = fit(_Loader([1, 1, 0, 0, 0, 0, -999, 7, 0], eligibility))
        self.assertEqual(first.positive_count, 2)
        self.assertEqual(first.negative_count, 4)
        self.assertEqual(first.eligible_train_count, 6)
        self.assertEqual(first.pos_weight, 2.0)
        self.assertEqual(first, second)

    def test_zero_positive_training_rows_fail_scientifically(self):
        with self.assertRaisesRegex(
            ScientificFeasibilityError, "NO POSITIVE TRAINING EXAMPLES"
        ):
            fit(_Loader([0, 0, 0], [True, True, False]))

    def test_validation_and_test_cannot_fit_weight(self):
        contract = support_contract()
        for partition in ("validation", "test"):
            loader = create_dataloader(
                support_dataset(partition), batch_size=64, shuffle=False, seed=1
            )
            with self.assertRaises(ValueError):
                fit_support_class_weight(
                    loader,
                    split_hash=sha256_file(SYNTHETIC_SPLIT),
                    label_contract_version=contract.contract_version,
                    event_dictionary_version=contract.event_dictionary_version,
                    source_config_hash="a" * 64,
                )

    def test_artifact_roundtrip_records_train_provenance(self):
        weight = fit(_Loader([1, 0], [True, True]))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weight.json"
            _, digest = weight.save(path)
            self.assertEqual(digest, sha256_file(path))
            self.assertEqual(SupportClassWeight.load(path), weight)
            self.assertEqual(weight.provenance, "ELIGIBLE_TRAIN_ROWS_ONLY")


if __name__ == "__main__":
    unittest.main()
