import tempfile
import unittest
from pathlib import Path

import torch

from data.collate import create_dataloader
from phase4_helpers import SYNTHETIC_SPLIT, tensor_dataset
from preprocess.target_scaler import (
    RecoveryTargetScaler,
    TargetScalerError,
    fit_recovery_target_scaler,
)
from vedant_infra.hashing import sha256_file


def loader(partition):
    return create_dataloader(
        tensor_dataset(partition), batch_size=64, shuffle=False, seed=11
    )


class TargetScalingTests(unittest.TestCase):
    def test_separate_train_eligible_statistics(self):
        scaler = fit_recovery_target_scaler(
            loader("train"), split_hash=sha256_file(SYNTHETIC_SPLIT)
        )
        self.assertEqual(scaler.mean_24, -1.5)
        self.assertEqual(scaler.scale_24, 0.5)
        self.assertEqual(scaler.eligible_count_24, 2)
        self.assertEqual(scaler.mean_48, -3.0)
        self.assertEqual(scaler.scale_48, 1.0)
        self.assertEqual(scaler.eligible_count_48, 1)
        self.assertEqual(scaler.source_partition, "train")

    def test_validation_and_test_loaders_are_rejected_for_fit(self):
        digest = sha256_file(SYNTHETIC_SPLIT)
        for partition in ("validation", "test"):
            with self.assertRaises(TargetScalerError):
                fit_recovery_target_scaler(loader(partition), split_hash=digest)

    def test_inverse_transform_and_artifact_roundtrip(self):
        scaler = fit_recovery_target_scaler(
            loader("train"), split_hash=sha256_file(SYNTHETIC_SPLIT)
        )
        raw = torch.tensor([[-2.0, -3.0], [-1.0, 1.0]])
        standardized = torch.stack(
            (
                scaler.transform_horizon(raw[:, 0], 0),
                scaler.transform_horizon(raw[:, 1], 1),
            ),
            dim=1,
        )
        self.assertTrue(torch.allclose(scaler.inverse_transform(standardized), raw))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scaler.json"
            _, digest = scaler.save(path)
            restored = RecoveryTargetScaler.load(path)
            self.assertEqual(digest, sha256_file(path))
            self.assertEqual(restored, scaler)
            self.assertTrue(
                torch.allclose(restored.inverse_transform(standardized), raw)
            )


if __name__ == "__main__":
    unittest.main()
