import tempfile
import unittest
from pathlib import Path

import torch

from training.checkpoint import (
    CheckpointError,
    load_checkpoint,
    metadata_sidecar_path,
    save_checkpoint,
)
from training.smoke import SmokeTestModel
from vedant_infra.hashing import sha256_file


def metadata():
    return {
        "run_id": "checkpoint-unit-smoke",
        "task": "synthetic_test",
        "model_family": "smoke_test",
        "seed": 7,
        "epoch": 2,
        "config_hash": "a" * 64,
        "tensor_contract_version": "tensor_contract_v1",
        "feature_schema_version": "synthetic_feature_schema_v1",
        "split_hash": "b" * 64,
        "code_commit": "UNAVAILABLE_NOT_GIT_REPOSITORY",
        "validation_value": 1.25,
        "synthetic_smoke_test": True,
    }


class CheckpointTests(unittest.TestCase):
    def test_model_optimizer_and_metadata_roundtrip(self):
        torch.manual_seed(7)
        model = SmokeTestModel(2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        expected = {key: value.clone() for key, value in model.state_dict().items()}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "smoke.pt"
            _, digest = save_checkpoint(path, model, optimizer, metadata())
            self.assertEqual(digest, sha256_file(path))
            for parameter in model.parameters():
                parameter.data.add_(10)
            loaded = load_checkpoint(
                path,
                model,
                optimizer,
                expected_tensor_contract_version="tensor_contract_v1",
                expected_feature_schema_version="synthetic_feature_schema_v1",
            )
            self.assertEqual(loaded["checkpoint_sha256"], digest)
            for key, value in model.state_dict().items():
                self.assertTrue(torch.equal(value, expected[key]))

    def test_corrupted_checkpoint_fails_hash_verification(self):
        model = SmokeTestModel(2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "smoke.pt"
            save_checkpoint(path, model, optimizer, metadata())
            path.write_bytes(path.read_bytes() + b"corrupt")
            with self.assertRaises(CheckpointError):
                load_checkpoint(
                    path,
                    model,
                    expected_tensor_contract_version="tensor_contract_v1",
                    expected_feature_schema_version="synthetic_feature_schema_v1",
                )

    def test_wrong_tensor_or_feature_version_fails(self):
        model = SmokeTestModel(2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "smoke.pt"
            save_checkpoint(path, model, optimizer, metadata())
            with self.assertRaises(CheckpointError):
                load_checkpoint(
                    path,
                    model,
                    expected_tensor_contract_version="wrong",
                    expected_feature_schema_version="synthetic_feature_schema_v1",
                )
            with self.assertRaises(CheckpointError):
                load_checkpoint(
                    path,
                    model,
                    expected_tensor_contract_version="tensor_contract_v1",
                    expected_feature_schema_version="wrong",
                )

    def test_missing_required_metadata_is_rejected(self):
        model = SmokeTestModel(2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        incomplete = metadata()
        del incomplete["split_hash"]
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(CheckpointError):
                save_checkpoint(Path(directory) / "smoke.pt", model, optimizer, incomplete)

    def test_nonfinite_validation_metadata_is_rejected(self):
        model = SmokeTestModel(2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        invalid = metadata()
        invalid["validation_value"] = float("nan")
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(CheckpointError):
                save_checkpoint(Path(directory) / "smoke.pt", model, optimizer, invalid)


if __name__ == "__main__":
    unittest.main()
