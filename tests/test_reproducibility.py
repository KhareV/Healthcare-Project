import csv
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from data.collate import create_dataloader
from phase4_helpers import tensor_dataset
from training.reproducibility import resolve_device, set_deterministic_seed
from training.smoke import SmokeTestModel
from vedant_infra.registry import (
    REGISTRY_COLUMNS,
    make_registry_record,
    upsert_registry_record,
    validate_registry,
)


class ReproducibilityTests(unittest.TestCase):
    def test_seed_reproduces_python_numpy_torch_and_model_initialization(self):
        set_deterministic_seed(1234)
        first = (random.random(), np.random.rand(), torch.rand(1), SmokeTestModel(2))
        first_state = {key: value.clone() for key, value in first[3].state_dict().items()}
        set_deterministic_seed(1234)
        second = (random.random(), np.random.rand(), torch.rand(1), SmokeTestModel(2))
        self.assertEqual(first[0], second[0])
        self.assertEqual(first[1], second[1])
        self.assertTrue(torch.equal(first[2], second[2]))
        for key, value in second[3].state_dict().items():
            self.assertTrue(torch.equal(first_state[key], value))

    def test_same_loader_seed_reproduces_shuffled_order(self):
        dataset = tensor_dataset("train")
        def order(seed):
            loader = create_dataloader(
                dataset, batch_size=1, shuffle=True, seed=seed
            )
            return [batch.identifiers["prediction_time"][0] for batch in loader]
        self.assertEqual(order(88), order(88))

    def test_cpu_device_is_supported_without_gpu(self):
        self.assertEqual(resolve_device("cpu"), torch.device("cpu"))

    def test_registry_upsert_records_one_completed_smoke_row(self):
        digest = "a" * 64
        record = make_registry_record(
            run_id="synthetic-smoke-unit",
            timestamp_utc="2026-09-17T00:00:00Z",
            task="synthetic_test",
            model_family="smoke_test",
            seed="1234",
            code_commit="UNAVAILABLE_NOT_GIT_REPOSITORY",
            config_ref="configs/smoke/test.json",
            config_hash=digest,
            split_hash=digest,
            feature_version="synthetic_feature_schema_v1",
            label_version="synthetic_contract_labels_v1",
            model_artifact_ref="artifacts/smoke/test.pt",
            model_sha256=digest,
            metrics_ref="artifacts/smoke/test.json",
            status="completed",
            notes="SYNTHETIC SMOKE TEST ONLY",
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "registry.csv"
            upsert_registry_record(path, record)
            upsert_registry_record(path, record)
            self.assertEqual(validate_registry(path), 1)
            with path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]["model_family"], "smoke_test")


if __name__ == "__main__":
    unittest.main()
