import json
import tempfile
import unittest
from pathlib import Path

import torch

from data.collate import create_dataloader
from phase4_helpers import feature_schema, tensor_dataset
from training.early_stopping import EarlyStopping
from training.engine import (
    TaskAdapter,
    create_optimizer,
    masked_mean_loss,
    train_with_early_stopping,
)
from training.logging import JsonlRunLogger
from training.smoke import SmokeTestModel


class MaskedLossTests(unittest.TestCase):
    def test_ineligible_large_error_has_no_effect(self):
        predictions = torch.tensor([2.0, 100.0, 5.0])
        targets = torch.tensor([1.0, 10.0, 3.0])
        eligibility = torch.tensor([True, False, True])
        result = masked_mean_loss(
            predictions,
            targets,
            eligibility,
            lambda prediction, target: (prediction - target) ** 2,
        )
        self.assertEqual(result.eligible_count, 2)
        self.assertAlmostEqual(result.loss.item(), 2.5)

    def test_zero_eligible_has_no_nan_and_no_fake_signal(self):
        result = masked_mean_loss(
            torch.tensor([1.0, 2.0]),
            torch.tensor([float("nan"), float("nan")]),
            torch.tensor([False, False]),
            lambda prediction, target: (prediction - target) ** 2,
        )
        self.assertEqual(result.eligible_count, 0)
        self.assertIsNone(result.loss)


class EarlyStoppingTests(unittest.TestCase):
    def test_controlled_validation_sequence_tracks_best_and_stops(self):
        stopper = EarlyStopping(mode="min", patience=2)
        decisions = [
            stopper.update(value, epoch)
            for epoch, value in enumerate((5.0, 4.0, 4.5, 4.6), start=1)
        ]
        self.assertEqual(stopper.best_epoch, 2)
        self.assertEqual(stopper.best_value, 4.0)
        self.assertFalse(decisions[2].should_stop)
        self.assertTrue(decisions[3].should_stop)

    def test_monitor_name_is_recorded_and_nonfinite_value_is_rejected(self):
        stopper = EarlyStopping(
            mode="max", patience=1, monitor_name="validation_synthetic_score"
        )
        self.assertEqual(stopper.monitor_name, "validation_synthetic_score")
        with self.assertRaises(ValueError):
            stopper.update(float("nan"), 1)


class TrainingIntegrationTests(unittest.TestCase):
    def test_generic_engine_saves_and_restores_best_synthetic_checkpoint(self):
        torch.manual_seed(91)
        train = tensor_dataset("train")
        validation = tensor_dataset("validation")
        train_loader = create_dataloader(
            train, batch_size=2, shuffle=True, seed=91
        )
        validation_loader = create_dataloader(
            validation, batch_size=2, shuffle=False, seed=91
        )
        model = SmokeTestModel(feature_schema().feature_dim)
        optimizer = create_optimizer(
            model.parameters(), name="adamw", learning_rate=0.01, weight_decay=0.0
        )
        adapter = TaskAdapter(
            task_name="synthetic_test",
            target_key="recovery24",
            eligibility_key="recovery24",
            per_item_loss=lambda prediction, target: (prediction - target) ** 2,
        )
        metadata = {
            "run_id": "unit-smoke",
            "task": "synthetic_test",
            "model_family": "smoke_test",
            "seed": 91,
            "epoch": 0,
            "config_hash": "a" * 64,
            "tensor_contract_version": "tensor_contract_v1",
            "feature_schema_version": "synthetic_feature_schema_v1",
            "split_hash": "b" * 64,
            "code_commit": "UNAVAILABLE_NOT_GIT_REPOSITORY",
            "validation_value": None,
            "synthetic_smoke_test": True,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = train_with_early_stopping(
                model=model,
                train_loader=train_loader,
                validation_loader=validation_loader,
                adapter=adapter,
                optimizer=optimizer,
                device=torch.device("cpu"),
                max_epochs=3,
                early_stopping=EarlyStopping(mode="min", patience=2),
                checkpoint_path=root / "smoke.pt",
                checkpoint_metadata=metadata,
                logger=JsonlRunLogger(root / "run.jsonl"),
            )
            self.assertTrue((root / "smoke.pt").exists())
            self.assertGreaterEqual(result.best_epoch, 1)
            events = [
                json.loads(line)
                for line in (root / "run.jsonl").read_text().splitlines()
            ]
            self.assertIn("best_checkpoint_restored", {item["event"] for item in events})
            epoch_events = [item for item in events if item["event"] == "epoch"]
            self.assertTrue(
                all(item["validation_monitor_name"] == "validation_loss" for item in epoch_events)
            )
            self.assertTrue(all("subject_id" not in item for item in events))


if __name__ == "__main__":
    unittest.main()
