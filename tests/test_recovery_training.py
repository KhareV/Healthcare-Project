import json
import tempfile
import unittest
from pathlib import Path

import torch

from data.collate import create_dataloader
from models.gru_recovery import (
    RECOVERY_HORIZON_ORDER,
    RecoveryArtifactError,
    RecoveryGRU,
    load_recovery_bundle,
)
from phase4_helpers import SYNTHETIC_SPLIT, tensor_dataset
from preprocess.target_scaler import fit_recovery_target_scaler
from test_gru_base import legal_config
from training.checkpoint import load_checkpoint
from training.early_stopping import EarlyStopping
from training.engine import create_optimizer, train_with_early_stopping
from training.logging import JsonlRunLogger
from training.reproducibility import set_deterministic_seed
from training.tasks.recovery import RecoveryTaskAdapter
from vedant_infra.hashing import sha256_file


class RecoveryTrainingTests(unittest.TestCase):
    def test_synthetic_training_checkpoint_reload_and_inverse_prediction(self):
        seed = 44
        set_deterministic_seed(seed)
        train = tensor_dataset("train")
        validation = tensor_dataset("validation")
        scaler = fit_recovery_target_scaler(
            create_dataloader(train, batch_size=64, shuffle=False, seed=seed),
            split_hash=sha256_file(SYNTHETIC_SPLIT),
        )
        train_loader = create_dataloader(
            train, batch_size=64, shuffle=True, seed=seed
        )
        validation_loader = create_dataloader(
            validation, batch_size=64, shuffle=False, seed=seed
        )
        model = RecoveryGRU(legal_config())
        optimizer = create_optimizer(
            model.parameters(), name="adamw", learning_rate=0.001, weight_decay=1e-5
        )
        metadata = {
            "run_id": "phase5-unit-smoke",
            "task": "recovery",
            "model_family": "gru",
            "model_config": model.model_config,
            "seed": seed,
            "epoch": 0,
            "config_hash": "a" * 64,
            "tensor_contract_version": "tensor_contract_v1",
            "feature_schema_version": "synthetic_feature_schema_v1",
            "split_hash": sha256_file(SYNTHETIC_SPLIT),
            "target_scaler_sha256": "b" * 64,
            "horizon_order": list(RECOVERY_HORIZON_ORDER),
            "code_commit": "UNAVAILABLE_NOT_GIT_REPOSITORY",
            "validation_value": 0.0,
            "synthetic_smoke_test": True,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "recovery.pt"
            result = train_with_early_stopping(
                model=model,
                train_loader=train_loader,
                validation_loader=validation_loader,
                adapter=RecoveryTaskAdapter(scaler),
                optimizer=optimizer,
                device=torch.device("cpu"),
                max_epochs=2,
                early_stopping=EarlyStopping(
                    mode="min", patience=8, monitor_name="validation_standardized_huber_loss"
                ),
                checkpoint_path=checkpoint,
                checkpoint_metadata=metadata,
                logger=JsonlRunLogger(root / "training.jsonl"),
            )
            restored = RecoveryGRU(legal_config())
            sidecar = load_checkpoint(
                checkpoint,
                restored,
                expected_tensor_contract_version="tensor_contract_v1",
                expected_feature_schema_version="synthetic_feature_schema_v1",
            )
            restored.eval()
            batch = next(iter(validation_loader))
            with torch.no_grad():
                raw = scaler.inverse_transform(restored(batch))
            self.assertEqual(tuple(raw.shape), (3, 2))
            self.assertTrue(torch.isfinite(raw).all())
            self.assertEqual(sidecar["horizon_order"], list(RECOVERY_HORIZON_ORDER))
            self.assertEqual(sidecar["checkpoint_sha256"], result.checkpoint_sha256)
            events = [json.loads(line) for line in (root / "training.jsonl").read_text().splitlines()]
            self.assertTrue(all("subject_id" not in event for event in events))

    def test_recovery_bundle_verifies_referenced_scaler_hash(self):
        # The configured smoke artifact proves fresh construction and scaler binding.
        root = Path(__file__).resolve().parents[1]
        checkpoint = root / "artifacts/smoke/phase5_synthetic_recovery_gru_v1/recovery_gru_smoke.pt"
        model, scaler, metadata = load_recovery_bundle(
            checkpoint,
            repository_root=root,
            expected_tensor_contract_version="tensor_contract_v1",
            expected_feature_schema_version="synthetic_feature_schema_v1",
        )
        self.assertIsInstance(model, RecoveryGRU)
        self.assertEqual(scaler.source_partition, "train")
        self.assertEqual(metadata["horizon_order"], list(RECOVERY_HORIZON_ORDER))

    def test_same_seed_reproduces_initialization_and_first_forward(self):
        batch = next(
            iter(create_dataloader(tensor_dataset("train"), batch_size=64, shuffle=False, seed=9))
        )
        outputs = []
        for _ in range(2):
            set_deterministic_seed(9)
            model = RecoveryGRU(legal_config())
            model.eval()
            outputs.append(model(batch).detach())
        self.assertTrue(torch.equal(outputs[0], outputs[1]))


if __name__ == "__main__":
    unittest.main()
