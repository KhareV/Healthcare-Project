import json
import tempfile
import unittest
from pathlib import Path

import torch

from data.collate import create_dataloader
from models.gru_icu_time import (
    ICU_TIME_RAW_OUTPUT,
    ICUTimeGRU,
    load_icu_time_bundle,
)
from models.icu_time_postprocess import (
    ICU_TIME_POSTPROCESS_VERSION,
    remaining_icu_hours_from_log_prediction,
)
from phase4_helpers import SYNTHETIC_SPLIT, tensor_dataset
from test_gru_base import legal_config
from training.early_stopping import EarlyStopping
from training.engine import create_optimizer, train_with_early_stopping
from training.logging import JsonlRunLogger
from training.reproducibility import set_deterministic_seed
from training.tasks.icu_time import ICUTimeTaskAdapter
from vedant_infra.hashing import sha256_file


class ICUTimeTrainingTests(unittest.TestCase):
    def test_training_checkpoint_bundle_and_postprocessed_hours(self):
        seed = 66
        set_deterministic_seed(seed)
        train_loader = create_dataloader(
            tensor_dataset("train"), batch_size=64, shuffle=True, seed=seed
        )
        validation_loader = create_dataloader(
            tensor_dataset("validation"), batch_size=64, shuffle=False, seed=seed
        )
        model = ICUTimeGRU(legal_config())
        optimizer = create_optimizer(
            model.parameters(), name="adamw", learning_rate=0.001, weight_decay=1e-5
        )
        metadata = {
            "run_id": "phase6-unit-smoke",
            "task": "icu_stay_time",
            "model_family": "gru",
            "model_config": model.model_config,
            "raw_output_meaning": ICU_TIME_RAW_OUTPUT,
            "postprocess_version": ICU_TIME_POSTPROCESS_VERSION,
            "seed": seed,
            "epoch": 0,
            "config_hash": "a" * 64,
            "tensor_contract_version": "tensor_contract_v1",
            "feature_schema_version": "synthetic_feature_schema_v1",
            "split_hash": sha256_file(SYNTHETIC_SPLIT),
            "code_commit": "UNAVAILABLE_NOT_GIT_REPOSITORY",
            "validation_value": 0.0,
            "synthetic_smoke_test": True,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "icu_time.pt"
            result = train_with_early_stopping(
                model=model,
                train_loader=train_loader,
                validation_loader=validation_loader,
                adapter=ICUTimeTaskAdapter(),
                optimizer=optimizer,
                device=torch.device("cpu"),
                max_epochs=2,
                early_stopping=EarlyStopping(
                    mode="min", patience=8, monitor_name="validation_log_space_huber_loss"
                ),
                checkpoint_path=checkpoint,
                checkpoint_metadata=metadata,
                logger=JsonlRunLogger(root / "training.jsonl"),
            )
            restored, sidecar = load_icu_time_bundle(
                checkpoint,
                expected_tensor_contract_version="tensor_contract_v1",
                expected_feature_schema_version="synthetic_feature_schema_v1",
            )
            restored.eval()
            with torch.no_grad():
                raw = restored(next(iter(validation_loader)))
                hours = remaining_icu_hours_from_log_prediction(raw)
            self.assertEqual(tuple(raw.shape), (3, 1))
            self.assertTrue(torch.isfinite(hours).all())
            self.assertTrue(torch.all(hours >= 0))
            self.assertEqual(sidecar["checkpoint_sha256"], result.checkpoint_sha256)
            events = [json.loads(line) for line in (root / "training.jsonl").read_text().splitlines()]
            self.assertTrue(all("subject_id" not in event for event in events))

    def test_same_seed_reproduces_initialization_and_forward(self):
        batch = next(
            iter(create_dataloader(tensor_dataset("train"), batch_size=64, shuffle=False, seed=7))
        )
        outputs = []
        for _ in range(2):
            set_deterministic_seed(7)
            model = ICUTimeGRU(legal_config())
            model.eval()
            outputs.append(model(batch).detach())
        self.assertTrue(torch.equal(outputs[0], outputs[1]))


if __name__ == "__main__":
    unittest.main()
