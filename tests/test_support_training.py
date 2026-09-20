import json
import tempfile
import unittest
from pathlib import Path

import torch

from data.collate import create_dataloader
from models.gru_support import (
    SUPPORT_OUTPUT_TYPE,
    SUPPORT_PROBABILITY_TRANSFORM,
    OrganSupportGRU,
    load_support_bundle,
    uncalibrated_support_probability,
)
from phase4_helpers import SYNTHETIC_SPLIT
from phase7_helpers import support_contract, support_dataset
from test_gru_base import legal_config
from training.class_weights import fit_support_class_weight
from training.early_stopping import EarlyStopping
from training.engine import create_optimizer, train_with_early_stopping
from training.logging import JsonlRunLogger
from training.reproducibility import set_deterministic_seed
from training.tasks.organ_support import OrganSupportTaskAdapter
from vedant_infra.hashing import sha256_file


class SupportTrainingTests(unittest.TestCase):
    def test_training_bundle_reload_and_raw_probability(self):
        seed = 77
        set_deterministic_seed(seed)
        train = support_dataset("train")
        validation = support_dataset("validation")
        weight_loader = create_dataloader(train, batch_size=64, shuffle=False, seed=seed)
        contract = support_contract()
        weight = fit_support_class_weight(
            weight_loader,
            split_hash=sha256_file(SYNTHETIC_SPLIT),
            label_contract_version=contract.contract_version,
            event_dictionary_version=contract.event_dictionary_version,
            source_config_hash="a" * 64,
        )
        train_loader = create_dataloader(train, batch_size=64, shuffle=True, seed=seed)
        validation_loader = create_dataloader(validation, batch_size=64, shuffle=False, seed=seed)
        model = OrganSupportGRU(legal_config())
        optimizer = create_optimizer(model.parameters(), name="adamw", learning_rate=0.001, weight_decay=1e-5)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            weight_path = root / "weight.json"
            _, weight_hash = weight.save(weight_path)
            metadata = {
                "run_id": "phase7-unit-smoke",
                "task": "organ_support",
                "model_family": "gru",
                "model_config": model.model_config,
                "output_type": SUPPORT_OUTPUT_TYPE,
                "probability_transform": SUPPORT_PROBABILITY_TRANSFORM,
                "class_weight_artifact": "weight.json",
                "class_weight_sha256": weight_hash,
                "label_contract_version": contract.contract_version,
                "event_dictionary_version": contract.event_dictionary_version,
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
            checkpoint = root / "support.pt"
            result = train_with_early_stopping(
                model=model,
                train_loader=train_loader,
                validation_loader=validation_loader,
                adapter=OrganSupportTaskAdapter(weight.pos_weight),
                optimizer=optimizer,
                device=torch.device("cpu"),
                max_epochs=2,
                early_stopping=EarlyStopping(mode="min", patience=8, monitor_name="validation_weighted_bce_loss_smoke_only"),
                checkpoint_path=checkpoint,
                checkpoint_metadata=metadata,
                logger=JsonlRunLogger(root / "training.jsonl"),
            )
            restored, restored_weight, sidecar = load_support_bundle(
                checkpoint,
                repository_root=root,
                expected_tensor_contract_version="tensor_contract_v1",
                expected_feature_schema_version="synthetic_feature_schema_v1",
            )
            restored.eval()
            with torch.no_grad():
                logits = restored(next(iter(validation_loader)))
                probability = uncalibrated_support_probability(logits)
            self.assertEqual(tuple(logits.shape), (3, 1))
            self.assertTrue(torch.all((probability >= 0) & (probability <= 1)))
            self.assertEqual(restored_weight, weight)
            self.assertEqual(sidecar["checkpoint_sha256"], result.checkpoint_sha256)
            events = [json.loads(line) for line in (root / "training.jsonl").read_text().splitlines()]
            self.assertTrue(all("subject_id" not in event for event in events))

    def test_same_seed_reproduces_initialization_and_forward(self):
        batch = next(iter(create_dataloader(support_dataset("train"), batch_size=64, shuffle=False, seed=8)))
        outputs = []
        for _ in range(2):
            set_deterministic_seed(8)
            model = OrganSupportGRU(legal_config())
            model.eval()
            outputs.append(model(batch).detach())
        self.assertTrue(torch.equal(outputs[0], outputs[1]))


if __name__ == "__main__":
    unittest.main()
