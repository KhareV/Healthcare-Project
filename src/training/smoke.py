"""Synthetic-only end-to-end Phase 4 smoke training command."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

import torch

from data.collate import create_dataloader
from data.dataset import CanonicalTensorDataset, SyntheticTensorizationPolicy
from data.schema import FeatureSchemaReference
from training.early_stopping import EarlyStopping
from training.engine import (
    TaskAdapter,
    create_optimizer,
    train_with_early_stopping,
)
from training.logging import JsonlRunLogger
from training.reproducibility import resolve_device, set_deterministic_seed
from vedant_infra.hashing import sha256_file
from vedant_infra.registry import make_registry_record, upsert_registry_record


class SmokeTestModel(torch.nn.Module):
    """Infrastructure-only linear probe; never a research candidate."""

    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.projection = torch.nn.Linear(8 * feature_dim, 1)

    def forward(self, batch) -> torch.Tensor:
        return self.projection(batch.sequence.flatten(start_dim=1)).squeeze(-1)


def _load_config(path: Path) -> Mapping[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("synthetic_smoke_test") is not True:
        raise ValueError("smoke config must be explicitly synthetic")
    if payload.get("model_family") != "smoke_test":
        raise ValueError("smoke config model_family must be smoke_test")
    contract_path = Path(str(payload["tensor_contract_path"]))
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if contract.get("contract_version") != payload.get("tensor_contract_version"):
        raise ValueError("smoke config tensor contract does not match contract artifact")
    if contract.get("sequence", {}).get("sequence_length") != 8:
        raise ValueError("smoke tensor contract must retain sequence length 8")
    return payload


def run_smoke(config_path: Path) -> Mapping[str, object]:
    config = _load_config(config_path)
    seed = int(config["seed"])
    set_deterministic_seed(seed)
    device = resolve_device(str(config["device"]))

    schema = FeatureSchemaReference(
        version=str(config["feature_schema_version"]),
        feature_names=tuple(config["feature_names"]),
        status=str(config["feature_schema_status"]),
    )
    policy = SyntheticTensorizationPolicy(
        missing_numeric_fill_value=float(config["synthetic_missing_fill_value"])
    )
    fixture_path = Path(str(config["fixture_path"]))
    train_dataset = CanonicalTensorDataset.from_fixture(
        fixture_path,
        schema,
        "train",
        policy,
        str(config["synthetic_split_path"]),
    )
    validation_dataset = CanonicalTensorDataset.from_fixture(
        fixture_path,
        schema,
        "validation",
        policy,
        str(config["synthetic_split_path"]),
    )
    batch_size = int(config["batch_size"])
    num_workers = int(config["num_workers"])
    train_loader = create_dataloader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
        num_workers=num_workers,
    )
    validation_loader = create_dataloader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
        num_workers=num_workers,
    )

    model = SmokeTestModel(schema.feature_dim)
    optimizer_config = config["optimizer"]
    optimizer = create_optimizer(
        model.parameters(),
        name=str(optimizer_config["name"]),
        learning_rate=float(optimizer_config["learning_rate"]),
        weight_decay=float(optimizer_config["weight_decay"]),
    )
    early_config = config["early_stopping"]
    early_stopping = EarlyStopping(
        mode=str(early_config["mode"]),
        patience=int(early_config["patience"]),
        min_delta=float(early_config["min_delta"]),
        monitor_name=str(early_config["monitor"]),
    )
    adapter = TaskAdapter(
        task_name=str(config["task"]),
        target_key="recovery24",
        eligibility_key="recovery24",
        per_item_loss=lambda prediction, target: (prediction - target) ** 2,
    )

    output_directory = Path(str(config["output_directory"]))
    output_directory.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_directory / str(config["checkpoint_filename"])
    log_path = output_directory / str(config["log_filename"])
    metrics_path = output_directory / str(config["metrics_filename"])
    logger = JsonlRunLogger(log_path)
    config_hash = sha256_file(config_path)
    split_hash = sha256_file(str(config["synthetic_split_path"]))
    checkpoint_metadata = {
        "run_id": config["run_id"],
        "task": config["task"],
        "model_family": "smoke_test",
        "seed": seed,
        "epoch": 0,
        "config_hash": config_hash,
        "tensor_contract_version": config["tensor_contract_version"],
        "feature_schema_version": config["feature_schema_version"],
        "split_hash": split_hash,
        "code_commit": config["code_commit"],
        "validation_value": None,
        "synthetic_smoke_test": True,
    }
    result = train_with_early_stopping(
        model=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        adapter=adapter,
        optimizer=optimizer,
        device=device,
        max_epochs=int(config["max_epochs"]),
        early_stopping=early_stopping,
        checkpoint_path=checkpoint_path,
        checkpoint_metadata=checkpoint_metadata,
        logger=logger,
    )
    metrics = {
        "run_id": result.run_id,
        "synthetic_smoke_test": True,
        "reportable_scientific_performance": False,
        "epochs_completed": result.epochs_completed,
        "best_epoch": result.best_epoch,
        "best_validation_value": result.best_validation_value,
        "stopped_early": result.stopped_early,
        "checkpoint_sha256": result.checkpoint_sha256,
        "device": str(device),
        "train_examples": len(train_dataset),
        "validation_examples": len(validation_dataset),
        "test_examples_accessed": 0,
    }
    metrics_path.write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    timestamp_utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    registry_record = make_registry_record(
        run_id=str(config["run_id"]),
        timestamp_utc=timestamp_utc,
        task=str(config["task"]),
        model_family="smoke_test",
        seed=str(seed),
        code_commit=str(config["code_commit"]),
        config_ref=config_path.as_posix(),
        config_hash=config_hash,
        search_space_hash="",
        split_hash=split_hash,
        feature_version=str(config["feature_schema_version"]),
        label_version=str(config["label_version"]),
        model_artifact_ref=checkpoint_path.as_posix(),
        model_sha256=result.checkpoint_sha256,
        metrics_ref=metrics_path.as_posix(),
        status="completed",
        parent_run_id="",
        notes="SYNTHETIC SMOKE TEST ONLY; NOT SCIENTIFIC PERFORMANCE",
    )
    upsert_registry_record(str(config["registry_path"]), registry_record)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    arguments = parser.parse_args()
    metrics = run_smoke(arguments.config)
    print(json.dumps(metrics, sort_keys=True))


if __name__ == "__main__":
    main()
