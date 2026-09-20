"""Synthetic-only Phase-5 recovery GRU smoke command."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

import torch

from data.collate import create_dataloader
from data.dataset import CanonicalTensorDataset, SyntheticTensorizationPolicy
from data.schema import FeatureSchemaReference
from models.gru import GRUEncoderConfig
from models.gru_recovery import (
    RECOVERY_HORIZON_ORDER,
    RecoveryGRU,
    load_recovery_bundle,
)
from preprocess.target_scaler import fit_recovery_target_scaler
from training.early_stopping import EarlyStopping
from training.engine import create_optimizer, train_with_early_stopping
from training.logging import JsonlRunLogger
from training.reproducibility import resolve_device, set_deterministic_seed
from training.tasks.recovery import RECOVERY_LOSS_AGGREGATION, RecoveryTaskAdapter
from vedant_infra.hashing import sha256_file
from vedant_infra.registry import make_registry_record, upsert_registry_record


def _load_config(path: Path) -> Mapping[str, object]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("synthetic_smoke_test") is not True:
        raise ValueError("recovery smoke config must be explicitly synthetic")
    if config.get("task") != "recovery" or config.get("model_family") != "gru":
        raise ValueError("Phase-5 smoke must identify recovery / gru")
    if config.get("participates_in_30_config_search") is not False:
        raise ValueError("Phase-5 smoke cannot participate in the later search")
    contract = json.loads(Path(str(config["tensor_contract_path"])).read_text())
    if contract.get("contract_version") != config.get("tensor_contract_version"):
        raise ValueError("tensor contract artifact/version mismatch")
    if contract.get("sequence", {}).get("sequence_length") != 8:
        raise ValueError("recovery GRU requires the frozen eight-bin contract")
    if int(config["max_epochs"]) > 60:
        raise ValueError("GRU training cap is 60 epochs per configuration")
    if str(config["optimizer"]["name"]) != "adamw":
        raise ValueError("frozen GRU optimizer family is AdamW")
    return config


def run_recovery_smoke(config_path: Path) -> Mapping[str, object]:
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
    dataset_arguments = (
        str(config["fixture_path"]),
        schema,
    )
    train_dataset = CanonicalTensorDataset.from_fixture(
        *dataset_arguments,
        "train",
        policy,
        str(config["synthetic_split_path"]),
    )
    validation_dataset = CanonicalTensorDataset.from_fixture(
        *dataset_arguments,
        "validation",
        policy,
        str(config["synthetic_split_path"]),
    )
    batch_size = int(config["batch_size"])
    workers = int(config["num_workers"])
    scaler_loader = create_dataloader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
        num_workers=workers,
    )
    split_hash = sha256_file(str(config["synthetic_split_path"]))
    scaler = fit_recovery_target_scaler(scaler_loader, split_hash=split_hash)

    output_directory = Path(str(config["output_directory"]))
    output_directory.mkdir(parents=True, exist_ok=True)
    scaler_path = output_directory / str(config["target_scaler_filename"])
    _, scaler_hash = scaler.save(scaler_path)

    train_loader = create_dataloader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
        num_workers=workers,
    )
    validation_loader = create_dataloader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
        num_workers=workers,
    )
    model_config = GRUEncoderConfig(**config["model"])
    model = RecoveryGRU(model_config)
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
    checkpoint_path = output_directory / str(config["checkpoint_filename"])
    log_path = output_directory / str(config["log_filename"])
    metrics_path = output_directory / str(config["metrics_filename"])
    config_hash = sha256_file(config_path)
    checkpoint_metadata = {
        "run_id": config["run_id"],
        "task": "recovery",
        "model_family": "gru",
        "model_config": model.model_config,
        "seed": seed,
        "epoch": 0,
        "config_hash": config_hash,
        "tensor_contract_version": config["tensor_contract_version"],
        "feature_schema_version": config["feature_schema_version"],
        "split_hash": split_hash,
        "target_scaler_artifact": scaler_path.as_posix(),
        "target_scaler_sha256": scaler_hash,
        "horizon_order": list(RECOVERY_HORIZON_ORDER),
        "loss": "huber",
        "loss_aggregation": RECOVERY_LOSS_AGGREGATION,
        "code_commit": config["code_commit"],
        "validation_monitor_name": early_stopping.monitor_name,
        "validation_value": 0.0,
        "synthetic_smoke_test": True,
        "participates_in_30_config_search": False,
    }
    loss_config = config["loss"]
    result = train_with_early_stopping(
        model=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        adapter=RecoveryTaskAdapter(
            scaler, huber_delta=float(loss_config["delta"])
        ),
        optimizer=optimizer,
        device=device,
        max_epochs=int(config["max_epochs"]),
        early_stopping=early_stopping,
        checkpoint_path=checkpoint_path,
        checkpoint_metadata=checkpoint_metadata,
        logger=JsonlRunLogger(log_path),
    )

    model, scaler, loaded_metadata = load_recovery_bundle(
        checkpoint_path,
        repository_root=Path("."),
        expected_tensor_contract_version=str(config["tensor_contract_version"]),
        expected_feature_schema_version=str(config["feature_schema_version"]),
    )
    if loaded_metadata["checkpoint_sha256"] != result.checkpoint_sha256:
        raise RuntimeError("fresh-loaded recovery checkpoint hash changed")
    model.to(device)
    model.eval()
    inverse_predictions = []
    with torch.no_grad():
        for batch in validation_loader:
            standardized = model(batch.to(device))
            inverse_predictions.append(scaler.inverse_transform(standardized).cpu())
    raw_predictions = torch.cat(inverse_predictions, dim=0)
    if raw_predictions.shape != (len(validation_dataset), 2):
        raise RuntimeError("inverse-transformed recovery prediction shape is invalid")
    if not torch.isfinite(raw_predictions).all():
        raise RuntimeError("inverse-transformed recovery predictions are non-finite")

    metrics = {
        "run_id": result.run_id,
        "task": "recovery",
        "model_family": "gru",
        "synthetic_smoke_test": True,
        "participates_in_30_config_search": False,
        "reportable_scientific_performance": False,
        "epochs_completed": result.epochs_completed,
        "best_epoch": result.best_epoch,
        "best_validation_value_standardized_huber": result.best_validation_value,
        "stopped_early": result.stopped_early,
        "checkpoint_sha256": result.checkpoint_sha256,
        "target_scaler_sha256": scaler_hash,
        "horizon_order": list(RECOVERY_HORIZON_ORDER),
        "inverse_prediction_shape": list(raw_predictions.shape),
        "inverse_predictions_finite": True,
        "raw_delta_predictions_clipped": False,
        "device": str(device),
        "train_examples": len(train_dataset),
        "validation_examples": len(validation_dataset),
        "test_examples_accessed": 0,
    }
    metrics_path.write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    registry_record = make_registry_record(
        run_id=str(config["run_id"]),
        timestamp_utc=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        task="recovery",
        model_family="gru",
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
        notes="PHASE-5 SYNTHETIC RECOVERY GRU SMOKE; NOT SEARCH; NOT SCIENTIFIC PERFORMANCE",
    )
    upsert_registry_record(str(config["registry_path"]), registry_record)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    arguments = parser.parse_args()
    print(json.dumps(run_recovery_smoke(arguments.config), sort_keys=True))


if __name__ == "__main__":
    main()
