"""Synthetic-only Phase-6 remaining-current-ICU-time GRU smoke command."""

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
from models.gru_icu_time import ICU_TIME_RAW_OUTPUT, ICUTimeGRU, load_icu_time_bundle
from models.icu_time_postprocess import (
    ICU_TIME_POSTPROCESS_VERSION,
    remaining_icu_hours_from_log_prediction,
)
from models.icu_time_target import validate_log1p_icu_time_targets
from training.early_stopping import EarlyStopping
from training.engine import create_optimizer, train_with_early_stopping
from training.logging import JsonlRunLogger
from training.reproducibility import resolve_device, set_deterministic_seed
from training.tasks.icu_time import ICUTimeTaskAdapter
from vedant_infra.hashing import sha256_file
from vedant_infra.registry import make_registry_record, upsert_registry_record


def _load_config(path: Path) -> Mapping[str, object]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("synthetic_smoke_test") is not True:
        raise ValueError("ICU-time smoke config must be explicitly synthetic")
    if config.get("task") != "icu_stay_time" or config.get("model_family") != "gru":
        raise ValueError("Phase-6 smoke must identify icu_stay_time / gru")
    if config.get("participates_in_30_config_search") is not False:
        raise ValueError("Phase-6 smoke cannot participate in the later search")
    contract = json.loads(Path(str(config["tensor_contract_path"])).read_text())
    if contract.get("contract_version") != config.get("tensor_contract_version"):
        raise ValueError("tensor contract artifact/version mismatch")
    if contract.get("sequence", {}).get("sequence_length") != 8:
        raise ValueError("ICU-time GRU requires the frozen eight-bin contract")
    if int(config["max_epochs"]) > 60:
        raise ValueError("GRU training cap is 60 epochs per configuration")
    if str(config["optimizer"]["name"]) != "adamw":
        raise ValueError("frozen GRU optimizer family is AdamW")
    if config.get("postprocess_version") != ICU_TIME_POSTPROCESS_VERSION:
        raise ValueError("ICU-time postprocess version mismatch")
    return config


def _validate_canonical_targets(loader) -> int:
    eligible_count = 0
    for batch in loader:
        mask = batch.eligibility["icu_time"]
        validate_log1p_icu_time_targets(batch.targets["icu_time"][mask])
        eligible_count += int(mask.sum().item())
    if eligible_count == 0:
        raise ValueError("synthetic partition has no eligible ICU-time targets")
    return eligible_count


def run_icu_time_smoke(config_path: Path) -> Mapping[str, object]:
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
    train_dataset = CanonicalTensorDataset.from_fixture(
        str(config["fixture_path"]),
        schema,
        "train",
        policy,
        str(config["synthetic_split_path"]),
    )
    validation_dataset = CanonicalTensorDataset.from_fixture(
        str(config["fixture_path"]),
        schema,
        "validation",
        policy,
        str(config["synthetic_split_path"]),
    )
    batch_size = int(config["batch_size"])
    workers = int(config["num_workers"])
    validation_loader = create_dataloader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
        num_workers=workers,
    )
    target_check_loader = create_dataloader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
        num_workers=workers,
    )
    train_eligible_count = _validate_canonical_targets(target_check_loader)
    validation_eligible_count = _validate_canonical_targets(validation_loader)
    train_loader = create_dataloader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
        num_workers=workers,
    )

    model_config = GRUEncoderConfig(**config["model"])
    model = ICUTimeGRU(model_config)
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
    output_directory = Path(str(config["output_directory"]))
    output_directory.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_directory / str(config["checkpoint_filename"])
    metrics_path = output_directory / str(config["metrics_filename"])
    split_hash = sha256_file(str(config["synthetic_split_path"]))
    checkpoint_metadata = {
        "run_id": config["run_id"],
        "task": "icu_stay_time",
        "model_family": "gru",
        "model_config": model.model_config,
        "raw_output_meaning": ICU_TIME_RAW_OUTPUT,
        "postprocess_version": ICU_TIME_POSTPROCESS_VERSION,
        "seed": seed,
        "epoch": 0,
        "config_hash": sha256_file(config_path),
        "tensor_contract_version": config["tensor_contract_version"],
        "feature_schema_version": config["feature_schema_version"],
        "split_hash": split_hash,
        "loss": "huber_on_log1p_remaining_current_icu_hours",
        "target_scaler": None,
        "code_commit": config["code_commit"],
        "validation_monitor_name": early_stopping.monitor_name,
        "validation_value": 0.0,
        "synthetic_smoke_test": True,
        "participates_in_30_config_search": False,
    }
    result = train_with_early_stopping(
        model=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        adapter=ICUTimeTaskAdapter(huber_delta=float(config["loss"]["delta"])),
        optimizer=optimizer,
        device=device,
        max_epochs=int(config["max_epochs"]),
        early_stopping=early_stopping,
        checkpoint_path=checkpoint_path,
        checkpoint_metadata=checkpoint_metadata,
        logger=JsonlRunLogger(output_directory / str(config["log_filename"])),
    )
    model, loaded_metadata = load_icu_time_bundle(
        checkpoint_path,
        expected_tensor_contract_version=str(config["tensor_contract_version"]),
        expected_feature_schema_version=str(config["feature_schema_version"]),
    )
    if loaded_metadata["checkpoint_sha256"] != result.checkpoint_sha256:
        raise RuntimeError("fresh-loaded ICU-time checkpoint hash changed")
    model.to(device)
    model.eval()
    predicted_hours = []
    with torch.no_grad():
        for batch in validation_loader:
            raw_log = model(batch.to(device))
            predicted_hours.append(
                remaining_icu_hours_from_log_prediction(raw_log).cpu()
            )
    hours = torch.cat(predicted_hours, dim=0)
    if hours.shape != (len(validation_dataset), 1):
        raise RuntimeError("postprocessed ICU-time prediction shape is invalid")
    if not torch.isfinite(hours).all() or torch.any(hours < 0):
        raise RuntimeError("postprocessed ICU-time hours must be finite and nonnegative")

    metrics = {
        "run_id": result.run_id,
        "task": "icu_stay_time",
        "model_family": "gru",
        "synthetic_smoke_test": True,
        "participates_in_30_config_search": False,
        "reportable_scientific_performance": False,
        "epochs_completed": result.epochs_completed,
        "best_epoch": result.best_epoch,
        "best_validation_value_log_space_huber": result.best_validation_value,
        "stopped_early": result.stopped_early,
        "checkpoint_sha256": result.checkpoint_sha256,
        "raw_output_meaning": ICU_TIME_RAW_OUTPUT,
        "postprocess_version": ICU_TIME_POSTPROCESS_VERSION,
        "postprocessed_hours_shape": list(hours.shape),
        "postprocessed_hours_finite": True,
        "postprocessed_hours_nonnegative": True,
        "device": str(device),
        "train_examples": len(train_dataset),
        "validation_examples": len(validation_dataset),
        "train_eligible_count": train_eligible_count,
        "validation_eligible_count": validation_eligible_count,
        "test_examples_accessed": 0,
    }
    metrics_path.write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    record = make_registry_record(
        run_id=str(config["run_id"]),
        timestamp_utc=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        task="icu_stay_time",
        model_family="gru",
        seed=str(seed),
        code_commit=str(config["code_commit"]),
        config_ref=config_path.as_posix(),
        config_hash=sha256_file(config_path),
        search_space_hash="",
        split_hash=split_hash,
        feature_version=str(config["feature_schema_version"]),
        label_version=str(config["label_version"]),
        model_artifact_ref=checkpoint_path.as_posix(),
        model_sha256=result.checkpoint_sha256,
        metrics_ref=metrics_path.as_posix(),
        status="completed",
        parent_run_id="",
        notes="PHASE-6 SYNTHETIC ICU-STAY-TIME GRU SMOKE; NOT SEARCH; NOT SCIENTIFIC PERFORMANCE",
    )
    upsert_registry_record(str(config["registry_path"]), record)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    arguments = parser.parse_args()
    print(json.dumps(run_icu_time_smoke(arguments.config), sort_keys=True))


if __name__ == "__main__":
    main()
