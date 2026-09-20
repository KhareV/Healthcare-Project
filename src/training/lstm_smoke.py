"""Synthetic fixed-cell LSTM sensitivity smoke runner for all three tasks."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

import torch

from data.collate import create_dataloader
from data.dataset import CanonicalTensorDataset, SyntheticTensorizationPolicy
from data.schema import FeatureSchemaReference
from data.support_labels import SupportLabelOverlayDataset, load_synthetic_support_contract
from models.gru import GRUEncoderConfig
from models.gru_icu_time import ICUTimeGRU
from models.gru_recovery import RecoveryGRU
from models.gru_support import OrganSupportGRU, uncalibrated_support_probability
from models.icu_time_postprocess import remaining_icu_hours_from_log_prediction
from models.lstm import (
    ICUTimeLSTM,
    OrganSupportLSTM,
    RecoveryLSTM,
    trainable_parameter_count,
)
from preprocess.target_scaler import RecoveryTargetScaler
from training.checkpoint import load_checkpoint
from training.class_weights import SupportClassWeight
from training.early_stopping import EarlyStopping
from training.engine import create_optimizer, train_with_early_stopping
from training.logging import JsonlRunLogger
from training.lstm_comparison import load_and_validate_comparison
from training.reproducibility import resolve_device, set_deterministic_seed
from training.tasks.icu_time import ICUTimeTaskAdapter
from training.tasks.organ_support import OrganSupportTaskAdapter
from training.tasks.recovery import RecoveryTaskAdapter
from vedant_infra.hashing import sha256_file
from vedant_infra.registry import make_registry_record, upsert_registry_record


def _dataset(config, schema, policy, partition):
    base = CanonicalTensorDataset.from_fixture(
        config["fixture_path"],
        schema,
        partition,
        policy,
        config["synthetic_split_path"],
    )
    if config["task"] == "organ_support":
        contract = load_synthetic_support_contract(config["support_label_path"])
        return SupportLabelOverlayDataset(base, contract)
    return base


def _task_components(config, model_config):
    task = config["task"]
    if task == "recovery":
        preprocessing = config["preprocessing"]
        scaler_path = Path(preprocessing["target_scaler_ref"])
        if sha256_file(scaler_path) != preprocessing["target_scaler_sha256"]:
            raise ValueError("recovery target-scaler hash mismatch")
        scaler = RecoveryTargetScaler.load(scaler_path)
        return (
            RecoveryLSTM(model_config),
            RecoveryGRU(model_config),
            RecoveryTaskAdapter(scaler),
            scaler,
        )
    if task == "icu_stay_time":
        return ICUTimeLSTM(model_config), ICUTimeGRU(model_config), ICUTimeTaskAdapter(), None
    if task == "organ_support":
        preprocessing = config["preprocessing"]
        weight_path = Path(preprocessing["class_weight_ref"])
        if sha256_file(weight_path) != preprocessing["class_weight_sha256"]:
            raise ValueError("support class-weight hash mismatch")
        weight = SupportClassWeight.load(weight_path)
        return (
            OrganSupportLSTM(model_config),
            OrganSupportGRU(model_config),
            OrganSupportTaskAdapter(weight.pos_weight),
            weight,
        )
    raise ValueError("unsupported LSTM smoke task")


def run_lstm_smoke(config_path: Path) -> Mapping[str, object]:
    config, _source = load_and_validate_comparison(config_path)
    if config["scientific_sensitivity"] is not False:
        raise ValueError("this command is synthetic smoke only")
    if sha256_file(config["source_gru_checkpoint_ref"]) != config["source_gru_checkpoint_hash"]:
        raise ValueError("source GRU checkpoint hash mismatch")
    training = config["mapped_training"]
    seed = int(training["seed"])
    set_deterministic_seed(seed)
    device = resolve_device(str(training["device"]))
    schema = FeatureSchemaReference(
        version=config["feature_schema_version"],
        feature_names=tuple(config["feature_names"]),
        status=config["feature_schema_status"],
    )
    policy = SyntheticTensorizationPolicy(
        missing_numeric_fill_value=float(config["synthetic_missing_fill_value"])
    )
    train_dataset = _dataset(config, schema, policy, "train")
    validation_dataset = _dataset(config, schema, policy, "validation")
    batch_size = int(training["batch_size"])
    workers = int(training["num_workers"])
    train_loader = create_dataloader(
        train_dataset, batch_size=batch_size, shuffle=True, seed=seed, num_workers=workers
    )
    validation_loader = create_dataloader(
        validation_dataset, batch_size=batch_size, shuffle=False, seed=seed, num_workers=workers
    )
    model_config = GRUEncoderConfig(**config["model"])
    model, source_gru_model, adapter, _task_artifact = _task_components(config, model_config)
    lstm_parameter_count = trainable_parameter_count(model)
    gru_parameter_count = trainable_parameter_count(source_gru_model)
    optimizer_config = training["optimizer"]
    optimizer = create_optimizer(
        model.parameters(),
        name=optimizer_config["name"],
        learning_rate=float(optimizer_config["learning_rate"]),
        weight_decay=float(optimizer_config["weight_decay"]),
    )
    early_config = training["early_stopping"]
    early_stopping = EarlyStopping(
        mode=early_config["mode"],
        patience=int(early_config["patience"]),
        min_delta=float(early_config["min_delta"]),
        monitor_name=early_config["monitor"],
    )
    output_directory = Path(config["output_directory"])
    output_directory.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_directory / config["checkpoint_filename"]
    metrics_path = output_directory / config["metrics_filename"]
    config_hash = sha256_file(config_path)
    metadata = {
        "run_id": config["run_id"],
        "task": config["task"],
        "model_family": "lstm",
        "recurrent_cell": "lstm",
        "sensitivity_only": True,
        "scientific_sensitivity": False,
        "run_type": "sensitivity_smoke",
        "source_selected_gru_run_id": None,
        "source_gru_run_id": config["source_gru_run_id"],
        "source_gru_config_hash": config["source_gru_config_hash"],
        "source_gru_checkpoint_hash": config["source_gru_checkpoint_hash"],
        "mapped_config": config["model"],
        "mapped_training": config["mapped_training"],
        "gru_trainable_parameter_count": gru_parameter_count,
        "lstm_trainable_parameter_count": lstm_parameter_count,
        "seed": seed,
        "epoch": 0,
        "config_hash": config_hash,
        "tensor_contract_version": config["tensor_contract_version"],
        "feature_schema_version": config["feature_schema_version"],
        "split_hash": sha256_file(config["synthetic_split_path"]),
        "dataset_sha256": config["dataset_sha256"],
        "seed_policy": config["seed_policy"],
        "preprocessing": config["preprocessing"],
        "label_version": config.get("label_version"),
        "code_commit": config["code_commit"],
        "validation_value": 0.0,
        "synthetic_smoke_test": True,
        "participates_in_lstm_search": False,
        "participates_in_serving_selection": False,
    }
    if config["task"] == "organ_support":
        contract = load_synthetic_support_contract(config["support_label_path"])
        metadata["label_version"] = contract.contract_version
        metadata["event_dictionary_version"] = contract.event_dictionary_version
        metadata["class_weight_sha256"] = config["preprocessing"]["class_weight_sha256"]
    result = train_with_early_stopping(
        model=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        adapter=adapter,
        optimizer=optimizer,
        device=device,
        max_epochs=int(training["max_epochs"]),
        early_stopping=early_stopping,
        checkpoint_path=checkpoint_path,
        checkpoint_metadata=metadata,
        logger=JsonlRunLogger(output_directory / config["log_filename"]),
    )
    fresh_model, _, _, fresh_artifact = _task_components(config, model_config)
    load_checkpoint(
        checkpoint_path,
        fresh_model,
        expected_tensor_contract_version=config["tensor_contract_version"],
        expected_feature_schema_version=config["feature_schema_version"],
    )
    fresh_model.to(device)
    fresh_model.eval()
    outputs = []
    with torch.no_grad():
        for batch in validation_loader:
            raw = fresh_model(batch.to(device))
            if config["task"] == "recovery":
                raw = fresh_artifact.inverse_transform(raw)
            elif config["task"] == "icu_stay_time":
                raw = remaining_icu_hours_from_log_prediction(raw)
            else:
                raw = uncalibrated_support_probability(raw)
            outputs.append(raw.cpu())
    predictions = torch.cat(outputs, dim=0)
    expected_width = 2 if config["task"] == "recovery" else 1
    if predictions.shape != (len(validation_dataset), expected_width):
        raise RuntimeError("LSTM smoke prediction shape mismatch")
    if not torch.isfinite(predictions).all():
        raise RuntimeError("LSTM smoke predictions must be finite")
    metrics = {
        "run_id": result.run_id,
        "task": config["task"],
        "model_family": "lstm",
        "run_type": "sensitivity_smoke",
        "sensitivity_only": True,
        "scientific_sensitivity": False,
        "source_selected_gru_run_id": None,
        "source_gru_run_id": config["source_gru_run_id"],
        "source_gru_config_hash": config["source_gru_config_hash"],
        "source_gru_checkpoint_hash": config["source_gru_checkpoint_hash"],
        "checkpoint_sha256": result.checkpoint_sha256,
        "epochs_completed": result.epochs_completed,
        "best_epoch": result.best_epoch,
        "best_validation_smoke_value": result.best_validation_value,
        "prediction_shape": list(predictions.shape),
        "predictions_finite": True,
        "gru_trainable_parameter_count": gru_parameter_count,
        "lstm_trainable_parameter_count": lstm_parameter_count,
        "participates_in_lstm_search": False,
        "participates_in_serving_selection": False,
        "reportable_scientific_performance": False,
        "test_examples_accessed": 0,
        "device": str(device),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    label_version = metadata.get("label_version") or "synthetic_contract_labels_v1"
    record = make_registry_record(
        run_id=config["run_id"],
        timestamp_utc=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        task=config["task"],
        model_family="lstm",
        seed=str(seed),
        code_commit=config["code_commit"],
        config_ref=config_path.as_posix(),
        config_hash=config_hash,
        search_space_hash="",
        split_hash=metadata["split_hash"],
        feature_version=config["feature_schema_version"],
        label_version=label_version,
        model_artifact_ref=checkpoint_path.as_posix(),
        model_sha256=result.checkpoint_sha256,
        metrics_ref=metrics_path.as_posix(),
        status="completed",
        parent_run_id=config["source_gru_run_id"],
        notes="RUN_TYPE=SENSITIVITY_SMOKE; SYNTHETIC; NOT SCIENTIFIC SENSITIVITY; NOT SEARCH; NOT SERVING CANDIDATE",
    )
    upsert_registry_record(config["registry_path"], record)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    arguments = parser.parse_args()
    print(json.dumps(run_lstm_smoke(arguments.config), sort_keys=True))


if __name__ == "__main__":
    main()
