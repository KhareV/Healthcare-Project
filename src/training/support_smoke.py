"""Synthetic-only Phase-7 uncalibrated organ-support GRU smoke command."""

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
from models.gru_support import (
    SUPPORT_OUTPUT_TYPE,
    SUPPORT_PROBABILITY_TRANSFORM,
    OrganSupportGRU,
    load_support_bundle,
    uncalibrated_support_probability,
)
from training.class_weights import fit_support_class_weight
from training.early_stopping import EarlyStopping
from training.engine import create_optimizer, train_with_early_stopping
from training.logging import JsonlRunLogger
from training.reproducibility import resolve_device, set_deterministic_seed
from training.tasks.organ_support import OrganSupportTaskAdapter
from vedant_infra.hashing import sha256_file
from vedant_infra.registry import make_registry_record, upsert_registry_record


def _load_config(path: Path) -> Mapping[str, object]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("synthetic_smoke_test") is not True:
        raise ValueError("support smoke config must be explicitly synthetic")
    if config.get("task") != "organ_support" or config.get("model_family") != "gru":
        raise ValueError("Phase-7 smoke must identify organ_support / gru")
    if config.get("participates_in_30_config_search") is not False:
        raise ValueError("Phase-7 smoke cannot participate in the later search")
    if config.get("calibration") is not None or config.get("operating_threshold") is not None:
        raise ValueError("Phase-7 smoke cannot calibrate or select a threshold")
    contract = json.loads(Path(str(config["tensor_contract_path"])).read_text())
    if contract.get("contract_version") != config.get("tensor_contract_version"):
        raise ValueError("tensor contract artifact/version mismatch")
    if int(config["max_epochs"]) > 60:
        raise ValueError("GRU training cap is 60 epochs per configuration")
    if str(config["optimizer"]["name"]) != "adamw":
        raise ValueError("frozen GRU optimizer family is AdamW")
    return config


def run_support_smoke(config_path: Path) -> Mapping[str, object]:
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
    support_contract = load_synthetic_support_contract(config["support_label_path"])

    def labelled_dataset(partition):
        base = CanonicalTensorDataset.from_fixture(
            str(config["fixture_path"]),
            schema,
            partition,
            policy,
            str(config["synthetic_split_path"]),
        )
        return SupportLabelOverlayDataset(base, support_contract)

    train_dataset = labelled_dataset("train")
    validation_dataset = labelled_dataset("validation")
    batch_size = int(config["batch_size"])
    workers = int(config["num_workers"])
    class_weight_loader = create_dataloader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
        num_workers=workers,
    )
    config_hash = sha256_file(config_path)
    split_hash = sha256_file(str(config["synthetic_split_path"]))
    class_weight = fit_support_class_weight(
        class_weight_loader,
        split_hash=split_hash,
        label_contract_version=support_contract.contract_version,
        event_dictionary_version=support_contract.event_dictionary_version,
        source_config_hash=config_hash,
    )
    output_directory = Path(str(config["output_directory"]))
    output_directory.mkdir(parents=True, exist_ok=True)
    class_weight_path = output_directory / str(config["class_weight_filename"])
    _, class_weight_hash = class_weight.save(class_weight_path)
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
    model = OrganSupportGRU(GRUEncoderConfig(**config["model"]))
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
    metrics_path = output_directory / str(config["metrics_filename"])
    metadata = {
        "run_id": config["run_id"],
        "task": "organ_support",
        "model_family": "gru",
        "model_config": model.model_config,
        "output_type": SUPPORT_OUTPUT_TYPE,
        "probability_transform": SUPPORT_PROBABILITY_TRANSFORM,
        "class_weight_artifact": class_weight_path.as_posix(),
        "class_weight_sha256": class_weight_hash,
        "class_weight_formula_version": class_weight.formula_version,
        "label_contract_version": support_contract.contract_version,
        "event_dictionary_version": support_contract.event_dictionary_version,
        "seed": seed,
        "epoch": 0,
        "config_hash": config_hash,
        "tensor_contract_version": config["tensor_contract_version"],
        "feature_schema_version": config["feature_schema_version"],
        "split_hash": split_hash,
        "code_commit": config["code_commit"],
        "validation_monitor_name": early_stopping.monitor_name,
        "validation_value": 0.0,
        "calibration": None,
        "operating_threshold": None,
        "synthetic_smoke_test": True,
        "participates_in_30_config_search": False,
    }
    result = train_with_early_stopping(
        model=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        adapter=OrganSupportTaskAdapter(pos_weight=class_weight.pos_weight),
        optimizer=optimizer,
        device=device,
        max_epochs=int(config["max_epochs"]),
        early_stopping=early_stopping,
        checkpoint_path=checkpoint_path,
        checkpoint_metadata=metadata,
        logger=JsonlRunLogger(output_directory / str(config["log_filename"])),
    )
    model, loaded_weight, loaded_metadata = load_support_bundle(
        checkpoint_path,
        repository_root=Path("."),
        expected_tensor_contract_version=str(config["tensor_contract_version"]),
        expected_feature_schema_version=str(config["feature_schema_version"]),
    )
    if loaded_metadata["checkpoint_sha256"] != result.checkpoint_sha256:
        raise RuntimeError("fresh-loaded support checkpoint hash changed")
    if loaded_weight != class_weight:
        raise RuntimeError("fresh-loaded support class weight changed")
    model.to(device)
    model.eval()
    probabilities = []
    with torch.no_grad():
        for batch in validation_loader:
            logits = model(batch.to(device))
            probabilities.append(uncalibrated_support_probability(logits).cpu())
    raw_probabilities = torch.cat(probabilities, dim=0)
    if raw_probabilities.shape != (len(validation_dataset), 1):
        raise RuntimeError("support probability shape is invalid")
    if not torch.isfinite(raw_probabilities).all() or torch.any(
        (raw_probabilities < 0) | (raw_probabilities > 1)
    ):
        raise RuntimeError("support probabilities must be finite and in [0,1]")

    metrics = {
        "run_id": result.run_id,
        "task": "organ_support",
        "model_family": "gru",
        "synthetic_smoke_test": True,
        "participates_in_30_config_search": False,
        "reportable_scientific_performance": False,
        "epochs_completed": result.epochs_completed,
        "best_epoch": result.best_epoch,
        "best_validation_value_weighted_bce": result.best_validation_value,
        "stopped_early": result.stopped_early,
        "checkpoint_sha256": result.checkpoint_sha256,
        "class_weight_sha256": class_weight_hash,
        "eligible_train_count": class_weight.eligible_train_count,
        "positive_train_count": class_weight.positive_count,
        "negative_train_count": class_weight.negative_count,
        "pos_weight": class_weight.pos_weight,
        "probability_transform": SUPPORT_PROBABILITY_TRANSFORM,
        "probability_shape": list(raw_probabilities.shape),
        "probabilities_finite_in_unit_interval": True,
        "probabilities_calibrated": False,
        "operating_threshold_selected": False,
        "device": str(device),
        "train_examples": len(train_dataset),
        "validation_examples": len(validation_dataset),
        "test_examples_accessed": 0,
    }
    metrics_path.write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    record = make_registry_record(
        run_id=str(config["run_id"]),
        timestamp_utc=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        task="organ_support",
        model_family="gru",
        seed=str(seed),
        code_commit=str(config["code_commit"]),
        config_ref=config_path.as_posix(),
        config_hash=config_hash,
        search_space_hash="",
        split_hash=split_hash,
        feature_version=str(config["feature_schema_version"]),
        label_version=support_contract.contract_version,
        model_artifact_ref=checkpoint_path.as_posix(),
        model_sha256=result.checkpoint_sha256,
        metrics_ref=metrics_path.as_posix(),
        status="completed",
        parent_run_id="",
        notes="PHASE-7 SYNTHETIC ORGAN-SUPPORT GRU SMOKE; UNCALIBRATED; NO THRESHOLD; NOT SEARCH; NOT SCIENTIFIC PERFORMANCE",
    )
    upsert_registry_record(str(config["registry_path"]), record)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    arguments = parser.parse_args()
    print(json.dumps(run_support_smoke(arguments.config), sort_keys=True))


if __name__ == "__main__":
    main()
