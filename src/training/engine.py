"""Generic masked training/validation engine with no task architecture."""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Optional, Protocol, Tuple

import torch

from data.collate import CanonicalBatch
from training.checkpoint import load_checkpoint, save_checkpoint
from training.early_stopping import EarlyStopping
from training.logging import JsonlRunLogger


class BatchModel(Protocol):
    def __call__(self, batch: CanonicalBatch) -> torch.Tensor:
        ...


class TrainingTaskAdapter(Protocol):
    task_name: str

    def compute_loss(
        self, predictions: torch.Tensor, batch: CanonicalBatch
    ) -> "MaskedLossResult":
        ...


@dataclass(frozen=True)
class TaskAdapter:
    task_name: str
    target_key: str
    eligibility_key: str
    per_item_loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

    def compute_loss(
        self, predictions: torch.Tensor, batch: CanonicalBatch
    ) -> "MaskedLossResult":
        return masked_mean_loss(
            predictions,
            batch.targets[self.target_key],
            batch.eligibility[self.eligibility_key],
            self.per_item_loss,
        )


@dataclass(frozen=True)
class MaskedLossResult:
    loss: Optional[torch.Tensor]
    eligible_count: int


@dataclass(frozen=True)
class EpochResult:
    mean_loss: Optional[float]
    eligible_count: int
    optimized_batches: int
    skipped_zero_eligible_batches: int


@dataclass(frozen=True)
class TrainingResult:
    run_id: str
    epochs_completed: int
    best_epoch: int
    best_validation_value: float
    stopped_early: bool
    checkpoint_path: str
    checkpoint_sha256: str


def masked_mean_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    eligibility: torch.Tensor,
    per_item_loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> MaskedLossResult:
    """Index eligible rows before loss; zero-eligible returns no signal."""

    if predictions.shape != targets.shape or targets.shape != eligibility.shape:
        raise ValueError("predictions, targets, and eligibility must have equal shape")
    if eligibility.dtype is not torch.bool:
        raise ValueError("eligibility must be a boolean tensor")
    eligible_count = int(eligibility.sum().item())
    if eligible_count == 0:
        return MaskedLossResult(loss=None, eligible_count=0)
    eligible_predictions = predictions[eligibility]
    eligible_targets = targets[eligibility]
    if not torch.isfinite(eligible_targets).all():
        raise ValueError("eligible targets must be finite")
    losses = per_item_loss(eligible_predictions, eligible_targets)
    if losses.shape != eligible_targets.shape:
        raise ValueError("per-item loss must preserve eligible target shape")
    return MaskedLossResult(loss=losses.mean(), eligible_count=eligible_count)


def run_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    adapter: TrainingTaskAdapter,
    device: torch.device,
    *,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> EpochResult:
    """Run one training or validation epoch with eligible-only means."""

    training = optimizer is not None
    model.train(training)
    total_weighted_loss = 0.0
    total_eligible = 0
    optimized_batches = 0
    skipped = 0

    for batch in loader:
        if not isinstance(batch, CanonicalBatch):
            raise TypeError("loader must emit CanonicalBatch")
        batch = batch.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            predictions = model(batch)
            result = adapter.compute_loss(predictions, batch)
            if result.loss is None:
                skipped += 1
                continue
            if training:
                result.loss.backward()
                optimizer.step()
                optimized_batches += 1
        total_weighted_loss += float(result.loss.detach().cpu()) * result.eligible_count
        total_eligible += result.eligible_count

    mean_loss = None if total_eligible == 0 else total_weighted_loss / total_eligible
    return EpochResult(
        mean_loss=mean_loss,
        eligible_count=total_eligible,
        optimized_batches=optimized_batches,
        skipped_zero_eligible_batches=skipped,
    )


def create_optimizer(
    parameters,
    *,
    name: str,
    learning_rate: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    """Configurable optimizer factory; values are supplied by run config."""

    if learning_rate <= 0 or weight_decay < 0:
        raise ValueError("optimizer learning_rate/weight_decay are invalid")
    if name == "adamw":
        return torch.optim.AdamW(
            parameters, lr=learning_rate, weight_decay=weight_decay
        )
    if name == "sgd":
        return torch.optim.SGD(
            parameters, lr=learning_rate, weight_decay=weight_decay
        )
    raise ValueError("unsupported optimizer: {}".format(name))


def train_with_early_stopping(
    *,
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    validation_loader: torch.utils.data.DataLoader,
    adapter: TrainingTaskAdapter,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    max_epochs: int,
    early_stopping: EarlyStopping,
    checkpoint_path: Path,
    checkpoint_metadata: Mapping[str, object],
    logger: JsonlRunLogger,
    validation_monitor: Optional[Callable[[EpochResult], float]] = None,
    scheduler=None,
) -> TrainingResult:
    """Run config-driven epochs, save best validation checkpoint, restore it."""

    if max_epochs < 1:
        raise ValueError("max_epochs must be at least 1")
    model.to(device)
    stopped_early = False
    epochs_completed = 0
    best_hash = None

    for epoch in range(1, max_epochs + 1):
        train_result = run_epoch(
            model, train_loader, adapter, device, optimizer=optimizer
        )
        validation_result = run_epoch(
            model, validation_loader, adapter, device, optimizer=None
        )
        if validation_result.mean_loss is None and validation_monitor is None:
            raise RuntimeError("validation has zero eligible examples for monitored task")
        monitor_value = (
            float(validation_monitor(validation_result))
            if validation_monitor is not None
            else float(validation_result.mean_loss)
        )
        decision = early_stopping.update(monitor_value, epoch)
        logger.log(
            {
                "event": "epoch",
                "run_id": checkpoint_metadata["run_id"],
                "epoch": epoch,
                "device": str(device),
                "train_loss": train_result.mean_loss,
                "validation_loss": validation_result.mean_loss,
                "validation_monitor_name": early_stopping.monitor_name,
                "validation_monitor_value": monitor_value,
                "train_eligible_count": train_result.eligible_count,
                "validation_eligible_count": validation_result.eligible_count,
                "skipped_zero_eligible_batches": train_result.skipped_zero_eligible_batches,
                "checkpoint_improved": decision.improved,
            }
        )
        if decision.improved:
            metadata = dict(checkpoint_metadata)
            metadata["epoch"] = epoch
            metadata["validation_monitor_name"] = early_stopping.monitor_name
            metadata["validation_value"] = monitor_value
            _, best_hash = save_checkpoint(
                checkpoint_path, model, optimizer, metadata
            )
            logger.log(
                {
                    "event": "checkpoint",
                    "run_id": checkpoint_metadata["run_id"],
                    "epoch": epoch,
                    "checkpoint_sha256": best_hash,
                }
            )
        if scheduler is not None:
            scheduler.step()
        epochs_completed = epoch
        if decision.should_stop:
            stopped_early = True
            logger.log(
                {
                    "event": "early_stop",
                    "run_id": checkpoint_metadata["run_id"],
                    "epoch": epoch,
                    "best_epoch": early_stopping.best_epoch,
                }
            )
            break

    if early_stopping.best_epoch is None or early_stopping.best_value is None:
        raise RuntimeError("training produced no best validation checkpoint")
    restored = load_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        expected_tensor_contract_version=str(
            checkpoint_metadata["tensor_contract_version"]
        ),
        expected_feature_schema_version=str(
            checkpoint_metadata["feature_schema_version"]
        ),
    )
    best_hash = str(restored["checkpoint_sha256"])
    logger.log(
        {
            "event": "best_checkpoint_restored",
            "run_id": checkpoint_metadata["run_id"],
            "best_epoch": early_stopping.best_epoch,
            "checkpoint_sha256": best_hash,
        }
    )
    return TrainingResult(
        run_id=str(checkpoint_metadata["run_id"]),
        epochs_completed=epochs_completed,
        best_epoch=early_stopping.best_epoch,
        best_validation_value=early_stopping.best_value,
        stopped_early=stopped_early,
        checkpoint_path=str(checkpoint_path),
        checkpoint_sha256=best_hash,
    )
