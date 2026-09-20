"""Independent-horizon standardized Huber loss for recovery."""

from dataclasses import dataclass

import torch
import torch.nn.functional as functional

from data.collate import CanonicalBatch
from preprocess.target_scaler import RecoveryTargetScaler
from training.engine import MaskedLossResult


RECOVERY_LOSS_AGGREGATION = "equal_mean_across_available_horizons"


def masked_recovery_huber_loss(
    predictions: torch.Tensor,
    raw_targets: torch.Tensor,
    eligibility: torch.Tensor,
    scaler: RecoveryTargetScaler,
    *,
    delta: float = 1.0,
) -> MaskedLossResult:
    """Average each eligible horizon, then give available horizons equal weight."""

    if predictions.ndim != 2 or predictions.shape[1] != 2:
        raise ValueError("recovery predictions must have shape [B,2]")
    if raw_targets.shape != predictions.shape or eligibility.shape != predictions.shape:
        raise ValueError("recovery targets and eligibility must match [B,2]")
    if eligibility.dtype is not torch.bool:
        raise ValueError("recovery eligibility must be boolean")
    if delta <= 0:
        raise ValueError("Huber delta must be positive")
    horizon_means = []
    eligible_count = 0
    for horizon_index in range(2):
        mask = eligibility[:, horizon_index]
        count = int(mask.sum().item())
        if count == 0:
            continue
        selected_targets = raw_targets[:, horizon_index][mask]
        if not torch.isfinite(selected_targets).all():
            raise ValueError("eligible recovery targets must be finite")
        standardized_targets = scaler.transform_horizon(
            selected_targets, horizon_index
        )
        selected_predictions = predictions[:, horizon_index][mask]
        losses = functional.huber_loss(
            selected_predictions,
            standardized_targets,
            reduction="none",
            delta=delta,
        )
        horizon_means.append(losses.mean())
        eligible_count += count
    if not horizon_means:
        return MaskedLossResult(loss=None, eligible_count=0)
    return MaskedLossResult(
        loss=torch.stack(horizon_means).mean(), eligible_count=eligible_count
    )


@dataclass(frozen=True)
class RecoveryTaskAdapter:
    scaler: RecoveryTargetScaler
    huber_delta: float = 1.0
    task_name: str = "recovery"
    aggregation: str = RECOVERY_LOSS_AGGREGATION

    def compute_loss(
        self, predictions: torch.Tensor, batch: CanonicalBatch
    ) -> MaskedLossResult:
        raw_targets = torch.stack(
            (batch.targets["recovery24"], batch.targets["recovery48"]), dim=1
        )
        eligibility = torch.stack(
            (
                batch.eligibility["recovery24"],
                batch.eligibility["recovery48"],
            ),
            dim=1,
        )
        return masked_recovery_huber_loss(
            predictions,
            raw_targets,
            eligibility,
            self.scaler,
            delta=self.huber_delta,
        )
