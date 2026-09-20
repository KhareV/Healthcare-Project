"""Eligible-only Huber loss on log1p remaining current-ICU hours."""

from dataclasses import dataclass

import torch
import torch.nn.functional as functional

from data.collate import CanonicalBatch
from models.icu_time_target import validate_log1p_icu_time_targets
from training.engine import MaskedLossResult


def masked_icu_time_huber_loss(
    predictions: torch.Tensor,
    log_targets: torch.Tensor,
    eligibility: torch.Tensor,
    *,
    delta: float = 1.0,
) -> MaskedLossResult:
    """Mask before validation/loss so ineligible values have no influence."""

    if predictions.ndim == 2 and predictions.shape[1] == 1:
        predictions = predictions[:, 0]
    if predictions.ndim != 1:
        raise ValueError("ICU-time predictions must have shape [B] or [B,1]")
    if log_targets.shape != predictions.shape or eligibility.shape != predictions.shape:
        raise ValueError("ICU-time target and eligibility must have shape [B]")
    if eligibility.dtype is not torch.bool:
        raise ValueError("ICU-time eligibility must be boolean")
    if delta <= 0:
        raise ValueError("Huber delta must be positive")
    eligible_count = int(eligibility.sum().item())
    if eligible_count == 0:
        return MaskedLossResult(loss=None, eligible_count=0)
    selected_targets = log_targets[eligibility]
    validate_log1p_icu_time_targets(selected_targets)
    losses = functional.huber_loss(
        predictions[eligibility], selected_targets, reduction="none", delta=delta
    )
    return MaskedLossResult(loss=losses.mean(), eligible_count=eligible_count)


@dataclass(frozen=True)
class ICUTimeTaskAdapter:
    huber_delta: float = 1.0
    task_name: str = "icu_stay_time"

    def compute_loss(
        self, predictions: torch.Tensor, batch: CanonicalBatch
    ) -> MaskedLossResult:
        return masked_icu_time_huber_loss(
            predictions,
            batch.targets["icu_time"],
            batch.eligibility["icu_time"],
            delta=self.huber_delta,
        )
