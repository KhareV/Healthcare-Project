"""Eligible-only weighted BCE-with-logits adapter for organ support."""

from dataclasses import dataclass

import torch
import torch.nn.functional as functional

from data.collate import CanonicalBatch
from training.engine import MaskedLossResult


def masked_support_bce_with_logits_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    eligibility: torch.Tensor,
    *,
    pos_weight: float,
) -> MaskedLossResult:
    """Mask first so censored targets have no loss or validation effect."""

    if logits.ndim == 2 and logits.shape[1] == 1:
        logits = logits[:, 0]
    if logits.ndim != 1:
        raise ValueError("support logits must have shape [B] or [B,1]")
    if labels.shape != logits.shape or eligibility.shape != logits.shape:
        raise ValueError("support labels and eligibility must have shape [B]")
    if eligibility.dtype is not torch.bool:
        raise ValueError("support eligibility must be boolean")
    if pos_weight < 0:
        raise ValueError("pos_weight cannot be negative")
    count = int(eligibility.sum().item())
    if count == 0:
        return MaskedLossResult(loss=None, eligible_count=0)
    selected_labels = labels[eligibility]
    if not torch.isfinite(selected_labels).all():
        raise ValueError("eligible support labels must be finite")
    if not torch.all((selected_labels == 0) | (selected_labels == 1)):
        raise ValueError("eligible support labels must be exactly 0 or 1")
    selected_logits = logits[eligibility]
    loss = functional.binary_cross_entropy_with_logits(
        selected_logits,
        selected_labels,
        pos_weight=selected_logits.new_tensor(pos_weight),
        reduction="mean",
    )
    return MaskedLossResult(loss=loss, eligible_count=count)


@dataclass(frozen=True)
class OrganSupportTaskAdapter:
    pos_weight: float
    task_name: str = "organ_support"

    def compute_loss(
        self, predictions: torch.Tensor, batch: CanonicalBatch
    ) -> MaskedLossResult:
        return masked_support_bce_with_logits_loss(
            predictions,
            batch.targets["organ_support"],
            batch.eligibility["organ_support"],
            pos_weight=self.pos_weight,
        )
