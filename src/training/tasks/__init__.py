"""Task-specific adapters for the shared training engine."""

from training.tasks.recovery import RecoveryTaskAdapter, masked_recovery_huber_loss
from training.tasks.icu_time import ICUTimeTaskAdapter, masked_icu_time_huber_loss
from training.tasks.organ_support import (
    OrganSupportTaskAdapter,
    masked_support_bce_with_logits_loss,
)

__all__ = [
    "RecoveryTaskAdapter",
    "masked_recovery_huber_loss",
    "ICUTimeTaskAdapter",
    "masked_icu_time_huber_loss",
    "OrganSupportTaskAdapter",
    "masked_support_bce_with_logits_loss",
]
