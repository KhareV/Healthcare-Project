"""Reusable validation-only early stopping state machine."""

from dataclasses import dataclass
import math
from typing import Optional


@dataclass(frozen=True)
class EarlyStoppingDecision:
    improved: bool
    should_stop: bool


class EarlyStopping:
    def __init__(
        self,
        *,
        mode: str,
        patience: int,
        min_delta: float = 0.0,
        monitor_name: str = "validation_loss",
    ) -> None:
        if mode not in ("min", "max"):
            raise ValueError("mode must be min or max")
        if patience < 1:
            raise ValueError("patience must be at least 1")
        if min_delta < 0:
            raise ValueError("min_delta cannot be negative")
        if not isinstance(monitor_name, str) or not monitor_name.strip():
            raise ValueError("monitor_name must be a non-empty string")
        self.mode = mode
        self.patience = patience
        self.min_delta = min_delta
        self.monitor_name = monitor_name
        self.best_value = None  # type: Optional[float]
        self.best_epoch = None  # type: Optional[int]
        self.bad_epochs = 0

    def _is_improvement(self, value: float) -> bool:
        if self.best_value is None:
            return True
        if self.mode == "min":
            return value < self.best_value - self.min_delta
        return value > self.best_value + self.min_delta

    def update(self, value: float, epoch: int) -> EarlyStoppingDecision:
        if not math.isfinite(value):
            raise ValueError("monitored validation value must be finite")
        if self._is_improvement(value):
            self.best_value = value
            self.best_epoch = epoch
            self.bad_epochs = 0
            return EarlyStoppingDecision(improved=True, should_stop=False)
        self.bad_epochs += 1
        return EarlyStoppingDecision(
            improved=False, should_stop=self.bad_epochs >= self.patience
        )
