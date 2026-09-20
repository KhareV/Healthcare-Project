"""Pure semantic checks for the upstream-owned current-ICU-time target."""

from datetime import datetime

import torch


def current_icu_remaining_hours(prediction_time: datetime, icu_outtime: datetime) -> float:
    """Reference the frozen current-ICU boundary; not a production label builder."""

    if not isinstance(prediction_time, datetime) or not isinstance(icu_outtime, datetime):
        raise ValueError("prediction_time and icu_outtime must be datetime values")
    hours = (icu_outtime - prediction_time).total_seconds() / 3600.0
    if hours < 0:
        raise ValueError("remaining current ICU hours cannot be negative")
    return hours


def log1p_remaining_icu_hours(remaining_hours: torch.Tensor) -> torch.Tensor:
    """Apply the deterministic target transform without fitted statistics."""

    if not torch.is_floating_point(remaining_hours):
        raise ValueError("remaining ICU hours must be floating point")
    if not torch.isfinite(remaining_hours).all():
        raise ValueError("remaining ICU hours must be finite")
    if torch.any(remaining_hours < 0):
        raise ValueError("remaining ICU hours cannot be negative")
    return torch.log1p(remaining_hours)


def validate_log1p_icu_time_targets(targets: torch.Tensor) -> None:
    """Fail closed when an upstream canonical log target is outside its domain."""

    if not torch.is_floating_point(targets) or not torch.isfinite(targets).all():
        raise ValueError("eligible ICU-time log targets must be finite floating point")
    if torch.any(targets < 0):
        raise ValueError("log1p remaining ICU-time targets cannot be negative")
    reconstructed = torch.expm1(targets)
    if not torch.allclose(torch.log1p(reconstructed), targets):
        raise ValueError("ICU-time target fails log1p/expm1 roundtrip")
