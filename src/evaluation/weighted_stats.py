"""Deterministic weighted statistical primitives for Phase 9 metrics."""

import math
from typing import Sequence


class WeightedStatisticError(ValueError):
    """Raised when values or weights cannot define the requested statistic."""


def _validated(values: Sequence[float], weights: Sequence[float]):
    if len(values) != len(weights):
        raise WeightedStatisticError("values and weights must have equal length")
    if not values:
        raise WeightedStatisticError("weighted statistic requires at least one value")
    converted_values = tuple(float(value) for value in values)
    converted_weights = tuple(float(weight) for weight in weights)
    if not all(math.isfinite(value) for value in converted_values):
        raise WeightedStatisticError("values must be finite")
    if not all(math.isfinite(weight) for weight in converted_weights):
        raise WeightedStatisticError("weights must be finite")
    if any(weight < 0 for weight in converted_weights):
        raise WeightedStatisticError("weights cannot be negative")
    total_weight = math.fsum(converted_weights)
    if total_weight <= 0:
        raise WeightedStatisticError("total weight must be positive")
    return converted_values, converted_weights, total_weight


def weighted_mean(values: Sequence[float], weights: Sequence[float]) -> float:
    values, weights, total_weight = _validated(values, weights)
    return math.fsum(value * weight for value, weight in zip(values, weights)) / total_weight


def weighted_rmse(errors: Sequence[float], weights: Sequence[float]) -> float:
    errors, weights, total_weight = _validated(errors, weights)
    mean_squared = math.fsum(
        error * error * weight for error, weight in zip(errors, weights)
    ) / total_weight
    return math.sqrt(mean_squared)


def weighted_percentile(
    values: Sequence[float], weights: Sequence[float], q: float
) -> float:
    """Return the lower/inverted-CDF weighted quantile for q in [0,1].

    UNLOCKED ENGINEERING DETAIL: at an exact cumulative-weight boundary, the
    lower observed value is returned. Zero-weight observations are ignored.
    """

    values, weights, total_weight = _validated(values, weights)
    quantile = float(q)
    if not math.isfinite(quantile) or not 0.0 <= quantile <= 1.0:
        raise WeightedStatisticError("q must be finite and within [0,1]")
    pairs = sorted(
        ((value, weight) for value, weight in zip(values, weights) if weight > 0),
        key=lambda pair: pair[0],
    )
    threshold = quantile * total_weight
    cumulative = 0.0
    for value, weight in pairs:
        cumulative += weight
        if cumulative >= threshold:
            return value
    return pairs[-1][0]


def weighted_median(values: Sequence[float], weights: Sequence[float]) -> float:
    return weighted_percentile(values, weights, 0.5)
