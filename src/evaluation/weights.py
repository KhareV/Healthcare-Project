"""One authoritative implementation of task-specific ICU-stay weights."""

from dataclasses import dataclass
from typing import Hashable, Sequence, Tuple


class StayWeightError(ValueError):
    """Raised when stay identifiers or eligibility cannot define safe weights."""


@dataclass(frozen=True)
class StayWeightResult:
    weights: Tuple[float, ...]
    eligible_indices: Tuple[int, ...]
    n_examples: int
    n_icu_stays: int


def _valid_stay_id(value: Hashable) -> bool:
    if value is None:
        return False
    if isinstance(value, str) and not value.strip():
        return False
    try:
        hash(value)
    except TypeError:
        return False
    return True


def compute_stay_weights(
    stay_ids: Sequence[Hashable], eligibility: Sequence[bool]
) -> StayWeightResult:
    """Return aligned weights: eligible row j in stay i receives exactly 1/n_i."""

    if len(stay_ids) != len(eligibility):
        raise StayWeightError("stay_ids and eligibility must have equal length")
    counts = {}
    eligible_indices = []
    for index, (stay_id, eligible) in enumerate(zip(stay_ids, eligibility)):
        if not _valid_stay_id(stay_id):
            raise StayWeightError("every prediction record requires a valid stay_id")
        if not isinstance(eligible, bool):
            raise StayWeightError("eligibility values must be boolean")
        if eligible:
            counts[stay_id] = counts.get(stay_id, 0) + 1
            eligible_indices.append(index)
    weights = tuple(
        1.0 / counts[stay_id] if eligible else 0.0
        for stay_id, eligible in zip(stay_ids, eligibility)
    )
    return StayWeightResult(
        weights=weights,
        eligible_indices=tuple(eligible_indices),
        n_examples=len(eligible_indices),
        n_icu_stays=len(counts),
    )
