"""Horizon-specific complete-component SOFA sensitivity evaluation."""

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence, Tuple

from evaluation.metrics import PredictionRecord, evaluate_recovery_horizon


SOFA_COMPONENT_COUNT = 6
COMPONENT_NAMES = (
    "respiratory",
    "coagulation",
    "liver",
    "cardiovascular",
    "central_nervous_system",
    "renal",
)


class SensitivityValidationError(ValueError):
    """Raised for invalid or contaminated complete-component evidence."""


@dataclass(frozen=True)
class CompleteComponentRecord:
    stay_id: str
    prediction_time: str
    partition: str
    target: float
    prediction: float
    eligible: bool
    baseline_component_observed: Tuple[bool, ...]
    future_component_observed: Tuple[bool, ...]


@dataclass(frozen=True)
class SensitivityHorizonResult:
    horizon: str
    full: object
    complete_component: Optional[object]
    complete_N_examples: int
    complete_N_ICU_stays: int
    subset_fraction: float
    status: str


@dataclass(frozen=True)
class CompleteComponentSensitivityResult:
    recovery24: SensitivityHorizonResult
    recovery48: SensitivityHorizonResult
    analysis_type: str = "evaluation_subset_only_no_retraining"
    test_accessed: bool = False


def _validate_records(
    records: Sequence[CompleteComponentRecord],
    horizon: str,
    expected_partition: str,
) -> None:
    if horizon not in ("24h", "48h"):
        raise SensitivityValidationError("horizon must be 24h or 48h")
    seen = set()
    for record in records:
        if record.partition != expected_partition:
            if expected_partition == "validation" and record.partition == "test":
                raise SensitivityValidationError(
                    "HARD FAIL — PHASE 14 TEST ACCESS FORBIDDEN"
                )
            raise SensitivityValidationError(
                "complete-component record partition does not match authorized evaluation"
            )
        key = (record.stay_id, record.prediction_time)
        if key in seen:
            raise SensitivityValidationError("duplicate canonical sensitivity row")
        seen.add(key)
        for flags in (
            record.baseline_component_observed,
            record.future_component_observed,
        ):
            if len(flags) != SOFA_COMPONENT_COUNT or any(
                not isinstance(value, bool) for value in flags
            ):
                raise SensitivityValidationError(
                    "all six authoritative SOFA component-observed flags are required"
                )


def complete_component_included(record: CompleteComponentRecord) -> bool:
    return (
        record.eligible
        and all(record.baseline_component_observed)
        and all(record.future_component_observed)
    )


def _evaluate_horizon(
    records: Sequence[CompleteComponentRecord],
    horizon: str,
    expected_partition: str,
):
    _validate_records(records, horizon, expected_partition)
    full_records = tuple(
        PredictionRecord(
            stay_id=record.stay_id,
            prediction_time=record.prediction_time,
            target=record.target,
            prediction=record.prediction,
            eligible=record.eligible,
        )
        for record in records
    )
    full = evaluate_recovery_horizon(full_records, horizon=horizon)
    subset_source = tuple(record for record in records if complete_component_included(record))
    if not subset_source:
        return SensitivityHorizonResult(
            horizon=horizon,
            full=full,
            complete_component=None,
            complete_N_examples=0,
            complete_N_ICU_stays=0,
            subset_fraction=0.0,
            status="EMPTY_COMPLETE_COMPONENT_SUBSET",
        )
    subset = tuple(
        PredictionRecord(
            stay_id=record.stay_id,
            prediction_time=record.prediction_time,
            target=record.target,
            prediction=record.prediction,
            eligible=True,
        )
        for record in subset_source
    )
    complete = evaluate_recovery_horizon(subset, horizon=horizon)
    return SensitivityHorizonResult(
        horizon=horizon,
        full=full,
        complete_component=complete,
        complete_N_examples=complete.counts["N_examples"],
        complete_N_ICU_stays=complete.counts["N_ICU_stays"],
        subset_fraction=complete.counts["N_examples"] / full.counts["N_examples"],
        status="COMPLETE",
    )


def evaluate_complete_component_sensitivity(
    recovery24: Sequence[CompleteComponentRecord],
    recovery48: Sequence[CompleteComponentRecord],
    *,
    expected_partition: str = "validation",
) -> CompleteComponentSensitivityResult:
    """Subset the same predictions independently per horizon and re-evaluate."""

    if expected_partition not in ("validation", "test"):
        raise SensitivityValidationError("evaluation partition must be validation or test")

    return CompleteComponentSensitivityResult(
        recovery24=_evaluate_horizon(recovery24, "24h", expected_partition),
        recovery48=_evaluate_horizon(recovery48, "48h", expected_partition),
    )
