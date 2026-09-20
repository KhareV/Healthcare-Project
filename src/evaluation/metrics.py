"""Stay-balanced point estimators shared by every model family."""

import math
from dataclasses import dataclass
from typing import Hashable, Mapping, Optional, Sequence, Tuple

from evaluation.weighted_stats import (
    weighted_mean,
    weighted_median,
    weighted_percentile,
    weighted_rmse,
)
from evaluation.weights import compute_stay_weights


METRIC_IMPLEMENTATION_VERSION = "stay_balanced_metrics_v1"
DIRECTIONAL_ZERO_BLOCKER = "BLOCKED — DIRECTIONAL AGREEMENT ZERO/TIE SEMANTICS REQUIRED"
AUPRC_CONVENTION = "weighted_average_precision_step_integral_v1"
AUPRC_REVIEW_BLOCKER = (
    "BLOCKED — AUPRC INTEGRATION CONVENTION REVIEW REQUIRED BEFORE REAL MODEL SELECTION"
)
WEIGHTED_QUANTILE_CONVENTION = "lower_inverted_cdf_first_cumulative_weight_gte_q_v1"


class MetricInputError(ValueError):
    """Raised for corrupt, ambiguous, or non-finite prediction records."""


class EmptyEligibleSetError(MetricInputError):
    """Raised when a task/horizon has no eligible prediction records."""


class DirectionalAgreementSemanticsError(MetricInputError):
    """Raised because exact-zero direction semantics are not frozen."""


@dataclass(frozen=True)
class PredictionRecord:
    stay_id: Hashable
    prediction_time: str
    target: float
    prediction: float
    eligible: bool


@dataclass(frozen=True)
class EvaluationResult:
    metrics: Mapping[str, Optional[float]]
    counts: Mapping[str, int]
    metadata: Mapping[str, object]


def _validate_unique_records(records: Sequence[PredictionRecord]) -> None:
    seen = set()
    for record in records:
        if not isinstance(record, PredictionRecord):
            raise MetricInputError("evaluation inputs must be PredictionRecord instances")
        if not isinstance(record.prediction_time, str) or not record.prediction_time.strip():
            raise MetricInputError("every prediction record requires prediction_time")
        key = (record.stay_id, record.prediction_time)
        if key in seen:
            raise MetricInputError(
                "duplicate canonical prediction record: stay_id + prediction_time"
            )
        seen.add(key)


def _eligible_view(records: Sequence[PredictionRecord]):
    _validate_unique_records(records)
    weight_result = compute_stay_weights(
        [record.stay_id for record in records], [record.eligible for record in records]
    )
    if weight_result.n_examples == 0:
        raise EmptyEligibleSetError("no eligible prediction records")
    eligible = tuple(records[index] for index in weight_result.eligible_indices)
    for record in eligible:
        if not math.isfinite(float(record.target)):
            raise MetricInputError("eligible targets must be finite")
        if not math.isfinite(float(record.prediction)):
            raise MetricInputError("eligible predictions must be finite")
    weights = tuple(weight_result.weights[index] for index in weight_result.eligible_indices)
    return eligible, weights, weight_result


def _base_metadata(task: str, **values):
    result = dict(values)
    result.update(
        {
            "task": task,
            "metric_implementation_version": METRIC_IMPLEMENTATION_VERSION,
            "weight_definition": "eligible_snapshot_weight_equals_1_over_task_specific_stay_count",
            "bootstrap_applied": False,
            "model_selection_applied": False,
        }
    )
    return result


def _metadata_values(metadata: Optional[Mapping[str, object]], **frozen):
    result = dict(metadata or {})
    result.update(frozen)
    return result


def _regression_metrics(records: Sequence[PredictionRecord]):
    eligible, weights, weight_result = _eligible_view(records)
    errors = tuple(record.prediction - record.target for record in eligible)
    absolute_errors = tuple(abs(error) for error in errors)
    return (
        {
            "mae": weighted_mean(absolute_errors, weights),
            "rmse": weighted_rmse(errors, weights),
        },
        eligible,
        weights,
        weight_result,
        absolute_errors,
    )


def _directional_agreement(eligible, weights) -> float:
    if any(record.target == 0 or record.prediction == 0 for record in eligible):
        raise DirectionalAgreementSemanticsError(DIRECTIONAL_ZERO_BLOCKER)
    agreements = tuple(
        1.0 if (record.target > 0) == (record.prediction > 0) else 0.0
        for record in eligible
    )
    return weighted_mean(agreements, weights)


def evaluate_recovery(
    recovery24: Sequence[PredictionRecord],
    recovery48: Sequence[PredictionRecord],
    *,
    metadata: Optional[Mapping[str, object]] = None,
) -> Mapping[str, EvaluationResult]:
    """Evaluate raw SOFA deltas separately for the independent 24h/48h horizons."""

    return {
        "24h": evaluate_recovery_horizon(recovery24, horizon="24h", metadata=metadata),
        "48h": evaluate_recovery_horizon(recovery48, horizon="48h", metadata=metadata),
    }


def evaluate_recovery_horizon(
    records: Sequence[PredictionRecord],
    *,
    horizon: str,
    metadata: Optional[Mapping[str, object]] = None,
) -> EvaluationResult:
    """Reusable single-horizon evaluator for direct bootstrap estimator reuse."""

    if horizon not in ("24h", "48h"):
        raise MetricInputError("recovery horizon must be 24h or 48h")
    metrics, eligible, weights, weight_result, _ = _regression_metrics(records)
    blockers = []
    try:
        metrics["directional_agreement"] = _directional_agreement(eligible, weights)
    except DirectionalAgreementSemanticsError:
        metrics["directional_agreement"] = None
        blockers.append(DIRECTIONAL_ZERO_BLOCKER)
    return EvaluationResult(
        metrics=metrics,
        counts={
            "N_examples": weight_result.n_examples,
            "N_ICU_stays": weight_result.n_icu_stays,
        },
        metadata=_base_metadata(
            "recovery",
            **_metadata_values(
                metadata,
                horizon=horizon,
                evaluation_domain="raw_sofa_delta_unclipped",
                blockers=tuple(blockers),
            ),
        ),
    )


def evaluate_icu_time(
    records: Sequence[PredictionRecord],
    *,
    percentile_levels: Sequence[float] = (),
    metadata: Optional[Mapping[str, object]] = None,
) -> EvaluationResult:
    """Evaluate true and predicted remaining ICU time supplied in hours."""

    metrics, _, weights, weight_result, absolute_errors = _regression_metrics(records)
    metrics["median_absolute_error"] = weighted_median(absolute_errors, weights)
    for level in percentile_levels:
        metrics["absolute_error_percentile_{:g}".format(100.0 * level)] = weighted_percentile(
            absolute_errors, weights, level
        )
    return EvaluationResult(
        metrics=metrics,
        counts={
            "N_examples": weight_result.n_examples,
            "N_ICU_stays": weight_result.n_icu_stays,
        },
        metadata=_base_metadata(
            "icu_stay_time",
            **_metadata_values(
                metadata,
                evaluation_domain="remaining_current_icu_hours",
                weighted_quantile_convention=WEIGHTED_QUANTILE_CONVENTION,
                percentile_levels=tuple(percentile_levels),
                percentile_level_status="UNLOCKED_REPORTING_PARAMETER",
            ),
        ),
    )


def _validate_binary_labels_and_probabilities(eligible) -> None:
    for record in eligible:
        if record.target not in (0, 1, 0.0, 1.0):
            raise MetricInputError("eligible classification labels must be binary")
        if not 0.0 <= record.prediction <= 1.0:
            raise MetricInputError("eligible probabilities must be within [0,1]")


def _weighted_average_precision(labels, probabilities, weights) -> float:
    positive_weight = math.fsum(weight for label, weight in zip(labels, weights) if label == 1)
    if positive_weight <= 0:
        return math.nan
    groups = {}
    for label, probability, weight in zip(labels, probabilities, weights):
        positive, total = groups.get(probability, (0.0, 0.0))
        groups[probability] = (positive + weight * label, total + weight)
    cumulative_positive = 0.0
    cumulative_total = 0.0
    average_precision = 0.0
    previous_recall = 0.0
    for probability in sorted(groups, reverse=True):
        group_positive, group_total = groups[probability]
        cumulative_positive += group_positive
        cumulative_total += group_total
        recall = cumulative_positive / positive_weight
        precision = cumulative_positive / cumulative_total
        average_precision += (recall - previous_recall) * precision
        previous_recall = recall
    return average_precision


def _weighted_auroc(labels, probabilities, weights) -> float:
    positives = [
        (probability, weight)
        for label, probability, weight in zip(labels, probabilities, weights)
        if label == 1
    ]
    negatives = [
        (probability, weight)
        for label, probability, weight in zip(labels, probabilities, weights)
        if label == 0
    ]
    positive_weight = math.fsum(weight for _, weight in positives)
    negative_weight = math.fsum(weight for _, weight in negatives)
    if positive_weight <= 0 or negative_weight <= 0:
        return math.nan
    concordance = 0.0
    for positive_probability, positive_row_weight in positives:
        for negative_probability, negative_row_weight in negatives:
            pair_weight = positive_row_weight * negative_row_weight
            if positive_probability > negative_probability:
                concordance += pair_weight
            elif positive_probability == negative_probability:
                concordance += 0.5 * pair_weight
    return concordance / (positive_weight * negative_weight)


def weighted_confusion_metrics(labels, probabilities, weights, threshold: float):
    if not (len(labels) == len(probabilities) == len(weights)) or len(labels) == 0:
        raise MetricInputError("labels, probabilities, and weights require equal nonzero length")
    converted = []
    for label, probability, weight in zip(labels, probabilities, weights):
        if label not in (0, 1, 0.0, 1.0):
            raise MetricInputError("confusion-matrix labels must be binary")
        probability_value = float(probability)
        weight_value = float(weight)
        if not math.isfinite(probability_value) or not 0.0 <= probability_value <= 1.0:
            raise MetricInputError("confusion-matrix probabilities must be within [0,1]")
        if not math.isfinite(weight_value) or weight_value < 0:
            raise MetricInputError("confusion-matrix weights must be finite and nonnegative")
        converted.append((int(label), probability_value, weight_value))
    if math.fsum(weight for _, _, weight in converted) <= 0:
        raise MetricInputError("confusion-matrix total weight must be positive")
    threshold_value = float(threshold)
    if not math.isfinite(threshold_value) or not 0.0 <= threshold_value <= 1.0:
        raise MetricInputError("fixed threshold must be within [0,1]")
    tp = tn = fp = fn = 0.0
    for label, probability, weight in converted:
        predicted_positive = probability >= threshold_value
        if label == 1 and predicted_positive:
            tp += weight
        elif label == 1:
            fn += weight
        elif predicted_positive:
            fp += weight
        else:
            tn += weight

    def ratio(numerator, denominator):
        return numerator / denominator if denominator > 0 else math.nan

    sensitivity = ratio(tp, tp + fn)
    specificity = ratio(tn, tn + fp)
    precision = ratio(tp, tp + fp)
    f1 = (
        2.0 * precision * sensitivity / (precision + sensitivity)
        if math.isfinite(precision)
        and math.isfinite(sensitivity)
        and precision + sensitivity > 0
        else math.nan
    )
    return {
        "weighted_TP": tp,
        "weighted_TN": tn,
        "weighted_FP": fp,
        "weighted_FN": fn,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "f1": f1,
    }


def evaluate_organ_support(
    records: Sequence[PredictionRecord],
    *,
    fixed_threshold: Optional[float] = None,
    threshold_identifier: Optional[str] = None,
    probability_type: str = "raw",
    metadata: Optional[Mapping[str, object]] = None,
) -> EvaluationResult:
    if probability_type not in ("raw", "calibrated"):
        raise MetricInputError("probability_type must be raw or calibrated")
    eligible, weights, weight_result = _eligible_view(records)
    _validate_binary_labels_and_probabilities(eligible)
    labels = tuple(int(record.target) for record in eligible)
    probabilities = tuple(record.prediction for record in eligible)
    metrics = {
        "auprc": _weighted_average_precision(labels, probabilities, weights),
        "auroc": _weighted_auroc(labels, probabilities, weights),
        "brier": weighted_mean(
            tuple((probability - label) ** 2 for label, probability in zip(labels, probabilities)),
            weights,
        ),
    }
    if fixed_threshold is not None:
        if not threshold_identifier:
            raise MetricInputError("supplied threshold requires a threshold_identifier")
        metrics.update(weighted_confusion_metrics(labels, probabilities, weights, fixed_threshold))
    positive_examples = sum(labels)
    positive_stays = len(
        {record.stay_id for record, label in zip(eligible, labels) if label == 1}
    )
    undefined = tuple(name for name, value in metrics.items() if math.isnan(value))
    return EvaluationResult(
        metrics=metrics,
        counts={
            "N_examples": weight_result.n_examples,
            "N_ICU_stays": weight_result.n_icu_stays,
            "N_positive_examples": positive_examples,
            "N_positive_stays": positive_stays,
        },
        metadata=_base_metadata(
            "organ_support",
            **_metadata_values(
                metadata,
                probability_type=probability_type,
                auprc_convention=AUPRC_CONVENTION,
                auprc_review_blocker=AUPRC_REVIEW_BLOCKER,
                auroc_convention="weighted_pairwise_concordance_half_credit_for_ties_v1",
                class_weight_used_for_evaluation=False,
                fixed_threshold=fixed_threshold,
                threshold_identifier=threshold_identifier,
                threshold_rule="positive_if_probability_greater_than_or_equal_threshold_v1",
                threshold_selected=False,
                calibration_fitted=False,
                undefined_metrics=undefined,
            ),
        ),
    )
