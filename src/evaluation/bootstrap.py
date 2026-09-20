"""Stay-cluster percentile bootstrap that reuses Phase-9 point estimators."""

import math
import random
from dataclasses import dataclass
from typing import Callable, Hashable, Mapping, Optional, Sequence, Tuple

from evaluation.metrics import (
    METRIC_IMPLEMENTATION_VERSION,
    PredictionRecord,
    evaluate_icu_time,
    evaluate_organ_support,
    evaluate_recovery_horizon,
)


BOOTSTRAP_IMPLEMENTATION_VERSION = "stay_cluster_percentile_bootstrap_v1"
BOOTSTRAP_METHOD = "stay_cluster_percentile_bootstrap"
CI_LEVEL = 0.95
CI_QUANTILES = (0.025, 0.975)
PERCENTILE_CONVENTION = "linear_type7_index_equals_q_times_n_minus_one_v1"
AGGREGATE_UNCERTAINTY_ONLY = (
    "aggregate_performance_uncertainty_not_individual_patient_prediction_interval"
)


class BootstrapError(ValueError):
    """Raised when a grouped-bootstrap request violates its contract."""


MetricCallback = Callable[[Sequence[PredictionRecord]], Optional[float]]


@dataclass(frozen=True)
class BootstrapCopiedRow:
    original_stay_id: Hashable
    bootstrap_cluster_id: Tuple[str, int, int]
    draw_index: int
    source_record: PredictionRecord

    def as_prediction_record(self) -> PredictionRecord:
        return PredictionRecord(
            stay_id=self.bootstrap_cluster_id,
            prediction_time=self.source_record.prediction_time,
            target=self.source_record.target,
            prediction=self.source_record.prediction,
            eligible=True,
        )


@dataclass(frozen=True)
class BootstrapReplicate:
    replicate_index: int
    sampled_original_stays: Tuple[Hashable, ...]
    copied_rows: Tuple[BootstrapCopiedRow, ...]

    @property
    def prediction_records(self) -> Tuple[PredictionRecord, ...]:
        return tuple(row.as_prediction_record() for row in self.copied_rows)

    @property
    def bootstrap_cluster_ids(self) -> Tuple[Tuple[str, int, int], ...]:
        return tuple(dict.fromkeys(row.bootstrap_cluster_id for row in self.copied_rows))


@dataclass(frozen=True)
class BootstrapResult:
    metric_name: str
    task: str
    horizon: Optional[str]
    point_estimate: Optional[float]
    ci_lower: Optional[float]
    ci_upper: Optional[float]
    ci_level: float
    method: str
    n_requested_replicates: int
    n_valid_replicates: int
    n_invalid_replicates: int
    n_original_stays: int
    n_original_examples: int
    bootstrap_seed: int
    metric_implementation_version: str
    bootstrap_implementation_version: str
    bootstrap_distribution: Tuple[float, ...]
    sampled_cluster_sequences: Tuple[Tuple[Hashable, ...], ...]
    metadata: Mapping[str, object]


def percentile(values: Sequence[float], q: float) -> float:
    """Unweighted Type-7 linear percentile for the bootstrap distribution."""

    if not values:
        raise BootstrapError("percentile requires at least one valid replicate")
    quantile = float(q)
    if not math.isfinite(quantile) or not 0.0 <= quantile <= 1.0:
        raise BootstrapError("percentile q must be within [0,1]")
    ordered = sorted(float(value) for value in values)
    if not all(math.isfinite(value) for value in ordered):
        raise BootstrapError("percentile values must be finite")
    position = quantile * (len(ordered) - 1)
    lower_index = int(math.floor(position))
    upper_index = int(math.ceil(position))
    if lower_index == upper_index:
        return ordered[lower_index]
    fraction = position - lower_index
    return ordered[lower_index] + fraction * (
        ordered[upper_index] - ordered[lower_index]
    )


def _eligible_index(records: Sequence[PredictionRecord]):
    by_stay = {}
    ordered_stays = []
    eligible_count = 0
    for record in records:
        if not isinstance(record, PredictionRecord):
            raise BootstrapError("bootstrap inputs must be PredictionRecord instances")
        if not record.eligible:
            continue
        if record.stay_id is None or (
            isinstance(record.stay_id, str) and not record.stay_id.strip()
        ):
            raise BootstrapError("eligible bootstrap rows require a valid stay_id")
        try:
            is_new_stay = record.stay_id not in by_stay
        except TypeError as error:
            raise BootstrapError("eligible bootstrap stay_id must be hashable") from error
        if is_new_stay:
            by_stay[record.stay_id] = []
            ordered_stays.append(record.stay_id)
        by_stay[record.stay_id].append(record)
        eligible_count += 1
    if not ordered_stays:
        raise BootstrapError("grouped bootstrap requires at least one eligible stay")
    return by_stay, tuple(ordered_stays), eligible_count


def materialize_stay_copy_replicate(
    records: Sequence[PredictionRecord],
    sampled_stays: Sequence[Hashable],
    *,
    replicate_index: int,
) -> BootstrapReplicate:
    """Carry all eligible rows and assign each sampled occurrence a unique ID."""

    by_stay, _, _ = _eligible_index(records)
    return _materialize_from_index(
        by_stay, sampled_stays, replicate_index=replicate_index
    )


def _materialize_from_index(by_stay, sampled_stays, *, replicate_index):
    copied_rows = []
    for draw_index, original_stay_id in enumerate(sampled_stays):
        if original_stay_id not in by_stay:
            raise BootstrapError("sampled stay has no eligible source rows")
        cluster_id = ("bootstrap_cluster_v1", replicate_index, draw_index)
        copied_rows.extend(
            BootstrapCopiedRow(
                original_stay_id=original_stay_id,
                bootstrap_cluster_id=cluster_id,
                draw_index=draw_index,
                source_record=record,
            )
            for record in by_stay[original_stay_id]
        )
    return BootstrapReplicate(
        replicate_index=replicate_index,
        sampled_original_stays=tuple(sampled_stays),
        copied_rows=tuple(copied_rows),
    )


def grouped_bootstrap(
    records: Sequence[PredictionRecord],
    *,
    metric_fn: MetricCallback,
    metric_name: str,
    task: str,
    n_bootstrap: int,
    seed: int,
    horizon: Optional[str] = None,
    forced_draws: Optional[Sequence[Sequence[Hashable]]] = None,
    metadata: Optional[Mapping[str, object]] = None,
) -> BootstrapResult:
    """Resample stay copies and call exactly the supplied Phase-9 estimator."""

    if isinstance(n_bootstrap, bool) or not isinstance(n_bootstrap, int) or n_bootstrap <= 0:
        raise BootstrapError("n_bootstrap must be a positive explicit integer")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise BootstrapError("bootstrap seed must be an explicit integer")
    if not callable(metric_fn):
        raise BootstrapError("metric_fn must be callable")
    by_stay, original_stays, eligible_count = _eligible_index(records)
    original_point = metric_fn(records)
    if original_point is not None:
        original_point = float(original_point)
    if forced_draws is not None and len(forced_draws) != n_bootstrap:
        raise BootstrapError("forced_draws must contain exactly n_bootstrap sequences")

    generator = random.Random(seed)
    valid_values = []
    sampled_sequences = []
    invalid_count = 0
    for replicate_index in range(n_bootstrap):
        if forced_draws is None:
            draws = tuple(
                original_stays[generator.randrange(len(original_stays))]
                for _ in range(len(original_stays))
            )
        else:
            draws = tuple(forced_draws[replicate_index])
            if len(draws) != len(original_stays):
                raise BootstrapError(
                    "each replicate must draw exactly the original eligible stay count"
                )
        replicate = _materialize_from_index(
            by_stay, draws, replicate_index=replicate_index
        )
        sampled_sequences.append(replicate.sampled_original_stays)
        value = metric_fn(replicate.prediction_records)
        if value is None or not math.isfinite(float(value)):
            invalid_count += 1
            continue
        valid_values.append(float(value))

    if valid_values:
        ci_lower = percentile(valid_values, CI_QUANTILES[0])
        ci_upper = percentile(valid_values, CI_QUANTILES[1])
    else:
        ci_lower = None
        ci_upper = None
    result_metadata = dict(metadata or {})
    result_metadata.update(
        {
            "ci_quantiles": CI_QUANTILES,
            "percentile_convention": PERCENTILE_CONVENTION,
            "resampling_unit": "eligible_icu_stay_copy",
            "sampling_with_replacement": True,
            "rows_sampled_independently": False,
            "minimum_valid_replicates": "UNLOCKED_REPORTING_QA_PARAMETER",
            "uncertainty_interpretation": AGGREGATE_UNCERTAINTY_ONLY,
            "test_data_accessed": False,
        }
    )
    return BootstrapResult(
        metric_name=metric_name,
        task=task,
        horizon=horizon,
        point_estimate=original_point,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        ci_level=CI_LEVEL,
        method=BOOTSTRAP_METHOD,
        n_requested_replicates=n_bootstrap,
        n_valid_replicates=len(valid_values),
        n_invalid_replicates=invalid_count,
        n_original_stays=len(original_stays),
        n_original_examples=eligible_count,
        bootstrap_seed=seed,
        metric_implementation_version=METRIC_IMPLEMENTATION_VERSION,
        bootstrap_implementation_version=BOOTSTRAP_IMPLEMENTATION_VERSION,
        bootstrap_distribution=tuple(valid_values),
        sampled_cluster_sequences=tuple(sampled_sequences),
        metadata=result_metadata,
    )


def bootstrap_recovery(
    recovery24: Sequence[PredictionRecord],
    recovery48: Sequence[PredictionRecord],
    *,
    n_bootstrap: int,
    seed: int,
    forced_draws_24: Optional[Sequence[Sequence[Hashable]]] = None,
    forced_draws_48: Optional[Sequence[Sequence[Hashable]]] = None,
    metadata: Optional[Mapping[str, object]] = None,
):
    output = {}
    for horizon, records, forced in (
        ("24h", recovery24, forced_draws_24),
        ("48h", recovery48, forced_draws_48),
    ):
        horizon_results = {}
        # Directional agreement remains blocked until zero/tie semantics are frozen.
        for metric_name in ("mae", "rmse"):
            def metric_fn(rows, name=metric_name, selected_horizon=horizon):
                return evaluate_recovery_horizon(
                    rows, horizon=selected_horizon
                ).metrics[name]

            horizon_results[metric_name] = grouped_bootstrap(
                records,
                metric_fn=metric_fn,
                metric_name=metric_name,
                task="recovery",
                horizon=horizon,
                n_bootstrap=n_bootstrap,
                seed=seed,
                forced_draws=forced,
                metadata=metadata,
            )
        output[horizon] = horizon_results
    return output


def bootstrap_icu_time(
    records: Sequence[PredictionRecord],
    *,
    n_bootstrap: int,
    seed: int,
    percentile_levels: Sequence[float] = (),
    forced_draws: Optional[Sequence[Sequence[Hashable]]] = None,
    metadata: Optional[Mapping[str, object]] = None,
):
    metric_names = ["median_absolute_error", "mae", "rmse"]
    metric_names.extend(
        "absolute_error_percentile_{:g}".format(100.0 * level)
        for level in percentile_levels
    )
    output = {}
    for metric_name in metric_names:
        def metric_fn(rows, name=metric_name):
            return evaluate_icu_time(
                rows, percentile_levels=percentile_levels
            ).metrics[name]

        output[metric_name] = grouped_bootstrap(
            records,
            metric_fn=metric_fn,
            metric_name=metric_name,
            task="icu_stay_time",
            n_bootstrap=n_bootstrap,
            seed=seed,
            forced_draws=forced_draws,
            metadata=metadata,
        )
    return output


def bootstrap_organ_support(
    records: Sequence[PredictionRecord],
    *,
    n_bootstrap: int,
    seed: int,
    fixed_threshold: Optional[float] = None,
    threshold_identifier: Optional[str] = None,
    probability_type: str = "raw",
    forced_draws: Optional[Sequence[Sequence[Hashable]]] = None,
    metadata: Optional[Mapping[str, object]] = None,
):
    metric_names = ["auprc", "auroc", "brier"]
    if fixed_threshold is not None:
        metric_names.extend(("sensitivity", "specificity", "precision", "f1"))
    output = {}
    for metric_name in metric_names:
        def metric_fn(rows, name=metric_name):
            return evaluate_organ_support(
                rows,
                fixed_threshold=fixed_threshold,
                threshold_identifier=threshold_identifier,
                probability_type=probability_type,
            ).metrics[name]

        output[metric_name] = grouped_bootstrap(
            records,
            metric_fn=metric_fn,
            metric_name=metric_name,
            task="organ_support",
            n_bootstrap=n_bootstrap,
            seed=seed,
            forced_draws=forced_draws,
            metadata={
                **dict(metadata or {}),
                "probability_type": probability_type,
                "fixed_threshold": fixed_threshold,
                "threshold_identifier": threshold_identifier,
                "calibrator_refit": False,
                "threshold_reselected": False,
            },
        )
    return output
