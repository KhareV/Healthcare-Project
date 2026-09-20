"""Frozen-prediction, descriptive error analysis with Phase-9 metric reuse."""

import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Optional, Sequence, Tuple

from evaluation.metrics import (
    EmptyEligibleSetError,
    PredictionRecord,
    evaluate_icu_time,
    evaluate_organ_support,
    evaluate_recovery_horizon,
)
from evaluation.sensitivity import CompleteComponentRecord
from evaluation.slices import (
    SliceDefinition,
    SliceSpecification,
    assign_slice_group,
    task_slice_definitions,
    validate_slice_specification,
)


ERROR_ANALYSIS_IMPLEMENTATION_VERSION = "prespecified_error_analysis_v1"
FINAL_TEST_DEFERRED = "FINAL TEST ERROR ANALYSIS — DEFERRED UNTIL PHASE 20"


class ErrorAnalysisValidationError(ValueError):
    """Raised when prediction, model, split, or slice lineage is incompatible."""


@dataclass(frozen=True)
class SelectedModelBinding:
    task: str
    model_family: str
    model_run_id: str
    model_hash: str
    feature_version: str
    label_version: str
    split_hash: str
    selected: bool
    probability_type: Optional[str] = None
    threshold: Optional[float] = None
    threshold_id: Optional[str] = None
    calibrator_id: Optional[str] = None
    calibrator_hash: Optional[str] = None


@dataclass(frozen=True)
class ErrorAnalysisRecord:
    row_id: str
    stay_id: str
    prediction_time: str
    split: str
    task: str
    horizon: Optional[str]
    target: float
    prediction: float
    eligible: bool
    model_run_id: str
    model_hash: str
    feature_version: str
    label_version: str
    split_hash: str
    slice_metadata: Mapping[str, object]
    complete_component_record: Optional[CompleteComponentRecord] = None
    probability_type: Optional[str] = None
    calibrator_id: Optional[str] = None
    calibrator_hash: Optional[str] = None


@dataclass(frozen=True)
class ErrorAnalysisTableRow:
    split: str
    task: str
    horizon: str
    slice_name: str
    slice_group: str
    slice_spec_version: str
    slice_spec_hash: str
    model_family: str
    model_run_id: str
    model_hash: str
    N_examples: int
    N_ICU_stays: int
    N_positive_examples: Optional[int]
    N_positive_stays: Optional[int]
    metric_name: str
    metric_value: Optional[float]
    probability_type: str
    threshold_id: str
    calibrator_id: str
    calibrator_hash: str
    feature_version: str
    label_version: str
    split_hash: str
    status: str
    implementation_version: str = ERROR_ANALYSIS_IMPLEMENTATION_VERSION


TASK_METRICS = {
    "recovery": ("mae", "rmse", "directional_agreement"),
    "icu_stay_time": ("median_absolute_error", "mae", "rmse"),
    "organ_support": (
        "auprc",
        "auroc",
        "brier",
        "sensitivity",
        "specificity",
        "precision",
        "f1",
    ),
}


def validate_binding(binding: SelectedModelBinding) -> None:
    if binding.task not in TASK_METRICS or binding.model_family not in ("xgboost", "gru"):
        raise ErrorAnalysisValidationError("selected model task/family is invalid")
    if binding.selected is not True:
        raise ErrorAnalysisValidationError("final error analysis requires the globally selected model")
    for value in (
        binding.model_run_id,
        binding.model_hash,
        binding.feature_version,
        binding.label_version,
        binding.split_hash,
    ):
        if not isinstance(value, str) or not value.strip():
            raise ErrorAnalysisValidationError("selected model lineage is incomplete")
    if binding.task == "organ_support":
        if binding.probability_type not in ("raw", "calibrated"):
            raise ErrorAnalysisValidationError("support probability type must be frozen")
        if binding.probability_type == "calibrated" and not (
            binding.calibrator_id and binding.calibrator_hash
        ):
            raise ErrorAnalysisValidationError("calibrated support requires global calibrator lineage")
        if (binding.threshold is None) != (binding.threshold_id is None):
            raise ErrorAnalysisValidationError("support threshold and identifier must be paired")


def _validate_records(
    records: Sequence[ErrorAnalysisRecord], binding: SelectedModelBinding, *, allow_test: bool
) -> str:
    validate_binding(binding)
    if not records:
        raise ErrorAnalysisValidationError("error analysis requires prediction records")
    splits = {record.split for record in records}
    if len(splits) != 1 or not splits <= {"validation", "test"}:
        raise ErrorAnalysisValidationError("validation and test records cannot be mixed")
    split = next(iter(splits))
    if split == "test" and not allow_test:
        raise ErrorAnalysisValidationError(FINAL_TEST_DEFERRED)
    seen = set()
    expected = {
        "task": binding.task,
        "model_run_id": binding.model_run_id,
        "model_hash": binding.model_hash,
        "feature_version": binding.feature_version,
        "label_version": binding.label_version,
        "split_hash": binding.split_hash,
    }
    for record in records:
        key = (record.task, record.horizon, record.stay_id, record.prediction_time)
        if key in seen:
            raise ErrorAnalysisValidationError("duplicate canonical prediction row")
        seen.add(key)
        for field, value in expected.items():
            if getattr(record, field) != value:
                raise ErrorAnalysisValidationError("prediction/model lineage mismatch: " + field)
        if record.task == "recovery" and record.horizon not in ("24h", "48h"):
            raise ErrorAnalysisValidationError("recovery rows require a 24h or 48h horizon")
        if record.task != "recovery" and record.horizon is not None:
            raise ErrorAnalysisValidationError("only recovery records have a horizon")
        if record.task == "organ_support" and (
            record.probability_type != binding.probability_type
            or record.calibrator_id != binding.calibrator_id
            or record.calibrator_hash != binding.calibrator_hash
        ):
            raise ErrorAnalysisValidationError(
                "support probability/calibrator lineage mismatch"
            )
        cc = record.complete_component_record
        if cc is not None and (
            cc.stay_id != record.stay_id
            or cc.prediction_time != record.prediction_time
            or cc.partition != record.split
            or cc.target != record.target
            or cc.prediction != record.prediction
            or cc.eligible != record.eligible
        ):
            raise ErrorAnalysisValidationError("Phase-14 complete-component record mismatch")
    return split


def _prediction_records(records: Sequence[ErrorAnalysisRecord]) -> Tuple[PredictionRecord, ...]:
    return tuple(
        PredictionRecord(
            stay_id=item.stay_id,
            prediction_time=item.prediction_time,
            target=item.target,
            prediction=item.prediction,
            eligible=item.eligible,
        )
        for item in records
    )


def _empty_counts(task: str) -> Mapping[str, int]:
    counts = {"N_examples": 0, "N_ICU_stays": 0}
    if task == "organ_support":
        counts.update({"N_positive_examples": 0, "N_positive_stays": 0})
    return counts


def _evaluate_group(
    records: Sequence[ErrorAnalysisRecord], binding: SelectedModelBinding, horizon: str
):
    converted = _prediction_records(records)
    try:
        if binding.task == "recovery":
            return evaluate_recovery_horizon(converted, horizon=horizon)
        if binding.task == "icu_stay_time":
            return evaluate_icu_time(converted)
        return evaluate_organ_support(
            converted,
            fixed_threshold=binding.threshold,
            threshold_identifier=binding.threshold_id,
            probability_type=binding.probability_type or "raw",
        )
    except EmptyEligibleSetError:
        return None


def evaluate_error_analysis(
    records: Sequence[ErrorAnalysisRecord],
    spec: SliceSpecification,
    binding: SelectedModelBinding,
    *,
    allow_test: bool = False,
) -> Tuple[ErrorAnalysisTableRow, ...]:
    """Evaluate every prespecified group; never trains, calibrates, or selects."""

    validate_slice_specification(spec, require_frozen=True)
    split = _validate_records(records, binding, allow_test=allow_test)
    definitions = task_slice_definitions(spec, binding.task)
    output = []
    horizons = ("24h", "48h") if binding.task == "recovery" else ("",)
    for horizon in horizons:
        horizon_records = tuple(
            item for item in records if binding.task != "recovery" or item.horizon == horizon
        )
        for definition in definitions:
            assignments = tuple(
                (
                    item,
                    assign_slice_group(
                        definition,
                        item.slice_metadata,
                        complete_component_record=item.complete_component_record,
                    ),
                )
                for item in horizon_records
            )
            for group in definition.groups:
                members = tuple(item for item, assigned in assignments if assigned == group.name)
                result = _evaluate_group(members, binding, horizon) if members else None
                counts = result.counts if result else _empty_counts(binding.task)
                metric_names = list(TASK_METRICS[binding.task])
                if result:
                    metric_names = [name for name in metric_names if name in result.metrics]
                for metric_name in metric_names:
                    value = result.metrics.get(metric_name) if result else None
                    status = "COMPLETE"
                    if result is None:
                        status = "EMPTY_ELIGIBLE_GROUP"
                    elif value is None or (isinstance(value, float) and math.isnan(value)):
                        status = "METRIC_UNDEFINED"
                        value = None
                    output.append(
                        ErrorAnalysisTableRow(
                            split=split,
                            task=binding.task,
                            horizon=horizon,
                            slice_name=definition.slice_name,
                            slice_group=group.name,
                            slice_spec_version=spec.spec_version,
                            slice_spec_hash=spec.content_hash,
                            model_family=binding.model_family,
                            model_run_id=binding.model_run_id,
                            model_hash=binding.model_hash,
                            N_examples=counts["N_examples"],
                            N_ICU_stays=counts["N_ICU_stays"],
                            N_positive_examples=counts.get("N_positive_examples"),
                            N_positive_stays=counts.get("N_positive_stays"),
                            metric_name=metric_name,
                            metric_value=value,
                            probability_type=binding.probability_type or "",
                            threshold_id=binding.threshold_id or "",
                            calibrator_id=binding.calibrator_id or "",
                            calibrator_hash=binding.calibrator_hash or "",
                            feature_version=binding.feature_version,
                            label_version=binding.label_version,
                            split_hash=binding.split_hash,
                            status=status,
                        )
                    )
            excluded = sum(1 for item, assigned in assignments if assigned is None and item.eligible)
            if excluded:
                output.append(
                    ErrorAnalysisTableRow(
                        split=split,
                        task=binding.task,
                        horizon=horizon,
                        slice_name=definition.slice_name,
                        slice_group="__EXCLUDED_MISSING__",
                        slice_spec_version=spec.spec_version,
                        slice_spec_hash=spec.content_hash,
                        model_family=binding.model_family,
                        model_run_id=binding.model_run_id,
                        model_hash=binding.model_hash,
                        N_examples=excluded,
                        N_ICU_stays=len(
                            {item.stay_id for item, assigned in assignments if assigned is None and item.eligible}
                        ),
                        N_positive_examples=None,
                        N_positive_stays=None,
                        metric_name="excluded_missing_count",
                        metric_value=float(excluded),
                        probability_type=binding.probability_type or "",
                        threshold_id=binding.threshold_id or "",
                        calibrator_id=binding.calibrator_id or "",
                        calibrator_hash=binding.calibrator_hash or "",
                        feature_version=binding.feature_version,
                        label_version=binding.label_version,
                        split_hash=binding.split_hash,
                        status="EXCLUDED_BY_FROZEN_MISSING_VALUE_POLICY",
                    )
                )
    return tuple(output)


def write_error_analysis_table(rows: Sequence[ErrorAnalysisTableRow], path: Path) -> Path:
    if not rows:
        raise ErrorAnalysisValidationError("cannot write an empty error-analysis table")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = tuple(asdict(rows[0]))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)
    return path


def load_prediction_artifact(manifest: Mapping[str, object]) -> Mapping[str, object]:
    """Fail before opening a path when a pre-Phase-20 manifest names test data."""

    if manifest.get("partition") == "test" or manifest.get("test_accessed") is True:
        raise ErrorAnalysisValidationError(FINAL_TEST_DEFERRED)
    path = Path(str(manifest.get("prediction_ref", "")))
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ErrorAnalysisValidationError("validation prediction artifact is unreadable") from error
    if payload.get("partition") != "validation":
        raise ErrorAnalysisValidationError("only validation predictions may be loaded in Phase 15")
    return payload
