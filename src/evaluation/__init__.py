"""Model-neutral stay-balanced point-estimate evaluation."""

from evaluation.metrics import (
    METRIC_IMPLEMENTATION_VERSION,
    DirectionalAgreementSemanticsError,
    EmptyEligibleSetError,
    EvaluationResult,
    PredictionRecord,
    evaluate_icu_time,
    evaluate_organ_support,
    evaluate_recovery,
    evaluate_recovery_horizon,
    weighted_confusion_metrics,
)
from evaluation.weighted_stats import (
    weighted_mean,
    weighted_median,
    weighted_percentile,
    weighted_rmse,
)
from evaluation.weights import StayWeightResult, compute_stay_weights
from evaluation.bootstrap import (
    BOOTSTRAP_IMPLEMENTATION_VERSION,
    BootstrapResult,
    bootstrap_icu_time,
    bootstrap_organ_support,
    bootstrap_recovery,
    grouped_bootstrap,
)

__all__ = [
    "METRIC_IMPLEMENTATION_VERSION",
    "DirectionalAgreementSemanticsError",
    "EmptyEligibleSetError",
    "EvaluationResult",
    "PredictionRecord",
    "StayWeightResult",
    "compute_stay_weights",
    "weighted_mean",
    "weighted_rmse",
    "weighted_median",
    "weighted_percentile",
    "weighted_confusion_metrics",
    "evaluate_recovery",
    "evaluate_recovery_horizon",
    "evaluate_icu_time",
    "evaluate_organ_support",
    "BOOTSTRAP_IMPLEMENTATION_VERSION",
    "BootstrapResult",
    "grouped_bootstrap",
    "bootstrap_recovery",
    "bootstrap_icu_time",
    "bootstrap_organ_support",
]
"""Evaluation, uncertainty, and validation-only model selection."""

from evaluation.select import (
    SelectionResult,
    build_selection_stage_manifest,
    select_task,
    selection_report,
    validate_selection_stage_manifest,
)
from evaluation.calibrate import apply_support_calibrator, fit_support_calibrator
from evaluation.threshold import apply_support_threshold, choose_support_threshold
from evaluation.ablations import derive_feature_manifest, evaluate_recovery_ablation
from evaluation.sensitivity import evaluate_complete_component_sensitivity
from evaluation.error_analysis import evaluate_error_analysis, write_error_analysis_table
from evaluation.slices import assign_slice_group, validate_slice_specification

__all__ = [
    "SelectionResult",
    "build_selection_stage_manifest",
    "select_task",
    "selection_report",
    "validate_selection_stage_manifest",
    "apply_support_calibrator",
    "fit_support_calibrator",
    "apply_support_threshold",
    "choose_support_threshold",
    "derive_feature_manifest",
    "evaluate_recovery_ablation",
    "evaluate_complete_component_sensitivity",
    "assign_slice_group",
    "validate_slice_specification",
    "evaluate_error_analysis",
    "write_error_analysis_table",
]
