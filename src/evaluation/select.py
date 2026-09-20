"""Authoritative validation-only Phase-12 per-task model selection."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional

from experiments.search_governance import canonical_sha256

from evaluation.selection_validation import (
    EXACT_TIE_UNLOCKED,
    EXPLANATION_METHOD,
    SELECTION_IMPLEMENTATION_VERSION,
    TIE_BLOCKER,
    TOLERANCE_UNLOCKED,
    CandidateSummary,
    SelectionValidationError,
    primary_metric,
    sha256_file,
    validate_pair,
)


@dataclass(frozen=True)
class SelectionResult:
    task: str
    primary_metric: str
    direction: str
    xgboost: CandidateSummary
    gru: CandidateSummary
    selected: CandidateSummary
    explanation_method: str
    selection_mode: str
    test_accessed: bool = False


def select_task(
    *,
    task: str,
    xgboost: CandidateSummary,
    gru: CandidateSummary,
    selection_mode: str = "synthetic",
    tie_tolerance: Optional[float] = None,
) -> SelectionResult:
    """Select strictly by the frozen primary metric after all parity gates."""

    if tie_tolerance is not None:
        raise SelectionValidationError(TOLERANCE_UNLOCKED)
    validate_pair(xgboost, gru, task=task, selection_mode=selection_mode)
    key, direction, xgb_value = primary_metric(task, xgboost)
    _, _, gru_value = primary_metric(task, gru)
    if xgb_value == gru_value:
        raise SelectionValidationError(
            TIE_BLOCKER + "; " + EXACT_TIE_UNLOCKED + "; " + TOLERANCE_UNLOCKED
        )
    if direction == "min":
        selected = xgboost if xgb_value < gru_value else gru
    else:
        selected = xgboost if xgb_value > gru_value else gru
    return SelectionResult(
        task=task,
        primary_metric=key,
        direction=direction,
        xgboost=xgboost,
        gru=gru,
        selected=selected,
        explanation_method=EXPLANATION_METHOD[selected.family],
        selection_mode=selection_mode,
    )


def selection_report(result: SelectionResult) -> Mapping[str, object]:
    """Create deterministic evidence without test fields or mutable timestamps."""

    return {
        "selection_implementation_version": SELECTION_IMPLEMENTATION_VERSION,
        "selection_mode": result.selection_mode,
        "task": result.task,
        "primary_metric": result.primary_metric,
        "direction": result.direction,
        "candidates": {
            "xgboost": _candidate_evidence(result.xgboost),
            "gru": _candidate_evidence(result.gru),
        },
        "selected_family": result.selected.family,
        "selected_run_id": result.selected.run_id,
        "selected_candidate_id": result.selected.candidate_id,
        "selected_artifact_ref": result.selected.artifact_ref,
        "selected_artifact_sha256": result.selected.artifact_sha256,
        "explanation_method": result.explanation_method,
        "parity_checks": {
            "information": "PASS",
            "validation_population": "PASS",
            "metric_implementation": "PASS",
            "artifact_compatibility": "PASS",
        },
        "test_accessed": False,
    }


def _candidate_evidence(candidate: CandidateSummary) -> Mapping[str, object]:
    return {
        "candidate_id": candidate.candidate_id,
        "run_id": candidate.run_id,
        "config_hash": candidate.config_hash,
        "search_version": candidate.search.search_version,
        "search_space_hash": candidate.search.search_space_hash,
        "candidate_list_hash": candidate.search.candidate_list_hash,
        "split_hash": candidate.split_hash,
        "feature_version": candidate.feature_version,
        "label_version": candidate.label_version,
        "preprocessing_hash": candidate.preprocessing_hash,
        "metric_implementation_version": candidate.metric_implementation_version,
        "probability_type": candidate.probability_type,
        "primary_and_secondary_validation_metrics": dict(candidate.metrics),
        "validation_population": dict(candidate.counts),
        "artifact_ref": candidate.artifact_ref,
        "artifact_sha256": candidate.artifact_sha256,
    }


def build_selection_stage_manifest(results: Mapping[str, SelectionResult]):
    """Build, but never write, a deterministic selection-stage manifest.

    The caller may persist a real manifest only after genuine scientific search
    evidence exists.  Synthetic output is explicitly non-serving.
    """

    expected = {"recovery", "icu_stay_time", "organ_support"}
    if set(results) != expected:
        raise SelectionValidationError("manifest requires one result for every task")
    modes = {result.selection_mode for result in results.values()}
    if len(modes) != 1:
        raise SelectionValidationError("manifest cannot mix synthetic and real evidence")
    tasks = {}
    for task in ("recovery", "icu_stay_time", "organ_support"):
        result = results[task]
        if result.task != task:
            raise SelectionValidationError("manifest task/result mismatch")
        selected = result.selected
        entry = {
            "family": selected.family,
            "run_id": selected.run_id,
            "candidate_id": selected.candidate_id,
            "config_hash": selected.config_hash,
            "search_version": selected.search.search_version,
            "search_space_hash": selected.search.search_space_hash,
            "candidate_list_hash": selected.search.candidate_list_hash,
            "artifact_ref": selected.artifact_ref,
            "artifact_sha256": selected.artifact_sha256,
            "artifact_metadata_ref": selected.artifact_metadata_ref,
            "feature_version": selected.feature_version,
            "label_version": selected.label_version,
            "split_hash": selected.split_hash,
            "preprocessing_hash": selected.preprocessing_hash,
            "code_commit": selected.code_commit,
            "selection_metric": result.primary_metric,
            "validation_metric_value": selected.metrics[result.primary_metric],
            "explanation_method": EXPLANATION_METHOD[selected.family],
        }
        if task == "organ_support":
            entry.update(
                {
                    "calibration_status": "PENDING_PHASE_13",
                    "calibrator": None,
                    "threshold": None,
                    "validation_prediction_ref": selected.validation_prediction_ref,
                }
            )
        tasks[task] = entry
    mode = modes.pop()
    content = {
        "manifest_version": "selected_models_selection_stage_v1",
        "selection_implementation_version": SELECTION_IMPLEMENTATION_VERSION,
        "selection_mode": mode,
        "status": (
            "SYNTHETIC_SELECTION_FRAMEWORK_EVIDENCE_ONLY"
            if mode == "synthetic"
            else "MODEL_SELECTION_FROZEN_CALIBRATION_PENDING_PHASE_13"
        ),
        "serving_ready": False,
        "test_accessed": False,
        "tasks": tasks,
    }
    return {**content, "manifest_sha256": canonical_sha256(content)}


def validate_selection_stage_manifest(manifest: Mapping[str, object]) -> None:
    if manifest.get("serving_ready") is not False or manifest.get("test_accessed") is not False:
        raise SelectionValidationError("Phase-12 selection-stage manifest is not serving ready")
    if str(manifest.get("status", "")).startswith("G3"):
        raise SelectionValidationError("Phase 12 cannot claim G3 selection/calibration freeze")
    tasks = manifest.get("tasks")
    if not isinstance(tasks, Mapping) or set(tasks) != {
        "recovery",
        "icu_stay_time",
        "organ_support",
    }:
        raise SelectionValidationError("manifest task map is incomplete")
    for task, entry in tasks.items():
        family = entry.get("family")
        if entry.get("explanation_method") != EXPLANATION_METHOD.get(family):
            raise SelectionValidationError("explanation routing does not match selected family")
        artifact = Path(str(entry.get("artifact_ref", "")))
        metadata_path = Path(str(entry.get("artifact_metadata_ref", "")))
        if not artifact.is_file() or not metadata_path.is_file():
            raise SelectionValidationError("manifest model artifact or metadata does not exist")
        if sha256_file(artifact) != entry.get("artifact_sha256"):
            raise SelectionValidationError("manifest model artifact SHA-256 mismatch")
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise SelectionValidationError("manifest model metadata is unreadable") from error
        expected = {
            "task": task,
            "model_family": family,
            "feature_schema_version": entry.get("feature_version"),
            "split_hash": entry.get("split_hash"),
            "preprocessing_hash": entry.get("preprocessing_hash"),
            "label_version": entry.get("label_version"),
        }
        for field, expected_value in expected.items():
            if metadata.get(field) != expected_value:
                raise SelectionValidationError("manifest artifact compatibility mismatch: " + field)
    support = tasks["organ_support"]
    if support.get("calibration_status") != "PENDING_PHASE_13":
        raise SelectionValidationError("support calibration must remain pending Phase 13")
    if support.get("calibrator") is not None or support.get("threshold") is not None:
        raise SelectionValidationError("Phase 12 cannot provide a calibrator or threshold")
    supplied_hash = manifest.get("manifest_sha256")
    content = dict(manifest)
    content.pop("manifest_sha256", None)
    if supplied_hash != canonical_sha256(content):
        raise SelectionValidationError("selection-stage manifest hash mismatch")
