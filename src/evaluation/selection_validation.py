"""Fail-closed validation for Phase-12 XGBoost-versus-GRU selection.

This module validates frozen evidence.  It does not train models, inspect test
artifacts, calibrate probabilities, or choose an operating threshold.
"""

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Sequence, Tuple

from evaluation.metrics import METRIC_IMPLEMENTATION_VERSION
from experiments.search_governance import (
    SEARCH_BUDGET,
    CandidateResult,
    GovernanceError,
    InformationManifest,
    canonical_sha256,
    choose_best_within_family,
    validate_information_parity,
)


SELECTION_IMPLEMENTATION_VERSION = "per_task_xgb_gru_selection_v1"
SERVING_FAMILIES = ("xgboost", "gru")
TASKS = ("recovery", "icu_stay_time", "organ_support")
EXPLANATION_METHOD = {
    "xgboost": "tree_shap",
    "gru": "integrated_gradients",
}
REAL_SELECTION_BLOCKER = "BLOCKED — REAL VALIDATION SEARCH RESULTS REQUIRED"
SEARCH_COMPLETION_BLOCKER = "BLOCKED — COMPLETED SCIENTIFIC CANDIDATE SEARCHES REQUIRED"
EXACT_TIE_UNLOCKED = "UNLOCKED PARAMETER — CROSS-FAMILY EXACT-TIE POLICY"
TOLERANCE_UNLOCKED = "UNLOCKED ENGINEERING PARAMETER — METRIC TIE TOLERANCE"
TIE_BLOCKER = "BLOCKED — TIE POLICY REQUIRED"
INFORMATION_MISMATCH = "SCIENTIFIC COMPARISON INVALID — XGB/GRU INFORMATION MISMATCH"


class SelectionValidationError(ValueError):
    """Raised when selection evidence is incomplete, contaminated, or incompatible."""


@dataclass(frozen=True)
class SearchEvidence:
    """Frozen exact-budget results and their declared candidate membership."""

    task: str
    family: str
    run_type: str
    search_version: str
    search_space_hash: str
    candidate_list_hash: str
    declared_candidates: Tuple[Mapping[str, object], ...]
    results: Tuple[CandidateResult, ...]
    referenced_best_candidate_id: str


@dataclass(frozen=True)
class CandidateSummary:
    """Selection input for one independently verified within-family winner."""

    task: str
    family: str
    candidate_id: str
    run_id: str
    status: str
    config_hash: str
    split_hash: str
    feature_version: str
    label_version: str
    preprocessing_hash: str
    code_commit: str
    artifact_ref: str
    artifact_sha256: str
    artifact_metadata_ref: str
    metrics: Mapping[str, float]
    counts: Mapping[str, int]
    metric_implementation_version: str
    source_partition: str
    test_accessed: bool
    information_manifest: InformationManifest
    search: SearchEvidence
    probability_type: Optional[str] = None
    validation_prediction_ref: Optional[str] = None


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def reject_test_fields(value, *, path="selection_input") -> None:
    """Reject test-labelled input structurally without dereferencing its values."""

    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).lower()
            if normalized != "test_accessed" and (
                normalized == "test"
                or normalized.startswith("test_")
                or normalized.endswith("_test")
            ):
                raise SelectionValidationError(
                    "HARD FAIL — TEST-LABELLED FIELD FORBIDDEN: " + path + "." + str(key)
                )
            reject_test_fields(child, path=path + "." + str(key))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            reject_test_fields(child, path="{}[{}]".format(path, index))


def _primary_spec(task: str):
    if task == "recovery":
        return "validation_recovery24_mae", "min"
    if task == "icu_stay_time":
        return "validation_median_absolute_error_hours", "min"
    if task == "organ_support":
        return "validation_auprc", "max"
    raise SelectionValidationError("unsupported selection task")


def _validate_search_evidence(candidate: CandidateSummary, *, selection_mode: str) -> None:
    search = candidate.search
    if search.task != candidate.task or search.family != candidate.family:
        raise SelectionValidationError("search task/family does not match candidate summary")
    if not search.search_version or not search.search_space_hash or not search.candidate_list_hash:
        raise SelectionValidationError("search lineage hashes/version are required")
    if len(search.declared_candidates) != SEARCH_BUDGET:
        raise SelectionValidationError(SEARCH_COMPLETION_BLOCKER)
    if len(search.results) != SEARCH_BUDGET:
        raise SelectionValidationError(SEARCH_COMPLETION_BLOCKER)
    declared_by_id = {}
    for declared in search.declared_candidates:
        candidate_id = declared.get("candidate_id")
        config_hash = declared.get("config_hash")
        if not isinstance(candidate_id, str) or not isinstance(config_hash, str):
            raise SelectionValidationError("declared candidate requires ID and config hash")
        if candidate_id in declared_by_id:
            raise SelectionValidationError("declared candidate IDs must be unique")
        if "config" in declared and canonical_sha256(declared["config"]) != config_hash:
            raise SelectionValidationError("declared candidate config hash mismatch")
        declared_by_id[candidate_id] = config_hash
    declared_ids = set(declared_by_id)
    if len(set(declared_by_id.values())) != SEARCH_BUDGET:
        raise SelectionValidationError("declared candidate config hashes must be unique")
    result_ids = [result.candidate_id for result in search.results]
    result_run_ids = [result.run_id for result in search.results]
    if len(set(result_ids)) != SEARCH_BUDGET or set(result_ids) != declared_ids:
        raise SelectionValidationError("exact-30 result accounting does not match candidate manifest")
    if len(set(result_run_ids)) != SEARCH_BUDGET:
        raise SelectionValidationError("each search result requires a unique run ID")
    if canonical_sha256(search.declared_candidates) != search.candidate_list_hash:
        raise SelectionValidationError("candidate-list hash mismatch")
    if candidate.candidate_id not in declared_by_id:
        raise SelectionValidationError("winner is outside the declared 30-candidate search")
    if declared_by_id[candidate.candidate_id] != candidate.config_hash:
        raise SelectionValidationError("winner config hash does not match candidate manifest")
    try:
        recomputed = choose_best_within_family(search.results, candidate.task)
    except GovernanceError as error:
        raise SelectionValidationError(str(error)) from error
    if recomputed.candidate_id != search.referenced_best_candidate_id:
        raise SelectionValidationError("referenced within-family best is not the recomputed best")
    if candidate.candidate_id != recomputed.candidate_id or candidate.run_id != recomputed.run_id:
        raise SelectionValidationError("candidate summary is not the verified within-family winner")
    primary_key, _ = _primary_spec(candidate.task)
    if float(candidate.metrics[primary_key]) != float(recomputed.metrics[primary_key]):
        raise SelectionValidationError("winner primary metric does not match frozen search result")
    if candidate.task == "organ_support" and recomputed.probability_type != candidate.probability_type:
        raise SelectionValidationError("winner support probability provenance mismatch")
    if selection_mode == "real" and search.run_type != "scientific_validation":
        raise SelectionValidationError(REAL_SELECTION_BLOCKER)
    if selection_mode == "real" and any(
        result.status.upper() not in ("COMPLETE", "COMPLETED") for result in search.results
    ):
        raise SelectionValidationError(SEARCH_COMPLETION_BLOCKER)
    if selection_mode == "real" and candidate.code_commit.startswith("UNAVAILABLE"):
        raise SelectionValidationError("BLOCKED — CODE COMMIT PROVENANCE REQUIRED")
    if selection_mode == "synthetic" and search.run_type != "synthetic_dry_run":
        raise SelectionValidationError("synthetic selection accepts synthetic dry-run evidence only")


def _validate_artifact(candidate: CandidateSummary) -> None:
    artifact = Path(candidate.artifact_ref)
    metadata_path = Path(candidate.artifact_metadata_ref)
    if not artifact.is_file() or not metadata_path.is_file():
        raise SelectionValidationError("model artifact and compatibility metadata must exist")
    if sha256_file(artifact) != candidate.artifact_sha256:
        raise SelectionValidationError("model artifact SHA-256 mismatch")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise SelectionValidationError("model artifact metadata is unreadable") from error
    expected = {
        "task": candidate.task,
        "model_family": candidate.family,
        "feature_schema_version": candidate.feature_version,
        "split_hash": candidate.split_hash,
        "preprocessing_hash": candidate.preprocessing_hash,
        "label_version": candidate.label_version,
    }
    for field, expected_value in expected.items():
        if metadata.get(field) != expected_value:
            raise SelectionValidationError("artifact compatibility mismatch: " + field)


def validate_candidate(candidate: CandidateSummary, *, selection_mode: str) -> None:
    if selection_mode not in ("synthetic", "real"):
        raise SelectionValidationError("selection_mode must be synthetic or real")
    if candidate.task not in TASKS or candidate.family not in SERVING_FAMILIES:
        raise SelectionValidationError("only task-specific XGBoost/GRU candidates are eligible")
    if candidate.status.upper() not in ("COMPLETE", "COMPLETED"):
        raise SelectionValidationError("candidate status is not complete")
    if candidate.source_partition != "validation" or candidate.test_accessed is not False:
        raise SelectionValidationError("HARD FAIL — PHASE 12 TEST ACCESS FORBIDDEN")
    if candidate.metric_implementation_version != METRIC_IMPLEMENTATION_VERSION:
        raise SelectionValidationError("candidate did not use the authoritative Phase-9 metric version")
    key, _ = _primary_spec(candidate.task)
    value = candidate.metrics.get(key)
    if value is None or not math.isfinite(float(value)):
        raise SelectionValidationError("candidate lacks a finite primary validation metric")
    required_counts = {"N_examples", "N_ICU_stays"}
    if candidate.task == "organ_support":
        required_counts.update(("N_positive_examples", "N_positive_stays"))
        if candidate.probability_type != "raw_uncalibrated":
            raise SelectionValidationError(
                "support selection requires raw uncalibrated probabilities"
            )
    if not required_counts.issubset(candidate.counts):
        raise SelectionValidationError("candidate lacks required validation population counts")
    if any(
        isinstance(candidate.counts[name], bool)
        or not isinstance(candidate.counts[name], int)
        or candidate.counts[name] < 0
        for name in required_counts
    ):
        raise SelectionValidationError("validation population counts must be nonnegative integers")
    if candidate.counts["N_examples"] == 0 or candidate.counts["N_ICU_stays"] == 0:
        raise SelectionValidationError("eligible validation population must be nonempty")
    _validate_search_evidence(candidate, selection_mode=selection_mode)
    _validate_artifact(candidate)


def validate_pair(
    xgboost: CandidateSummary,
    gru: CandidateSummary,
    *,
    task: str,
    selection_mode: str,
) -> None:
    if xgboost.family != "xgboost" or gru.family != "gru":
        raise SelectionValidationError("comparison requires XGBoost and GRU in fixed family slots")
    if xgboost.task != task or gru.task != task:
        raise SelectionValidationError("candidate task does not match requested task")
    validate_candidate(xgboost, selection_mode=selection_mode)
    validate_candidate(gru, selection_mode=selection_mode)
    try:
        validate_information_parity(gru.information_manifest, xgboost.information_manifest)
    except GovernanceError as error:
        raise SelectionValidationError(INFORMATION_MISMATCH + ": " + str(error)) from error
    if xgboost.counts != gru.counts:
        raise SelectionValidationError("validation population parity failure")
    for field in ("split_hash", "feature_version", "label_version", "preprocessing_hash"):
        if getattr(xgboost, field) != getattr(gru, field):
            raise SelectionValidationError("candidate lineage parity failure: " + field)


def primary_metric(task: str, candidate: CandidateSummary):
    key, direction = _primary_spec(task)
    return key, direction, float(candidate.metrics[key])


def candidate_summary_from_mapping(payload: Mapping[str, object]) -> CandidateSummary:
    """Parse a JSON-compatible frozen summary after rejecting test-labelled keys."""

    reject_test_fields(payload)
    try:
        information = InformationManifest(**payload["information_manifest"])
        search_payload = payload["search"]
        results = tuple(CandidateResult(**item) for item in search_payload["results"])
        search = SearchEvidence(
            task=search_payload["task"],
            family=search_payload["family"],
            run_type=search_payload["run_type"],
            search_version=search_payload["search_version"],
            search_space_hash=search_payload["search_space_hash"],
            candidate_list_hash=search_payload["candidate_list_hash"],
            declared_candidates=tuple(search_payload["declared_candidates"]),
            results=results,
            referenced_best_candidate_id=search_payload["referenced_best_candidate_id"],
        )
        values = dict(payload)
        values["information_manifest"] = information
        values["search"] = search
        return CandidateSummary(**values)
    except (KeyError, TypeError) as error:
        raise SelectionValidationError("invalid frozen candidate-summary schema") from error


def load_candidate_summary(path) -> CandidateSummary:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise SelectionValidationError("candidate summary is unreadable") from error
    if not isinstance(payload, Mapping):
        raise SelectionValidationError("candidate summary must be a JSON object")
    return candidate_summary_from_mapping(payload)
