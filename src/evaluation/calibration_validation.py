"""Contracts and fail-closed validation for Phase-13 support calibration."""

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Hashable, Mapping, Optional, Tuple

from evaluation.selection_validation import reject_test_fields, sha256_file
from experiments.search_governance import canonical_sha256


CALIBRATION_IMPLEMENTATION_VERSION = "support_isotonic_calibration_v1"
PREDICTION_ARTIFACT_VERSION = "support_validation_predictions_v1"
REAL_SELECTED_MODEL_BLOCKER = "BLOCKED — REAL SELECTED ORGAN-SUPPORT CLASSIFIER REQUIRED"
REAL_PREREQUISITES_BLOCKER = "BLOCKED — REAL CALIBRATION PREREQUISITES INCOMPLETE"
ISOTONIC_DETAIL_UNLOCKED = "UNLOCKED ENGINEERING PARAMETER — ISOTONIC IMPLEMENTATION DETAIL"
ISOTONIC_WEIGHTING_UNLOCKED = (
    "UNLOCKED ENGINEERING PARAMETER — ISOTONIC CALIBRATION SAMPLE WEIGHTING"
)


class CalibrationValidationError(ValueError):
    """Raised when calibration input or lineage is unsafe or incompatible."""


@dataclass(frozen=True)
class SupportPredictionRecord:
    row_id: str
    stay_id: Hashable
    raw_probability: float
    label: int
    eligible: bool


@dataclass(frozen=True)
class SelectedSupportModel:
    task: str
    family: str
    run_id: str
    candidate_id: str
    config_hash: str
    artifact_ref: str
    artifact_sha256: str
    split_hash: str
    feature_version: str
    label_version: str
    preprocessing_hash: str
    code_commit: str
    selection_metric: str
    probability_type: str
    validation_prediction_ref: str
    validation_prediction_sha256: str
    selection_mode: str
    test_accessed: bool
    selection_manifest_ref: Optional[str] = None
    selection_manifest_sha256: Optional[str] = None


@dataclass(frozen=True)
class IsotonicPolicy:
    policy_version: str
    out_of_bounds: str
    sample_weighting: str
    increasing: bool = True
    y_min: float = 0.0
    y_max: float = 1.0
    approved_for_real: bool = False


@dataclass(frozen=True)
class FittedSupportCalibrator:
    artifact_version: str
    implementation_version: str
    implementation_library: str
    implementation_library_version: str
    policy: IsotonicPolicy
    x_thresholds: Tuple[float, ...]
    y_thresholds: Tuple[float, ...]
    selected_family: str
    selected_run_id: str
    selected_candidate_id: str
    selected_config_hash: str
    selected_model_sha256: str
    validation_prediction_ref: str
    validation_prediction_sha256: str
    validation_split_hash: str
    feature_version: str
    label_version: str
    preprocessing_hash: str
    eligible_population_hash: str
    counts: Mapping[str, int]
    code_commit: str
    calibration_mode: str
    governance: Optional[Mapping[str, object]] = None
    test_accessed: bool = False


def _require_nonempty(value, field):
    if not isinstance(value, str) or not value.strip():
        raise CalibrationValidationError(field + " is required")


def validate_selected_support_model(model: SelectedSupportModel, *, mode: str) -> None:
    if mode not in ("synthetic", "real"):
        raise CalibrationValidationError("calibration mode must be synthetic or real")
    if model.task != "organ_support":
        raise CalibrationValidationError("Phase 13 calibrates organ support only")
    if model.family not in ("xgboost", "gru"):
        raise CalibrationValidationError("selected support family must be XGBoost or GRU")
    if model.selection_metric != "validation_auprc":
        raise CalibrationValidationError("support model must be selected by raw validation AUPRC")
    if model.probability_type != "raw_uncalibrated":
        raise CalibrationValidationError("calibration input must be raw model probabilities")
    if model.test_accessed is not False:
        raise CalibrationValidationError("HARD FAIL — PHASE 13 TEST ACCESS FORBIDDEN")
    for field in (
        "run_id",
        "candidate_id",
        "config_hash",
        "artifact_ref",
        "artifact_sha256",
        "split_hash",
        "feature_version",
        "label_version",
        "preprocessing_hash",
        "validation_prediction_ref",
        "validation_prediction_sha256",
    ):
        _require_nonempty(getattr(model, field), field)
    artifact = Path(model.artifact_ref)
    if not artifact.is_file() or sha256_file(artifact) != model.artifact_sha256:
        raise CalibrationValidationError("selected support model artifact/hash mismatch")
    prediction = Path(model.validation_prediction_ref)
    if not prediction.is_file() or sha256_file(prediction) != model.validation_prediction_sha256:
        raise CalibrationValidationError("validation prediction artifact/hash mismatch")
    if mode == "real":
        if model.selection_mode != "real":
            raise CalibrationValidationError(REAL_SELECTED_MODEL_BLOCKER)
        if model.code_commit.startswith("UNAVAILABLE"):
            raise CalibrationValidationError("BLOCKED — CODE COMMIT PROVENANCE REQUIRED")
        if not model.selection_manifest_ref or not model.selection_manifest_sha256:
            raise CalibrationValidationError(REAL_SELECTED_MODEL_BLOCKER)
        manifest_path = Path(model.selection_manifest_ref)
        if (
            not manifest_path.is_file()
            or sha256_file(manifest_path) != model.selection_manifest_sha256
        ):
            raise CalibrationValidationError("selected-model manifest artifact/hash mismatch")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("manifest_version") == "validation_family_selection_v1":
                support = manifest.get("tasks", {}).get("organ_support", {})
                if (manifest.get("status") != "VALIDATION_FAMILY_SELECTION_FROZEN"
                        or support.get("selected_family") != model.family
                        or support.get("selected_candidate_id") != model.candidate_id
                        or support.get("selected_run_id") != model.run_id
                        or manifest.get("test_accessed") is not False):
                    raise CalibrationValidationError(
                        "model is not the Stage-2 selected support classifier"
                    )
            else:
                from evaluation.select import validate_selection_stage_manifest
                validate_selection_stage_manifest(manifest)
        except (OSError, ValueError, KeyError) as error:
            raise CalibrationValidationError("selected-model manifest is invalid") from error
        support = manifest["tasks"]["organ_support"]
        identity = {
            "family": model.family,
            "run_id": model.run_id,
            "candidate_id": model.candidate_id,
            "config_hash": model.config_hash,
            "artifact_sha256": model.artifact_sha256,
        }
        if any(support.get(field) != value for field, value in identity.items()):
            raise CalibrationValidationError(
                "model is not the manifest-selected support classifier"
            )
    elif model.selection_mode != "synthetic":
        raise CalibrationValidationError("synthetic calibration requires synthetic selection evidence")


def validate_isotonic_policy(policy: IsotonicPolicy, *, mode: str) -> None:
    if policy.out_of_bounds not in ("clip", "raise"):
        raise CalibrationValidationError(ISOTONIC_DETAIL_UNLOCKED)
    if policy.sample_weighting not in ("stay_balanced", "uniform_rows"):
        raise CalibrationValidationError(ISOTONIC_WEIGHTING_UNLOCKED)
    if policy.increasing is not True or policy.y_min != 0.0 or policy.y_max != 1.0:
        raise CalibrationValidationError("isotonic probability bounds/direction are incompatible")
    _require_nonempty(policy.policy_version, "isotonic policy_version")
    if mode == "real" and policy.approved_for_real is not True:
        raise CalibrationValidationError(
            REAL_PREREQUISITES_BLOCKER + "; " + ISOTONIC_DETAIL_UNLOCKED
        )


def load_validation_predictions(model: SelectedSupportModel) -> Tuple[SupportPredictionRecord, ...]:
    try:
        text = Path(model.validation_prediction_ref).read_text(encoding="utf-8")
    except OSError as error:
        raise CalibrationValidationError("validation prediction artifact is unreadable") from error
    try:
        payload = json.loads(text)
    except ValueError:
        # Scientific Phase-12 predictions are immutable JSONL.  Accept that
        # repository-native representation without regenerating inference.
        try:
            source_rows = [json.loads(line) for line in text.splitlines() if line.strip()]
        except ValueError as error:
            raise CalibrationValidationError("validation prediction artifact is unreadable") from error
        rows = []
        for row in source_rows:
            rows.append({
                "row_id": "{}|{}|{}".format(row.get("stay_id"), row.get("prediction_time"), row.get("grid_index")),
                "stay_id": row.get("stay_id"),
                "organ_support_eligible": row.get("eligible"),
                "raw_probability": row.get("prediction"),
                "label": row.get("true_target"),
            })
        payload = {
            "artifact_version": PREDICTION_ARTIFACT_VERSION,
            "task": "organ_support", "family": model.family, "run_id": model.run_id,
            "candidate_id": model.candidate_id, "config_hash": model.config_hash,
            "model_sha256": model.artifact_sha256, "split_hash": model.split_hash,
            "feature_version": model.feature_version, "label_version": model.label_version,
            "probability_type": "raw_uncalibrated", "partition": "validation",
            "test_accessed": False, "rows": rows,
        }
    if not isinstance(payload, Mapping):
        raise CalibrationValidationError("validation prediction artifact must be a JSON object")
    try:
        reject_test_fields(payload, path="validation_prediction_artifact")
    except ValueError as error:
        raise CalibrationValidationError(str(error)) from error
    expected = {
        "artifact_version": PREDICTION_ARTIFACT_VERSION,
        "task": "organ_support",
        "family": model.family,
        "run_id": model.run_id,
        "candidate_id": model.candidate_id,
        "config_hash": model.config_hash,
        "model_sha256": model.artifact_sha256,
        "split_hash": model.split_hash,
        "feature_version": model.feature_version,
        "label_version": model.label_version,
        "probability_type": "raw_uncalibrated",
        "partition": "validation",
        "test_accessed": False,
    }
    for field, expected_value in expected.items():
        if payload.get(field) != expected_value:
            raise CalibrationValidationError("prediction artifact compatibility mismatch: " + field)
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise CalibrationValidationError("validation prediction artifact requires rows")
    converted = []
    seen = set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise CalibrationValidationError("prediction rows must be objects")
        row_id = row.get("row_id")
        stay_id = row.get("stay_id")
        eligible = row.get("organ_support_eligible")
        if not isinstance(row_id, str) or not row_id or row_id in seen:
            raise CalibrationValidationError("prediction row IDs must be unique nonempty strings")
        seen.add(row_id)
        if stay_id is None or not isinstance(eligible, bool):
            raise CalibrationValidationError("prediction row requires stay_id and boolean eligibility")
        if eligible:
            raw = row.get("raw_probability")
            label = row.get("label")
            if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                raise CalibrationValidationError("eligible raw probabilities must be numeric")
            if not math.isfinite(float(raw)) or not 0.0 <= float(raw) <= 1.0:
                raise CalibrationValidationError("eligible raw probabilities must be within [0,1]")
            if label not in (0, 1):
                raise CalibrationValidationError("eligible support labels must be binary")
            converted.append(
                SupportPredictionRecord(row_id, stay_id, float(raw), int(label), True)
            )
    if not converted or {record.label for record in converted} != {0, 1}:
        raise CalibrationValidationError("eligible validation rows must contain both classes")
    return tuple(converted)


def eligible_population_hash(records: Tuple[SupportPredictionRecord, ...]) -> str:
    identity = tuple(
        {
            "row_id": record.row_id,
            "stay_id": record.stay_id,
            "label": record.label,
            "eligible": record.eligible,
        }
        for record in records
    )
    return canonical_sha256(identity)


def canonical_json_bytes(value) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()
