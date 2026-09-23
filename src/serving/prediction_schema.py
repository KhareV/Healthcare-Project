"""Deterministic validator for the Phase-5 prediction contract.

This module validates interface data only. It does not build features, load
models, run inference, reconstruct SOFA, calibrate probabilities, or explain
predictions.
"""

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Mapping, Sequence

from evaluation.selection_validation import EXPLANATION_METHOD, SERVING_FAMILIES, TASKS
from vedant_infra.hashing import is_sha256, sha256_file


SCHEMA_VERSION = "prediction_schema_v1"
REPLAY_MODE = "RETROSPECTIVE_SEQUENTIAL_REPLAY"
FIXTURE_CLASSIFICATION = (
    "SYNTHETIC_MOCK_CONTRACT_OUTPUT_NOT_MODEL_PERFORMANCE_NOT_CLINICAL_DATA"
)
TOP_LEVEL_RESPONSE = {
    "schema_version",
    "prediction_time",
    "mode",
    "model_versions",
    "recovery",
    "icu_stay_time_hours",
    "organ_support_probability_calibrated",
    "explanation_features",
    "data_quality",
    "model_metadata",
}
FORBIDDEN_CONFIDENCE_KEYS = {
    "recovery_confidence",
    "icu_time_confidence",
    "lower_ci",
    "upper_ci",
    "prediction_interval",
    "certainty_percentage",
    "quality_score",
    "confidence_score",
}
PROHIBITED_TERMINOLOGY = (
    "hospital discharge eta",
    "survival prediction",
    "recovery time",
    "real-time clinical prediction",
    "clinical confidence",
    "patient prediction interval",
    "escalation risk",
    "causal explanation",
)


class PredictionSchemaError(ValueError):
    """Raised when a request, response, or fixture violates the v1 contract."""


def load_contract(path: Path) -> Mapping[str, object]:
    try:
        contract = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PredictionSchemaError("prediction contract is unreadable") from error
    if not isinstance(contract, dict) or contract.get("contract_version") != SCHEMA_VERSION:
        raise PredictionSchemaError("prediction contract version mismatch")
    if contract.get("validation_format") != "project_explicit_contract_v1":
        raise PredictionSchemaError("prediction validation format mismatch")
    return contract


def schema_sha256(path: Path) -> str:
    load_contract(path)
    return sha256_file(path)


def _exact_keys(value: object, expected: set[str], field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise PredictionSchemaError(field + " must be an object")
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise PredictionSchemaError(
            field + " fields mismatch; missing={} extra={}".format(missing, extra)
        )
    return value


def _identifier(value: object) -> bool:
    return (isinstance(value, int) and not isinstance(value, bool)) or (
        isinstance(value, str) and bool(value.strip())
    )


def _timestamp(value: object, field: str) -> datetime:
    if not isinstance(value, str):
        raise PredictionSchemaError(field + " must be a timezone-aware ISO-8601 string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise PredictionSchemaError(field + " is malformed") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise PredictionSchemaError(field + " must be timezone-aware")
    return parsed


def _finite(value: object, field: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PredictionSchemaError(field + " must be a number")
    number = float(value)
    if not math.isfinite(number) or (nonnegative and number < 0):
        raise PredictionSchemaError(field + " is outside its finite range")
    return number


def _positive_int(value: object, field: str, *, allow_zero: bool = True) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise PredictionSchemaError(field + " must be an integer")
    minimum = 0 if allow_zero else 1
    if value < minimum:
        raise PredictionSchemaError(field + " is outside its count range")
    return value


def _nonempty(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PredictionSchemaError(field + " must be a non-empty string")
    return value


def _reject_forbidden_keys(value: object, path: str = "response") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key).lower() in FORBIDDEN_CONFIDENCE_KEYS:
                raise PredictionSchemaError("unsupported confidence field: " + path + "." + str(key))
            _reject_forbidden_keys(child, path + "." + str(key))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_forbidden_keys(child, "{}[{}]".format(path, index))


def validate_request(payload: object) -> Mapping[str, object]:
    request = _exact_keys(payload, {"stay_id", "prediction_time"}, "request")
    if not _identifier(request["stay_id"]):
        raise PredictionSchemaError("stay_id must be a non-empty integer or string")
    _timestamp(request["prediction_time"], "prediction_time")
    return request


def _validate_artifact_identity(value: object, field: str, *, synthetic: bool) -> None:
    identity = _nonempty(value, field)
    if synthetic:
        if identity != "SYNTHETIC_HASH_NOT_REAL" and not is_sha256(identity):
            raise PredictionSchemaError(
                field + " must use a synthetic placeholder or exact fixture SHA-256"
            )
    elif not is_sha256(identity):
        raise PredictionSchemaError(field + " must be a real lowercase SHA-256 identity")


def _validate_model_versions(value: object, *, synthetic: bool) -> Mapping[str, object]:
    versions = _exact_keys(value, set(TASKS), "model_versions")
    for task in TASKS:
        expected = {"family", "model_version", "artifact_sha256", "preprocessor_sha256"}
        if task == "organ_support":
            expected.add("calibrator_sha256")
        entry = _exact_keys(versions[task], expected, "model_versions." + task)
        if entry["family"] not in SERVING_FAMILIES:
            raise PredictionSchemaError("serving family must be xgboost or gru")
        version = _nonempty(entry["model_version"], "model_versions." + task + ".model_version")
        if synthetic and not version.startswith("SYNTHETIC_"):
            raise PredictionSchemaError("synthetic model version must be unmistakably synthetic")
        if not synthetic and version.startswith("SYNTHETIC_"):
            raise PredictionSchemaError("synthetic model version cannot authorize real serving")
        for field in expected - {"family", "model_version"}:
            _validate_artifact_identity(entry[field], "model_versions." + task + "." + field, synthetic=synthetic)
    return versions


def _validate_recovery(value: object) -> None:
    required = {"delta_24h", "delta_48h", "reconstructed_sofa_24h", "reconstructed_sofa_48h"}
    recovery = _exact_keys(value, required, "recovery")
    for field in required:
        _finite(recovery[field], "recovery." + field)


def _validate_explanations(value: object, versions: Mapping[str, object]) -> None:
    explanations = _exact_keys(value, set(TASKS), "explanation_features")
    for task in TASKS:
        entry = _exact_keys(
            explanations[task], {"family", "explanation_method", "items"},
            "explanation_features." + task,
        )
        family = versions[task]["family"]
        if entry["family"] != family or entry["explanation_method"] != EXPLANATION_METHOD[family]:
            raise PredictionSchemaError("explanation routing does not match selected family for " + task)
        items = entry["items"]
        if not isinstance(items, list):
            raise PredictionSchemaError("explanation items must be a list")
        for index, item in enumerate(items):
            item = _exact_keys(item, {"feature_name", "attribution"}, "explanation item")
            _nonempty(item["feature_name"], "feature_name")
            _finite(item["attribution"], "attribution")


def _validate_data_quality(value: object) -> None:
    required = {
        "total_bins", "observed_bins", "padding_bins", "total_feature_values",
        "observed_feature_values", "missing_feature_values",
    }
    quality = _exact_keys(value, required, "data_quality")
    for field in required:
        _positive_int(quality[field], "data_quality." + field)
    if quality["total_bins"] != 8:
        raise PredictionSchemaError("data_quality.total_bins must match the eight-bin tensor contract")
    if quality["observed_bins"] + quality["padding_bins"] > quality["total_bins"]:
        raise PredictionSchemaError("observed and padding bin counts exceed total bins")
    if quality["observed_feature_values"] + quality["missing_feature_values"] != quality["total_feature_values"]:
        raise PredictionSchemaError("feature observation/missingness counts do not reconcile")


def _validate_version_map(value: object, field: str, *, synthetic: bool) -> None:
    versions = _exact_keys(value, set(TASKS), field)
    for task in TASKS:
        version = _nonempty(versions[task], field + "." + task)
        if synthetic and not version.startswith("SYNTHETIC_"):
            raise PredictionSchemaError(field + " fixture version must be synthetic")
        if not synthetic and version.startswith("SYNTHETIC_"):
            raise PredictionSchemaError(field + " synthetic version cannot authorize real serving")


def _validate_metadata(value: object, *, synthetic: bool) -> None:
    metadata = _exact_keys(
        value, {"feature_version", "label_version", "split_version", "manifest_version"},
        "model_metadata",
    )
    _validate_version_map(metadata["feature_version"], "feature_version", synthetic=synthetic)
    _validate_version_map(metadata["label_version"], "label_version", synthetic=synthetic)
    version = _nonempty(metadata["manifest_version"], "model_metadata.manifest_version")
    if synthetic and "synthetic" not in version.lower():
        raise PredictionSchemaError("manifest_version fixture value must be synthetic")
    if not synthetic and "synthetic" in version.lower():
        raise PredictionSchemaError("manifest_version synthetic value cannot authorize real serving")
    split_version = _nonempty(metadata["split_version"], "model_metadata.split_version")
    # split_version cannot use the same lowercase-substring check: the
    # accepted scientific project scope is itself a synthetic cardiac
    # benchmark, so the real, G3-frozen split version legitimately contains
    # the lowercase word "synthetic" (e.g. "synthetic_split_v2") without
    # being a Phase-6 mock placeholder. Use the same unmistakable-placeholder
    # convention as _validate_version_map instead (an uppercase "SYNTHETIC_"
    # prefix), which the Phase-6 fixture's split version already follows.
    if synthetic and not split_version.startswith("SYNTHETIC_"):
        raise PredictionSchemaError("split_version fixture value must be synthetic")
    if not synthetic and split_version.startswith("SYNTHETIC_"):
        raise PredictionSchemaError("split_version synthetic value cannot authorize real serving")


def validate_response(payload: object, *, synthetic: bool = False) -> Mapping[str, object]:
    response = _exact_keys(payload, TOP_LEVEL_RESPONSE, "response")
    _reject_forbidden_keys(response)
    if response["schema_version"] != SCHEMA_VERSION:
        raise PredictionSchemaError("response schema version mismatch")
    _timestamp(response["prediction_time"], "prediction_time")
    if response["mode"] != REPLAY_MODE:
        raise PredictionSchemaError("unsupported serving mode")
    versions = _validate_model_versions(response["model_versions"], synthetic=synthetic)
    _validate_recovery(response["recovery"])
    _finite(response["icu_stay_time_hours"], "icu_stay_time_hours", nonnegative=True)
    probability = _finite(
        response["organ_support_probability_calibrated"],
        "organ_support_probability_calibrated",
    )
    if probability < 0 or probability > 1:
        raise PredictionSchemaError("calibrated organ-support probability must be in [0,1]")
    _validate_explanations(response["explanation_features"], versions)
    _validate_data_quality(response["data_quality"])
    _validate_metadata(response["model_metadata"], synthetic=synthetic)
    return response


def validate_exchange(request: object, response: object, *, synthetic: bool = False) -> None:
    request_value = validate_request(request)
    response_value = validate_response(response, synthetic=synthetic)
    if request_value["prediction_time"] != response_value["prediction_time"]:
        raise PredictionSchemaError("response prediction_time differs from requested cutoff")


def load_synthetic_fixture(path: Path, *, payload_kind: str) -> Mapping[str, object]:
    try:
        fixture = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PredictionSchemaError("synthetic fixture is unreadable") from error
    fixture = _exact_keys(fixture, {"fixture_classification", "payload_kind", "payload"}, "fixture")
    if fixture["fixture_classification"] != FIXTURE_CLASSIFICATION or fixture["payload_kind"] != payload_kind:
        raise PredictionSchemaError("fixture is not explicitly classified as synthetic contract data")
    if payload_kind == "request":
        return validate_request(fixture["payload"])
    if payload_kind == "response":
        return validate_response(fixture["payload"], synthetic=True)
    raise PredictionSchemaError("unknown fixture payload kind")


def lint_contract_text(text: str) -> Sequence[str]:
    lowered = text.lower()
    return tuple(term for term in PROHIBITED_TERMINOLOGY if term in lowered)
