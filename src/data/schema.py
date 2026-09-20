"""Tensor-contract validation and synthetic canonical serialization.

This module validates representation only. It performs no feature building,
forward fill, imputation, scaling, label derivation, batching, or training.
"""

import json
import math
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

from data.split import era_to_split


Identifier = Union[int, str]
Number = Union[int, float]

TENSOR_CONTRACT_VERSION = "tensor_contract_v1"
TIMESTAMP_SPEC_VERSION = "timestamp_spec_v1"
SEQUENCE_LENGTH = 8
HISTORY_DTYPE = "float32"
MASK_DTYPE = "bool"
TSLO_UNRESOLVED_STATUS = "BLOCKED_TSLO_NO_OBSERVATION_REPRESENTATION_REQUIRED"
STATIC_UNRESOLVED_STATUS = "BLOCKED_STATIC_FEATURE_CONTRACT_REQUIRED"
SYNTHETIC_DATASET_KIND = "SYNTHETIC_CANONICAL_FIXTURE_NOT_REAL_CLINICAL_DATA"
SYNTHETIC_SERIALIZATION_STATUS = (
    "UNLOCKED_ENGINEERING_PARAMETER_SYNTHETIC_FIXTURES_ONLY"
)


class TensorContractError(ValueError):
    """Raised when a canonical example violates the tensor contract."""


@dataclass(frozen=True)
class FeatureSchemaReference:
    """External feature schema reference; real values come from Sanskruti."""

    version: str
    feature_names: Tuple[str, ...]
    status: str
    tslo_no_observation_value: Optional[float] = None
    tslo_no_observation_source: Optional[str] = None
    static_feature_names: Optional[Tuple[str, ...]] = None

    @property
    def feature_dim(self) -> int:
        return len(self.feature_names)


@dataclass(frozen=True)
class TaskEligibility:
    recovery24_eligible: bool
    recovery48_eligible: bool
    icu_time_eligible: bool
    organ_support_eligible: bool


@dataclass(frozen=True)
class TaskTargets:
    recovery24_delta_sofa: Optional[float]
    recovery48_delta_sofa: Optional[float]
    icu_time_log1p_hours: Optional[float]
    organ_support_target: Optional[int]


@dataclass(frozen=True)
class CanonicalExample:
    subject_id: Identifier
    stay_id: Identifier
    prediction_time: str
    grid_index: int
    icu_elapsed_hours: int
    anchor_year_group: str
    split: str
    tensor_contract_version: str
    timestamp_spec_version: str
    feature_schema_version: str
    history_dtype: str
    history_values: Tuple[Tuple[Optional[Number], ...], ...]
    padding_mask_dtype: str
    padding_mask: Tuple[bool, ...]
    observation_mask_dtype: str
    observation_mask: Tuple[Tuple[bool, ...], ...]
    tslo_hours: Optional[Tuple[Tuple[float, ...], ...]]
    tslo_status: str
    static_features: Optional[Tuple[Optional[Number], ...]]
    static_feature_names: Optional[Tuple[str, ...]]
    static_status: str
    eligibility: TaskEligibility
    targets: TaskTargets
    case_tags: Tuple[str, ...]


@dataclass(frozen=True)
class CanonicalDataset:
    dataset_kind: str
    serialization_status: str
    tensor_contract_version: str
    timestamp_spec_version: str
    feature_schema: FeatureSchemaReference
    fixture_semantics: Mapping[str, object]
    examples: Tuple[CanonicalExample, ...]


def _valid_identifier(value: object) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return True
    return isinstance(value, str) and bool(value.strip())


def _is_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _validate_matrix_shape(
    matrix: Sequence[Sequence[object]], rows: int, columns: int, name: str
) -> None:
    if len(matrix) != rows:
        raise TensorContractError(
            "{} must contain exactly {} temporal bins".format(name, rows)
        )
    if any(len(row) != columns for row in matrix):
        raise TensorContractError(
            "{} must align to feature dimension {}".format(name, columns)
        )


def _validate_targets(example: CanonicalExample) -> None:
    pairs = (
        (
            "recovery24",
            example.eligibility.recovery24_eligible,
            example.targets.recovery24_delta_sofa,
        ),
        (
            "recovery48",
            example.eligibility.recovery48_eligible,
            example.targets.recovery48_delta_sofa,
        ),
        (
            "icu_time",
            example.eligibility.icu_time_eligible,
            example.targets.icu_time_log1p_hours,
        ),
        (
            "organ_support",
            example.eligibility.organ_support_eligible,
            example.targets.organ_support_target,
        ),
    )
    for name, eligible, target in pairs:
        if eligible and target is None:
            raise TensorContractError(
                "{} is eligible but its synthetic target is absent".format(name)
            )
        if not eligible and target is not None:
            raise TensorContractError(
                "{} is ineligible but its target is populated".format(name)
            )
    for value in (
        example.targets.recovery24_delta_sofa,
        example.targets.recovery48_delta_sofa,
        example.targets.icu_time_log1p_hours,
    ):
        if value is not None and not _is_number(value):
            raise TensorContractError("regression target must be finite numeric or null")
    if (
        example.targets.organ_support_target is not None
        and example.targets.organ_support_target not in (0, 1)
    ):
        raise TensorContractError("organ_support_target must be 0, 1, or null")


def validate_canonical_example(
    example: CanonicalExample, feature_schema: FeatureSchemaReference
) -> None:
    """Validate one example without changing or preprocessing it."""

    if not _valid_identifier(example.subject_id) or not _valid_identifier(
        example.stay_id
    ):
        raise TensorContractError("subject_id and stay_id are required")
    try:
        datetime.fromisoformat(example.prediction_time)
    except (TypeError, ValueError) as error:
        raise TensorContractError("prediction_time must be ISO-8601") from error
    if isinstance(example.grid_index, bool) or not isinstance(example.grid_index, int):
        raise TensorContractError("grid_index must be an integer")
    if not 0 <= example.grid_index <= 11:
        raise TensorContractError("grid_index must be in 0..11")
    if example.icu_elapsed_hours != 24 + 6 * example.grid_index:
        raise TensorContractError("icu_elapsed_hours does not match timestamp grid")
    if example.split != era_to_split(example.anchor_year_group):
        raise TensorContractError("split does not match frozen anchor era")

    if example.tensor_contract_version != TENSOR_CONTRACT_VERSION:
        raise TensorContractError("tensor contract version mismatch")
    if example.timestamp_spec_version != TIMESTAMP_SPEC_VERSION:
        raise TensorContractError("timestamp specification version mismatch")
    if example.feature_schema_version != feature_schema.version:
        raise TensorContractError("feature schema version mismatch")
    if not feature_schema.feature_names:
        raise TensorContractError("feature schema must declare at least one feature")
    if len(set(feature_schema.feature_names)) != feature_schema.feature_dim:
        raise TensorContractError("feature names must be unique and ordered")

    if example.history_dtype != HISTORY_DTYPE:
        raise TensorContractError("history dtype must be float32")
    if example.padding_mask_dtype != MASK_DTYPE:
        raise TensorContractError("padding mask dtype must be bool")
    if example.observation_mask_dtype != MASK_DTYPE:
        raise TensorContractError("observation mask dtype must be bool")

    feature_dim = feature_schema.feature_dim
    _validate_matrix_shape(
        example.history_values, SEQUENCE_LENGTH, feature_dim, "history_values"
    )
    _validate_matrix_shape(
        example.observation_mask,
        SEQUENCE_LENGTH,
        feature_dim,
        "observation_mask",
    )
    if len(example.padding_mask) != SEQUENCE_LENGTH:
        raise TensorContractError("padding_mask must have shape [8]")
    if any(not isinstance(value, bool) for value in example.padding_mask):
        raise TensorContractError("padding_mask values must be bool")

    valid_bin_seen = False
    for bin_index in range(SEQUENCE_LENGTH):
        padded = example.padding_mask[bin_index]
        if not padded:
            valid_bin_seen = True
        elif valid_bin_seen:
            raise TensorContractError("padding must be a contiguous prefix")

        for feature_index in range(feature_dim):
            value = example.history_values[bin_index][feature_index]
            observed = example.observation_mask[bin_index][feature_index]
            if not isinstance(observed, bool):
                raise TensorContractError("observation_mask values must be bool")
            if padded and (value is not None or observed):
                raise TensorContractError(
                    "padded bins cannot contain clinical values or observations"
                )
            if not padded:
                if observed and not _is_number(value):
                    raise TensorContractError(
                        "genuine observations require finite numeric values"
                    )
                if not observed and value is not None:
                    raise TensorContractError(
                        "synthetic raw-missing values must remain null before imputation"
                    )

    if feature_schema.tslo_no_observation_source is None:
        if example.tslo_hours is not None or example.tslo_status != TSLO_UNRESOLVED_STATUS:
            raise TensorContractError(
                "TSLO must remain blocked until its no-observation representation is frozen"
            )
    else:
        if example.tslo_hours is None:
            raise TensorContractError("resolved TSLO contract requires [8,F] values")
        _validate_matrix_shape(
            example.tslo_hours, SEQUENCE_LENGTH, feature_dim, "tslo_hours"
        )
        for row in example.tslo_hours:
            if any(not _is_number(value) for value in row):
                raise TensorContractError("TSLO values must be finite numeric")

    if feature_schema.static_feature_names is None:
        if (
            example.static_features is not None
            or example.static_feature_names is not None
            or example.static_status != STATIC_UNRESOLVED_STATUS
        ):
            raise TensorContractError(
                "static features must remain blocked until their schema is frozen"
            )
    else:
        if example.static_feature_names != feature_schema.static_feature_names:
            raise TensorContractError("static feature order mismatch")
        if example.static_features is None or len(example.static_features) != len(
            feature_schema.static_feature_names
        ):
            raise TensorContractError("static feature vector shape mismatch")
        if any(value is not None and not _is_number(value) for value in example.static_features):
            raise TensorContractError("static values must be finite numeric or null")

    if any(not isinstance(value, bool) for value in example.eligibility.__dict__.values()):
        raise TensorContractError("all eligibility masks must be explicit booleans")
    _validate_targets(example)


def canonicalize_dataset(dataset: CanonicalDataset) -> CanonicalDataset:
    """Return deterministic subject/stay/prediction-time ordering."""

    ordered = tuple(
        sorted(
            dataset.examples,
            key=lambda item: (
                str(item.subject_id),
                str(item.stay_id),
                item.prediction_time,
            ),
        )
    )
    return CanonicalDataset(
        dataset_kind=dataset.dataset_kind,
        serialization_status=dataset.serialization_status,
        tensor_contract_version=dataset.tensor_contract_version,
        timestamp_spec_version=dataset.timestamp_spec_version,
        feature_schema=dataset.feature_schema,
        fixture_semantics=dataset.fixture_semantics,
        examples=ordered,
    )


def validate_canonical_dataset(
    dataset: CanonicalDataset, expected_feature_schema: FeatureSchemaReference
) -> None:
    """Validate versions, schema reference, examples, and canonical keys."""

    if dataset.dataset_kind != SYNTHETIC_DATASET_KIND:
        raise TensorContractError("Phase 3 accepts synthetic fixtures only")
    if dataset.serialization_status != SYNTHETIC_SERIALIZATION_STATUS:
        raise TensorContractError("unexpected synthetic serialization status")
    if dataset.tensor_contract_version != TENSOR_CONTRACT_VERSION:
        raise TensorContractError("dataset tensor contract version mismatch")
    if dataset.timestamp_spec_version != TIMESTAMP_SPEC_VERSION:
        raise TensorContractError("dataset timestamp specification mismatch")
    if dataset.feature_schema != expected_feature_schema:
        raise TensorContractError("dataset feature order/version does not match expectation")

    seen_keys = set()
    for example in dataset.examples:
        validate_canonical_example(example, expected_feature_schema)
        key = (type(example.stay_id), example.stay_id, example.prediction_time)
        if key in seen_keys:
            raise TensorContractError("duplicate canonical key stay_id+prediction_time")
        seen_keys.add(key)


def _example_from_mapping(data: Mapping[str, object]) -> CanonicalExample:
    eligibility_data = data["eligibility"]
    targets_data = data["targets"]
    if not isinstance(eligibility_data, Mapping) or not isinstance(targets_data, Mapping):
        raise TensorContractError("eligibility and targets must be objects")
    return CanonicalExample(
        subject_id=data["subject_id"],  # type: ignore[arg-type]
        stay_id=data["stay_id"],  # type: ignore[arg-type]
        prediction_time=data["prediction_time"],  # type: ignore[arg-type]
        grid_index=data["grid_index"],  # type: ignore[arg-type]
        icu_elapsed_hours=data["icu_elapsed_hours"],  # type: ignore[arg-type]
        anchor_year_group=data["anchor_year_group"],  # type: ignore[arg-type]
        split=data["split"],  # type: ignore[arg-type]
        tensor_contract_version=data["tensor_contract_version"],  # type: ignore[arg-type]
        timestamp_spec_version=data["timestamp_spec_version"],  # type: ignore[arg-type]
        feature_schema_version=data["feature_schema_version"],  # type: ignore[arg-type]
        history_dtype=data["history_dtype"],  # type: ignore[arg-type]
        history_values=tuple(tuple(row) for row in data["history_values"]),  # type: ignore[arg-type]
        padding_mask_dtype=data["padding_mask_dtype"],  # type: ignore[arg-type]
        padding_mask=tuple(data["padding_mask"]),  # type: ignore[arg-type]
        observation_mask_dtype=data["observation_mask_dtype"],  # type: ignore[arg-type]
        observation_mask=tuple(
            tuple(row) for row in data["observation_mask"]  # type: ignore[arg-type]
        ),
        tslo_hours=(
            None
            if data["tslo_hours"] is None
            else tuple(tuple(row) for row in data["tslo_hours"])  # type: ignore[arg-type]
        ),
        tslo_status=data["tslo_status"],  # type: ignore[arg-type]
        static_features=(
            None
            if data["static_features"] is None
            else tuple(data["static_features"])  # type: ignore[arg-type]
        ),
        static_feature_names=(
            None
            if data["static_feature_names"] is None
            else tuple(data["static_feature_names"])  # type: ignore[arg-type]
        ),
        static_status=data["static_status"],  # type: ignore[arg-type]
        eligibility=TaskEligibility(
            recovery24_eligible=eligibility_data["recovery24_eligible"],  # type: ignore[arg-type]
            recovery48_eligible=eligibility_data["recovery48_eligible"],  # type: ignore[arg-type]
            icu_time_eligible=eligibility_data["icu_time_eligible"],  # type: ignore[arg-type]
            organ_support_eligible=eligibility_data["organ_support_eligible"],  # type: ignore[arg-type]
        ),
        targets=TaskTargets(
            recovery24_delta_sofa=targets_data["recovery24_delta_sofa"],  # type: ignore[arg-type]
            recovery48_delta_sofa=targets_data["recovery48_delta_sofa"],  # type: ignore[arg-type]
            icu_time_log1p_hours=targets_data["icu_time_log1p_hours"],  # type: ignore[arg-type]
            organ_support_target=targets_data["organ_support_target"],  # type: ignore[arg-type]
        ),
        case_tags=tuple(data["case_tags"]),  # type: ignore[arg-type]
    )


def dataset_from_mapping(data: Mapping[str, object]) -> CanonicalDataset:
    """Parse a JSON-compatible mapping without performing preprocessing."""

    schema_data = data["synthetic_feature_schema"]
    if not isinstance(schema_data, Mapping):
        raise TensorContractError("synthetic_feature_schema must be an object")
    feature_schema = FeatureSchemaReference(
        version=schema_data["version"],  # type: ignore[arg-type]
        feature_names=tuple(schema_data["feature_names"]),  # type: ignore[arg-type]
        status=schema_data["status"],  # type: ignore[arg-type]
        tslo_no_observation_value=schema_data.get("tslo_no_observation_value"),  # type: ignore[arg-type]
        tslo_no_observation_source=schema_data.get("tslo_no_observation_source"),  # type: ignore[arg-type]
        static_feature_names=(
            None
            if schema_data.get("static_feature_names") is None
            else tuple(schema_data["static_feature_names"])  # type: ignore[arg-type]
        ),
    )
    examples_data = data["examples"]
    if not isinstance(examples_data, list):
        raise TensorContractError("examples must be a list")
    return CanonicalDataset(
        dataset_kind=data["dataset_kind"],  # type: ignore[arg-type]
        serialization_status=data["serialization_status"],  # type: ignore[arg-type]
        tensor_contract_version=data["tensor_contract_version"],  # type: ignore[arg-type]
        timestamp_spec_version=data["timestamp_spec_version"],  # type: ignore[arg-type]
        feature_schema=feature_schema,
        fixture_semantics=data["fixture_semantics"],  # type: ignore[arg-type]
        examples=tuple(_example_from_mapping(item) for item in examples_data),
    )


def _example_to_mapping(example: CanonicalExample) -> Mapping[str, object]:
    return {
        "subject_id": example.subject_id,
        "stay_id": example.stay_id,
        "prediction_time": example.prediction_time,
        "grid_index": example.grid_index,
        "icu_elapsed_hours": example.icu_elapsed_hours,
        "anchor_year_group": example.anchor_year_group,
        "split": example.split,
        "tensor_contract_version": example.tensor_contract_version,
        "timestamp_spec_version": example.timestamp_spec_version,
        "feature_schema_version": example.feature_schema_version,
        "history_dtype": example.history_dtype,
        "history_values": [list(row) for row in example.history_values],
        "padding_mask_dtype": example.padding_mask_dtype,
        "padding_mask": list(example.padding_mask),
        "observation_mask_dtype": example.observation_mask_dtype,
        "observation_mask": [list(row) for row in example.observation_mask],
        "tslo_hours": (
            None
            if example.tslo_hours is None
            else [list(row) for row in example.tslo_hours]
        ),
        "tslo_status": example.tslo_status,
        "static_features": (
            None if example.static_features is None else list(example.static_features)
        ),
        "static_feature_names": (
            None
            if example.static_feature_names is None
            else list(example.static_feature_names)
        ),
        "static_status": example.static_status,
        "eligibility": dict(example.eligibility.__dict__),
        "targets": dict(example.targets.__dict__),
        "case_tags": list(example.case_tags),
    }


def dataset_to_mapping(dataset: CanonicalDataset) -> Mapping[str, object]:
    canonical = canonicalize_dataset(dataset)
    return {
        "dataset_kind": canonical.dataset_kind,
        "serialization_status": canonical.serialization_status,
        "tensor_contract_version": canonical.tensor_contract_version,
        "timestamp_spec_version": canonical.timestamp_spec_version,
        "synthetic_feature_schema": {
            "version": canonical.feature_schema.version,
            "feature_names": list(canonical.feature_schema.feature_names),
            "status": canonical.feature_schema.status,
            "tslo_no_observation_value": canonical.feature_schema.tslo_no_observation_value,
            "tslo_no_observation_source": canonical.feature_schema.tslo_no_observation_source,
            "static_feature_names": (
                None
                if canonical.feature_schema.static_feature_names is None
                else list(canonical.feature_schema.static_feature_names)
            ),
        },
        "fixture_semantics": canonical.fixture_semantics,
        "examples": [_example_to_mapping(item) for item in canonical.examples],
    }


def serialize_canonical_dataset(
    dataset: CanonicalDataset, expected_feature_schema: FeatureSchemaReference
) -> bytes:
    """Validate and serialize deterministic synthetic JSON bytes."""

    canonical = canonicalize_dataset(dataset)
    validate_canonical_dataset(canonical, expected_feature_schema)
    text = json.dumps(
        dataset_to_mapping(canonical),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return (text + "\n").encode("utf-8")


def deserialize_canonical_dataset(
    content: bytes, expected_feature_schema: FeatureSchemaReference
) -> CanonicalDataset:
    """Deserialize, canonicalize, and validate synthetic fixture bytes."""

    try:
        payload = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise TensorContractError("synthetic fixture is not valid UTF-8 JSON") from error
    if not isinstance(payload, Mapping):
        raise TensorContractError("synthetic fixture root must be an object")
    dataset = canonicalize_dataset(dataset_from_mapping(payload))
    validate_canonical_dataset(dataset, expected_feature_schema)
    return dataset


def temporal_information_views(example: CanonicalExample) -> Mapping[str, object]:
    """Expose parity views without adding model-family-specific information."""

    history_flat = tuple(value for row in example.history_values for value in row)
    observation_flat = tuple(
        value for row in example.observation_mask for value in row
    )
    tslo_flat = (
        None
        if example.tslo_hours is None
        else tuple(value for row in example.tslo_hours for value in row)
    )
    return {
        "gru": {
            "history_values": example.history_values,
            "observation_mask": example.observation_mask,
            "padding_mask": example.padding_mask,
            "tslo_hours": example.tslo_hours,
            "static_features": example.static_features,
        },
        "xgboost": {
            "history_values": history_flat,
            "observation_mask": observation_flat,
            "padding_mask": example.padding_mask,
            "tslo_hours": tslo_flat,
            "static_features": example.static_features,
        },
    }


def unflatten_temporal(values: Sequence[object], feature_dim: int) -> Tuple[Tuple[object, ...], ...]:
    """Reconstruct [8,F] solely for information-parity assertions."""

    if len(values) != SEQUENCE_LENGTH * feature_dim:
        raise TensorContractError("flattened temporal view has incompatible length")
    return tuple(
        tuple(values[index * feature_dim : (index + 1) * feature_dim])
        for index in range(SEQUENCE_LENGTH)
    )
