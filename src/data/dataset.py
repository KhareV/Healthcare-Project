"""PyTorch Dataset for validated canonical examples.

Only explicit synthetic tensorization is supported in Phase 4. No production
imputer, normalizer, encoder, target scaler, or clinical feature builder lives
in this module.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple, Union

import torch
from torch.utils.data import Dataset

from data.schema import (
    CanonicalDataset,
    CanonicalExample,
    FeatureSchemaReference,
    deserialize_canonical_dataset,
    validate_canonical_dataset,
)
from data.split import load_split_csv


PathLike = Union[str, Path]
PARTITIONS = ("train", "validation", "test")
SYNTHETIC_TENSORIZATION_STATUS = (
    "ENGINEERING_TEST_DEFAULT_NOT_PRODUCTION_PREPROCESSING"
)


class DatasetContractError(ValueError):
    """Raised when canonical data cannot safely become model input."""


@dataclass(frozen=True)
class SyntheticTensorizationPolicy:
    """Explicit test-only null filling needed to construct dense tensors."""

    missing_numeric_fill_value: float
    status: str = SYNTHETIC_TENSORIZATION_STATUS

    def validate(self) -> None:
        if self.status != SYNTHETIC_TENSORIZATION_STATUS:
            raise DatasetContractError("tensorization policy is not synthetic-test-only")
        if isinstance(self.missing_numeric_fill_value, bool) or not isinstance(
            self.missing_numeric_fill_value, (int, float)
        ):
            raise DatasetContractError("synthetic missing fill must be numeric")


class CanonicalTensorDataset(Dataset):
    """Model-family-neutral view of one validated subject partition."""

    def __init__(
        self,
        canonical_dataset: CanonicalDataset,
        expected_feature_schema: FeatureSchemaReference,
        partition: str,
        tensorization_policy: SyntheticTensorizationPolicy,
        expected_subject_splits: Mapping[object, str],
        *,
        allow_empty: bool = False,
    ) -> None:
        validate_canonical_dataset(canonical_dataset, expected_feature_schema)
        tensorization_policy.validate()
        if partition not in PARTITIONS:
            raise DatasetContractError("partition must be train, validation, or test")
        for example in canonical_dataset.examples:
            expected_split = expected_subject_splits.get(example.subject_id)
            if expected_split is None:
                raise DatasetContractError(
                    "subject is absent from the supplied split artifact"
                )
            if example.split != expected_split:
                raise DatasetContractError(
                    "canonical example conflicts with supplied subject split"
                )
        examples = tuple(
            example for example in canonical_dataset.examples if example.split == partition
        )
        if not examples and not allow_empty:
            raise DatasetContractError("requested partition contains no canonical examples")
        self._canonical_dataset = canonical_dataset
        self._feature_schema = expected_feature_schema
        self._partition = partition
        self._policy = tensorization_policy
        self._examples = examples

    @classmethod
    def from_fixture(
        cls,
        fixture_path: PathLike,
        expected_feature_schema: FeatureSchemaReference,
        partition: str,
        tensorization_policy: SyntheticTensorizationPolicy,
        split_path: PathLike,
        *,
        allow_empty: bool = False,
    ) -> "CanonicalTensorDataset":
        dataset = deserialize_canonical_dataset(
            Path(fixture_path).read_bytes(), expected_feature_schema
        )
        split_rows = load_split_csv(split_path)
        expected_subject_splits = {
            row.subject_id: row.split for row in split_rows
        }
        return cls(
            dataset,
            expected_feature_schema,
            partition,
            tensorization_policy,
            expected_subject_splits,
            allow_empty=allow_empty,
        )

    @property
    def partition(self) -> str:
        return self._partition

    @property
    def feature_names(self) -> Tuple[str, ...]:
        return self._feature_schema.feature_names

    @property
    def feature_schema_version(self) -> str:
        return self._feature_schema.version

    @property
    def tensor_contract_version(self) -> str:
        return self._canonical_dataset.tensor_contract_version

    @property
    def timestamp_spec_version(self) -> str:
        return self._canonical_dataset.timestamp_spec_version

    @property
    def examples(self) -> Tuple[CanonicalExample, ...]:
        return self._examples

    def __len__(self) -> int:
        return len(self._examples)

    def _sequence_tensor(self, example: CanonicalExample) -> torch.Tensor:
        fill = float(self._policy.missing_numeric_fill_value)
        dense = [
            [fill if value is None else float(value) for value in row]
            for row in example.history_values
        ]
        return torch.tensor(dense, dtype=torch.float32)

    def __getitem__(self, index: int) -> Mapping[str, object]:
        example = self._examples[index]
        return {
            "identifiers": {
                "subject_id": example.subject_id,
                "stay_id": example.stay_id,
                "prediction_time": example.prediction_time,
                "grid_index": example.grid_index,
                "split": example.split,
            },
            "sequence": self._sequence_tensor(example),
            "padding_mask": torch.tensor(example.padding_mask, dtype=torch.bool),
            "observation_mask": torch.tensor(
                example.observation_mask, dtype=torch.bool
            ),
            "tslo": (
                None
                if example.tslo_hours is None
                else torch.tensor(example.tslo_hours, dtype=torch.float32)
            ),
            "static_features": (
                None
                if example.static_features is None
                else torch.tensor(example.static_features, dtype=torch.float32)
            ),
            "targets": {
                "recovery24": example.targets.recovery24_delta_sofa,
                "recovery48": example.targets.recovery48_delta_sofa,
                "icu_time": example.targets.icu_time_log1p_hours,
                "organ_support": example.targets.organ_support_target,
            },
            "eligibility": {
                "recovery24": example.eligibility.recovery24_eligible,
                "recovery48": example.eligibility.recovery48_eligible,
                "icu_time": example.eligibility.icu_time_eligible,
                "organ_support": example.eligibility.organ_support_eligible,
            },
            "versions": {
                "tensor_contract": example.tensor_contract_version,
                "timestamp_spec": example.timestamp_spec_version,
                "feature_schema": example.feature_schema_version,
            },
            "feature_names": self.feature_names,
            "synthetic_tensorization_status": self._policy.status,
        }


def subjects_in_dataset(dataset: CanonicalTensorDataset) -> frozenset:
    """Return subject IDs for explicit partition-isolation assertions."""

    return frozenset(example.subject_id for example in dataset.examples)
