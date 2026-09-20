"""Narrow dependency interfaces for Phase-6 serving orchestration.

Implementations that touch raw history, build features, or compute real model
attributions belong to later Pulkit phases.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Protocol


class TaskPredictor(Protocol):
    task: str
    family: str
    model_version: str
    artifact_sha256: str

    def predict(self, prepared_input: object) -> object:
        """Return one task-specific model-domain output without training."""


class FrozenPreprocessor(Protocol):
    task: str
    artifact_sha256: str

    def transform(self, prepared_input: object) -> object:
        """Apply an already-fitted immutable transformation; no fit method exists."""


class CanonicalInputProvider(Protocol):
    def get_canonical_input(
        self,
        *,
        stay_id: object,
        prediction_time: str,
        task: str,
        family: str,
        feature_version: str,
    ) -> object:
        """Build the family view from cutoff-safe canonical history features."""

    def data_quality(self, *, stay_id: object, prediction_time: str) -> Mapping[str, int]:
        """Return explicit mask/count metadata for the schema payload."""


class ServingPostprocessor(Protocol):
    def recovery(
        self,
        raw_output: object,
        prepared_input: object,
        *,
        stay_id: object,
        prediction_time: str,
        model_identity: ModelIdentity,
    ) -> Mapping[str, float]:
        """Adapt independent recovery outputs to the Phase-5 response contract."""

    def icu_stay_time_hours(self, raw_output: object) -> float:
        """Adapt the frozen model-domain ICU output to hours."""


class SupportCalibrator(Protocol):
    artifact_sha256: str

    def transform(self, raw_probability: float) -> float:
        """Apply a frozen calibrator. Deliberately exposes no fitting interface."""


@dataclass(frozen=True)
class ModelIdentity:
    task: str
    family: str
    model_version: str
    artifact_ref: str
    artifact_sha256: str
    preprocessor_ref: str
    preprocessor_sha256: str
    feature_version: str
    label_version: str
    split_hash: str
    explanation_method: str
    target_transform_ref: Optional[str]
    target_transform_sha256: Optional[str]
