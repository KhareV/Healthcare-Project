"""Pydantic v2 mirrors of the Phase-5 prediction exchange contract."""

from typing import List, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr, model_validator

from serving.prediction_schema import validate_request


Identifier = Union[StrictInt, StrictStr]
FiniteFloat = float


class StrictContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PredictionRequest(StrictContractModel):
    stay_id: Identifier
    prediction_time: StrictStr

    @model_validator(mode="after")
    def enforce_phase5_contract(self):
        validate_request(self.model_dump())
        return self


class ModelVersion(StrictContractModel):
    family: Literal["xgboost", "gru"]
    model_version: StrictStr
    artifact_sha256: StrictStr
    preprocessor_sha256: StrictStr


class SupportModelVersion(ModelVersion):
    calibrator_sha256: StrictStr


class ModelVersions(StrictContractModel):
    recovery: ModelVersion
    icu_stay_time: ModelVersion
    organ_support: SupportModelVersion


class RecoveryResponse(StrictContractModel):
    delta_24h: FiniteFloat = Field(allow_inf_nan=False)
    delta_48h: FiniteFloat = Field(allow_inf_nan=False)
    reconstructed_sofa_24h: FiniteFloat = Field(allow_inf_nan=False)
    reconstructed_sofa_48h: FiniteFloat = Field(allow_inf_nan=False)


class ExplanationItem(StrictContractModel):
    feature_name: StrictStr
    attribution: FiniteFloat = Field(allow_inf_nan=False)


class TaskExplanation(StrictContractModel):
    family: Literal["xgboost", "gru"]
    explanation_method: Literal["tree_shap", "integrated_gradients"]
    items: List[ExplanationItem]


class TaskExplanations(StrictContractModel):
    recovery: TaskExplanation
    icu_stay_time: TaskExplanation
    organ_support: TaskExplanation


class DataQuality(StrictContractModel):
    total_bins: int = Field(ge=0)
    observed_bins: int = Field(ge=0)
    padding_bins: int = Field(ge=0)
    total_feature_values: int = Field(ge=0)
    observed_feature_values: int = Field(ge=0)
    missing_feature_values: int = Field(ge=0)


class TaskVersionMap(StrictContractModel):
    recovery: StrictStr
    icu_stay_time: StrictStr
    organ_support: StrictStr


class PredictionModelMetadata(StrictContractModel):
    feature_version: TaskVersionMap
    label_version: TaskVersionMap
    split_version: StrictStr
    manifest_version: StrictStr


class PredictionResponse(StrictContractModel):
    schema_version: Literal["prediction_schema_v1"]
    prediction_time: StrictStr
    mode: Literal["RETROSPECTIVE_SEQUENTIAL_REPLAY"]
    model_versions: ModelVersions
    recovery: RecoveryResponse
    icu_stay_time_hours: FiniteFloat = Field(ge=0, allow_inf_nan=False)
    organ_support_probability_calibrated: FiniteFloat = Field(
        ge=0, le=1, allow_inf_nan=False
    )
    explanation_features: TaskExplanations
    data_quality: DataQuality
    model_metadata: PredictionModelMetadata


class ErrorDetail(StrictContractModel):
    code: StrictStr
    message: StrictStr


class ErrorResponse(StrictContractModel):
    error: ErrorDetail


class HealthResponse(StrictContractModel):
    status: Literal["alive"]
    ready: bool
    serving_scope: Literal["real", "synthetic", "unavailable"]
    mode: Literal["RETROSPECTIVE_SEQUENTIAL_REPLAY"]


class TaskServingMetadata(StrictContractModel):
    family: Literal["xgboost", "gru"]
    model_version: StrictStr
    artifact_sha256: StrictStr
    preprocessor_sha256: StrictStr
    feature_version: StrictStr
    label_version: StrictStr
    split_hash: StrictStr
    explanation_method: Literal["tree_shap", "integrated_gradients"]


class TaskServingMetadataMap(StrictContractModel):
    recovery: TaskServingMetadata
    icu_stay_time: TaskServingMetadata
    organ_support: TaskServingMetadata


class SupportServingMetadata(StrictContractModel):
    calibrator_sha256: StrictStr
    threshold_sha256: StrictStr
    threshold: FiniteFloat = Field(ge=0, le=1, allow_inf_nan=False)
    comparator: Literal["greater_than_or_equal"]


class ModelMetadataResponse(StrictContractModel):
    mode: Literal["RETROSPECTIVE_SEQUENTIAL_REPLAY"]
    serving_scope: Literal["real", "synthetic"]
    manifest_version: StrictStr
    manifest_sha256: StrictStr
    split_version: StrictStr
    tasks: TaskServingMetadataMap
    organ_support: SupportServingMetadata
