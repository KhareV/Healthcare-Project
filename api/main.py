"""FastAPI facade for retrospective sequential research replay.

This module contains no feature engineering, model loading, selection,
calibration, attribution, or recovery reconstruction logic.
"""

import logging
from typing import Mapping, Optional, Protocol

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from api.schemas import (
    ErrorResponse,
    HealthResponse,
    ModelMetadataResponse,
    PredictionRequest,
    PredictionResponse,
)
from evaluation.selection_validation import EXPLANATION_METHOD, TASKS
from explainability.router import ExplanationRouterError
from serving.artifacts import ServingArtifactError
from serving.history import HistoryContractError, IllegalCutoffError, UnknownStayError
from serving.pipeline import PredictionPipelineError
from serving.postprocessing import ServingPostprocessError
from serving.preprocessing import ServingFeatureContractError
from serving.recovery import RecoveryServingError
from serving.prediction_schema import (
    PredictionSchemaError,
    REPLAY_MODE,
    validate_exchange,
)


LOGGER = logging.getLogger(__name__)


class MetadataCompatibilityError(RuntimeError):
    """The reported metadata is not bound to the active serving bundle."""


class PipelineLike(Protocol):
    @property
    def bundle(self):
        """Return the validated immutable serving bundle."""

    def predict(self, request: object) -> Mapping[str, object]:
        """Run the canonical prediction pipeline once."""


class MetadataProvider(Protocol):
    def get_metadata(self, pipeline: PipelineLike) -> Mapping[str, object]:
        """Return metadata for the exact active serving bundle."""


class BundleMetadataProvider:
    """Serialize identities from the already validated serving bundle."""

    def get_metadata(self, pipeline: PipelineLike) -> Mapping[str, object]:
        bundle = pipeline.bundle
        task_metadata = {}
        for task in TASKS:
            selected = bundle.task(task)
            identity = selected.identity
            if (
                getattr(selected.predictor, "task", None) != task
                or getattr(selected.predictor, "family", None) != identity.family
                or getattr(selected.predictor, "model_version", None)
                != identity.model_version
                or getattr(selected.predictor, "artifact_sha256", None)
                != identity.artifact_sha256
                or getattr(selected.preprocessor, "task", None) != task
                or getattr(selected.preprocessor, "artifact_sha256", None)
                != identity.preprocessor_sha256
                or identity.explanation_method != EXPLANATION_METHOD[identity.family]
            ):
                raise MetadataCompatibilityError(
                    "active serving metadata is incompatible"
                )
            task_metadata[task] = {
                "family": identity.family,
                "model_version": identity.model_version,
                "artifact_sha256": identity.artifact_sha256,
                "preprocessor_sha256": identity.preprocessor_sha256,
                "feature_version": identity.feature_version,
                "label_version": identity.label_version,
                "split_hash": identity.split_hash,
                "explanation_method": identity.explanation_method,
            }
        return {
            "mode": REPLAY_MODE,
            "serving_scope": bundle.scope,
            "manifest_version": bundle.manifest_version,
            "manifest_sha256": bundle.manifest_file_sha256,
            "split_version": bundle.split_version,
            "tasks": task_metadata,
            "organ_support": {
                "calibrator_sha256": bundle.support_calibrator_sha256,
                "threshold_sha256": bundle.support_threshold_sha256,
                "threshold": bundle.support_threshold,
                "comparator": "greater_than_or_equal",
            },
        }


def _error(status_code: int, code: str, message: str) -> JSONResponse:
    payload = ErrorResponse(error={"code": code, "message": message})
    return JSONResponse(status_code=status_code, content=payload.model_dump())


def _validated_metadata(
    pipeline: PipelineLike, provider: MetadataProvider
) -> ModelMetadataResponse:
    try:
        value = ModelMetadataResponse.model_validate(provider.get_metadata(pipeline))
    except ValidationError as error:
        raise MetadataCompatibilityError(
            "serving metadata does not satisfy its contract"
        ) from error
    bundle = pipeline.bundle
    if (
        value.manifest_version != bundle.manifest_version
        or value.manifest_sha256 != bundle.manifest_file_sha256
        or value.split_version != bundle.split_version
        or value.serving_scope != bundle.scope
    ):
        raise MetadataCompatibilityError(
            "serving metadata is bound to a different manifest"
        )
    for task in TASKS:
        identity = bundle.task(task).identity
        reported = getattr(value.tasks, task)
        expected = (
            identity.family,
            identity.model_version,
            identity.artifact_sha256,
            identity.preprocessor_sha256,
            identity.feature_version,
            identity.label_version,
            identity.split_hash,
            identity.explanation_method,
        )
        actual = (
            reported.family,
            reported.model_version,
            reported.artifact_sha256,
            reported.preprocessor_sha256,
            reported.feature_version,
            reported.label_version,
            reported.split_hash,
            reported.explanation_method,
        )
        if actual != expected:
            raise MetadataCompatibilityError(
                "serving task metadata is incompatible"
            )
    if (
        value.organ_support.calibrator_sha256
        != bundle.support_calibrator_sha256
        or value.organ_support.threshold_sha256
        != bundle.support_threshold_sha256
        or value.organ_support.threshold != bundle.support_threshold
    ):
        raise MetadataCompatibilityError(
            "support serving metadata is incompatible"
        )
    return value


def _assert_prediction_metadata(
    response: Mapping[str, object], metadata: ModelMetadataResponse
) -> None:
    reported = response.get("model_metadata")
    versions = response.get("model_versions")
    if not isinstance(reported, Mapping) or not isinstance(versions, Mapping):
        raise MetadataCompatibilityError("prediction metadata is absent")
    if (
        reported.get("manifest_version") != metadata.manifest_version
        or reported.get("split_version") != metadata.split_version
    ):
        raise MetadataCompatibilityError(
            "prediction response uses stale manifest metadata"
        )
    feature_versions = reported.get("feature_version")
    label_versions = reported.get("label_version")
    if not isinstance(feature_versions, Mapping) or not isinstance(
        label_versions, Mapping
    ):
        raise MetadataCompatibilityError("prediction version metadata is absent")
    for task in TASKS:
        task_version = versions.get(task)
        task_metadata = getattr(metadata.tasks, task)
        if not isinstance(task_version, Mapping) or (
            task_version.get("family") != task_metadata.family
            or task_version.get("model_version") != task_metadata.model_version
            or task_version.get("artifact_sha256") != task_metadata.artifact_sha256
            or task_version.get("preprocessor_sha256")
            != task_metadata.preprocessor_sha256
            or feature_versions.get(task) != task_metadata.feature_version
            or label_versions.get(task) != task_metadata.label_version
        ):
            raise MetadataCompatibilityError(
                "prediction task metadata is incompatible"
            )


def create_app(
    pipeline: Optional[PipelineLike] = None,
    *,
    metadata_provider: Optional[MetadataProvider] = None,
) -> FastAPI:
    """Construct an app without loading artifacts or patient data at import."""

    provider = metadata_provider or BundleMetadataProvider()
    application = FastAPI(
        title="Personalized Patient Recovery Trajectory Research Replay",
        description=(
            "Local retrospective sequential-replay research prototype. "
            "Not real-time clinical inference or bedside decision support."
        ),
        version="prediction_schema_v1",
    )

    @application.exception_handler(RequestValidationError)
    async def request_validation_handler(
        _request: Request, _error_value: RequestValidationError
    ) -> JSONResponse:
        return _error(422, "request_validation_error", "request is invalid")

    @application.exception_handler(UnknownStayError)
    async def unknown_stay_handler(
        _request: Request, _error_value: UnknownStayError
    ) -> JSONResponse:
        return _error(404, "unknown_stay", "requested stay is unavailable")

    @application.exception_handler(IllegalCutoffError)
    async def illegal_cutoff_handler(
        _request: Request, _error_value: IllegalCutoffError
    ) -> JSONResponse:
        return _error(
            422,
            "illegal_prediction_time",
            "prediction_time is not a legal cutoff for the requested stay",
        )

    availability_errors = (
        ServingArtifactError,
        PredictionPipelineError,
        PredictionSchemaError,
        ExplanationRouterError,
        ServingPostprocessError,
        RecoveryServingError,
        ServingFeatureContractError,
        HistoryContractError,
        MetadataCompatibilityError,
    )

    async def availability_handler(
        _request: Request, _error_value: Exception
    ) -> JSONResponse:
        return _error(
            503,
            "serving_unavailable",
            "validated serving dependencies are unavailable",
        )

    for error_type in availability_errors:
        application.add_exception_handler(error_type, availability_handler)

    @application.exception_handler(Exception)
    async def unexpected_handler(
        _request: Request, error_value: Exception
    ) -> JSONResponse:
        LOGGER.error("unexpected API failure type=%s", type(error_value).__name__)
        return _error(500, "internal_error", "internal server error")

    @application.get("/health", response_model=HealthResponse)
    def health() -> Mapping[str, object]:
        if pipeline is None:
            return {
                "status": "alive",
                "ready": False,
                "serving_scope": "unavailable",
                "mode": REPLAY_MODE,
            }
        _validated_metadata(pipeline, provider)
        return {
            "status": "alive",
            "ready": pipeline.bundle.scope == "real",
            "serving_scope": pipeline.bundle.scope,
            "mode": REPLAY_MODE,
        }

    @application.get(
        "/model-metadata",
        response_model=ModelMetadataResponse,
        responses={503: {"model": ErrorResponse}},
    )
    def model_metadata() -> ModelMetadataResponse:
        if pipeline is None:
            raise MetadataCompatibilityError("real serving bundle is unavailable")
        return _validated_metadata(pipeline, provider)

    @application.post(
        "/predict",
        response_model=PredictionResponse,
        responses={404: {"model": ErrorResponse}, 422: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
    )
    def predict(request: PredictionRequest) -> Mapping[str, object]:
        if pipeline is None:
            raise MetadataCompatibilityError("real serving bundle is unavailable")
        metadata = _validated_metadata(pipeline, provider)
        request_payload = request.model_dump()
        response = pipeline.predict(request_payload)
        validate_exchange(
            request_payload,
            response,
            synthetic=pipeline.bundle.scope == "synthetic",
        )
        _assert_prediction_metadata(response, metadata)
        return response

    return application


# Production artifacts are deliberately not loaded at import. Until an
# approved composition root supplies the real validated bundle, readiness is
# false and prediction/metadata fail with 503 rather than using test fixtures.
app = create_app()
