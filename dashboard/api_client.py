"""Narrow HTTP client for the Phase-11 serving API."""

from typing import Mapping, Protocol

import httpx
from pydantic import ValidationError

from api.schemas import HealthResponse, ModelMetadataResponse, PredictionResponse
from serving.prediction_schema import validate_response


class DashboardAPIError(RuntimeError):
    def __init__(self, status_code: int, code: str, safe_message: str) -> None:
        super().__init__(safe_message)
        self.status_code = status_code
        self.code = code
        self.safe_message = safe_message


class DashboardAPI(Protocol):
    def get_health(self) -> Mapping[str, object]:
        """Return validated API health metadata."""

    def get_model_metadata(self) -> Mapping[str, object]:
        """Return validated active-bundle metadata."""

    def predict(self, stay_id: object, prediction_time: str) -> Mapping[str, object]:
        """Request one canonical prediction for one exact replay cutoff."""


class HTTPDashboardAPIClient:
    """No-retry client with a conservative development timeout."""

    def __init__(
        self,
        base_url: str,
        *,
        synthetic: bool,
        timeout_seconds: float = 5.0,
    ) -> None:
        if not isinstance(base_url, str) or not base_url.startswith(("http://", "https://")):
            raise ValueError("dashboard API base URL must be HTTP(S)")
        if timeout_seconds <= 0:
            raise ValueError("dashboard API timeout must be positive")
        self._base_url = base_url.rstrip("/")
        self._synthetic = synthetic
        self._timeout = timeout_seconds

    def _request(self, method: str, path: str, **kwargs) -> Mapping[str, object]:
        try:
            response = httpx.request(
                method,
                self._base_url + path,
                timeout=self._timeout,
                **kwargs,
            )
        except httpx.HTTPError as error:
            raise DashboardAPIError(
                0, "network_unavailable", "serving API is unavailable"
            ) from error
        try:
            payload = response.json()
        except ValueError as error:
            raise DashboardAPIError(
                response.status_code,
                "malformed_response",
                "serving API returned an incompatible response",
            ) from error
        if response.status_code >= 400:
            code = "api_error"
            message = "serving request failed"
            if isinstance(payload, Mapping):
                detail = payload.get("error")
                if isinstance(detail, Mapping):
                    if isinstance(detail.get("code"), str):
                        code = detail["code"]
                    if isinstance(detail.get("message"), str):
                        message = detail["message"]
            raise DashboardAPIError(response.status_code, code, message)
        if not isinstance(payload, Mapping):
            raise DashboardAPIError(
                response.status_code,
                "malformed_response",
                "serving API returned an incompatible response",
            )
        return payload

    def get_health(self) -> Mapping[str, object]:
        payload = self._request("GET", "/health")
        try:
            return HealthResponse.model_validate(payload).model_dump()
        except ValidationError as error:
            raise DashboardAPIError(
                200, "malformed_response", "health response is incompatible"
            ) from error

    def get_model_metadata(self) -> Mapping[str, object]:
        payload = self._request("GET", "/model-metadata")
        try:
            return ModelMetadataResponse.model_validate(payload).model_dump()
        except ValidationError as error:
            raise DashboardAPIError(
                200, "malformed_response", "model metadata is incompatible"
            ) from error

    def predict(self, stay_id: object, prediction_time: str) -> Mapping[str, object]:
        payload = self._request(
            "POST",
            "/predict",
            json={"stay_id": stay_id, "prediction_time": prediction_time},
        )
        try:
            value = PredictionResponse.model_validate(payload).model_dump()
            validate_response(value, synthetic=self._synthetic)
        except (ValidationError, ValueError) as error:
            raise DashboardAPIError(
                200, "malformed_response", "prediction response is incompatible"
            ) from error
        return value
