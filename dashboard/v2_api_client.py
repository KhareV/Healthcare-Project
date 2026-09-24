"""Thin HTTP client the V2 dashboard uses to call the V2 API. Mirrors
dashboard/api_client.py's pattern: genuine HTTP calls, no precomputed
lookup table, no local prediction logic."""

from __future__ import annotations

from typing import Mapping, Optional, Protocol

import httpx


class V2DashboardAPI(Protocol):
    def predict(self, stay_id: str, prediction_time: str) -> Mapping[str, object]: ...

    def health(self) -> Mapping[str, object]: ...


class HTTPV2DashboardAPIClient:
    def __init__(self, base_url: str, *, timeout_seconds: float = 8.0, client: Optional[httpx.Client] = None):
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout_seconds
        self._client = client

    def predict(self, stay_id: str, prediction_time: str) -> Mapping[str, object]:
        if self._client is not None:
            response = self._client.post(f"{self._base_url}/predict", json={"stay_id": stay_id, "prediction_time": prediction_time})
        else:
            response = httpx.post(f"{self._base_url}/predict", json={"stay_id": stay_id, "prediction_time": prediction_time}, timeout=self._timeout)
        response.raise_for_status()
        return response.json()

    def health(self) -> Mapping[str, object]:
        if self._client is not None:
            response = self._client.get(f"{self._base_url}/health")
        else:
            response = httpx.get(f"{self._base_url}/health", timeout=self._timeout)
        response.raise_for_status()
        return response.json()


class InProcessV2DashboardAPIClient:
    """Wraps a starlette TestClient bound to the real V2 FastAPI app -- a
    genuine request/response cycle through the same app, without requiring
    a separately running uvicorn process. Used for tests/evidence capture."""

    def __init__(self, test_client):
        self._client = test_client

    def predict(self, stay_id: str, prediction_time: str) -> Mapping[str, object]:
        response = self._client.post("/predict", json={"stay_id": stay_id, "prediction_time": prediction_time})
        response.raise_for_status()
        return response.json()

    def health(self) -> Mapping[str, object]:
        response = self._client.get("/health")
        response.raise_for_status()
        return response.json()
