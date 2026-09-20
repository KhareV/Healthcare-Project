import httpx
import pytest

from dashboard.api_client import DashboardAPIError, HTTPDashboardAPIClient
from tests.dashboard_helpers import metadata, response_for


def _response(status, payload, request):
    return httpx.Response(status, json=payload, request=request)


def test_http_client_uses_only_serving_endpoints(monkeypatch):
    calls = []

    def request(method, url, **kwargs):
        calls.append((method, url, kwargs))
        req = httpx.Request(method, url)
        if url.endswith("/model-metadata"):
            return _response(200, metadata(), req)
        return _response(200, response_for(kwargs["json"]["prediction_time"]), req)

    monkeypatch.setattr(httpx, "request", request)
    client = HTTPDashboardAPIClient("http://api.test/", synthetic=True)
    client.get_model_metadata()
    client.predict("SYNTHETIC_PHASE13_STAY_A", "2026-02-02T00:00:00+00:00")
    assert [(item[0], item[1]) for item in calls] == [
        ("GET", "http://api.test/model-metadata"),
        ("POST", "http://api.test/predict"),
    ]
    assert calls[1][2]["json"]["stay_id"] == "SYNTHETIC_PHASE13_STAY_A"


@pytest.mark.parametrize("status", (404, 422, 503))
def test_api_errors_are_safe(monkeypatch, status):
    def request(method, url, **kwargs):
        return _response(status, {"error": {"code": "unavailable", "message": "safe"}}, httpx.Request(method, url))

    monkeypatch.setattr(httpx, "request", request)
    with pytest.raises(DashboardAPIError) as captured:
        HTTPDashboardAPIClient("http://api.test", synthetic=True).get_model_metadata()
    assert captured.value.status_code == status
    assert captured.value.safe_message == "safe"


def test_network_errors_are_normalized(monkeypatch):
    def request(method, url, **kwargs):
        raise httpx.ConnectError("secret transport detail")

    monkeypatch.setattr(httpx, "request", request)
    with pytest.raises(DashboardAPIError, match="serving API is unavailable"):
        HTTPDashboardAPIClient("http://api.test", synthetic=True).get_model_metadata()


def test_malformed_success_response_fails_client_contract(monkeypatch):
    def request(method, url, **kwargs):
        return _response(200, {"unexpected": True}, httpx.Request(method, url))

    monkeypatch.setattr(httpx, "request", request)
    with pytest.raises(DashboardAPIError, match="metadata is incompatible"):
        HTTPDashboardAPIClient("http://api.test", synthetic=True).get_model_metadata()
