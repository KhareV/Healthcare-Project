"""Stage-4 real dashboard: real catalog + live API, retrospective replay only.

A genuine uvicorn server is started in a background thread so the dashboard
exercises its actual HTTP client path (``dashboard.api_client.HTTPDashboardAPIClient``)
against the real Stage-4 API, proving the replay panel is bound to live
recomputation rather than any precomputed lookup table.
"""

import socket
import threading
import time

import httpx
import pytest
import uvicorn

from api.main import create_app
from dashboard.app import create_real_dashboard_app
from dashboard.real_catalog import build_real_dashboard_catalog
from serving.real.bundle import resolve_stage4_bundle
from stage4_helpers import ROOT


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture(scope="module")
def live_api_url():
    resolution = resolve_stage4_bundle(ROOT)
    app = create_app(pipeline=resolution.pipeline)
    port = _free_port()
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 10
    url = "http://127.0.0.1:{}".format(port)
    while time.time() < deadline:
        try:
            httpx.get(url + "/health", timeout=1.0)
            break
        except httpx.HTTPError:
            time.sleep(0.1)
    else:
        raise RuntimeError("live Stage-4 API did not start in time")
    yield url, resolution
    server.should_exit = True
    thread.join(timeout=5)


@pytest.fixture(scope="module")
def real_catalog(live_api_url):
    _url, resolution = live_api_url
    return build_real_dashboard_catalog(
        ROOT,
        current_sofa_provider=resolution.current_sofa_provider,
        split_by_stay=resolution.split_by_stay,
        limit=3,
    )


def test_real_catalog_excludes_sealed_test_subjects(real_catalog, live_api_url):
    _url, resolution = live_api_url
    for stay_id in real_catalog.stay_ids:
        assert resolution.split_by_stay[stay_id] != "test"


def test_dashboard_replay_calls_live_api_not_a_lookup_table(live_api_url, real_catalog):
    url, _resolution = live_api_url
    app = create_real_dashboard_app(url, catalog=real_catalog)
    stay_id = real_catalog.stay_ids[0]
    cutoff = real_catalog.stay(stay_id).legal_cutoffs[0]

    from starlette.testclient import TestClient

    client = TestClient(app)
    response = client.get("/", params={"stay_id": stay_id, "prediction_time": cutoff})
    assert response.status_code == 200
    body = response.text
    assert "RETROSPECTIVE SEQUENTIAL REPLAY" in body
    assert stay_id in body


def test_dashboard_matches_direct_api_prediction(live_api_url, real_catalog):
    url, resolution = live_api_url
    stay_id = real_catalog.stay_ids[0]
    cutoff = real_catalog.stay(stay_id).legal_cutoffs[0]
    direct = resolution.pipeline.predict({"stay_id": stay_id, "prediction_time": cutoff})
    api_response = httpx.post(url + "/predict", json={"stay_id": stay_id, "prediction_time": cutoff}, timeout=30.0).json()
    assert direct == api_response


def test_dashboard_refuses_sealed_stay_end_to_end(live_api_url):
    from stage4_helpers import sealed_test_stay, legal_cutoff

    url, _resolution = live_api_url
    stay_id = sealed_test_stay()
    response = httpx.post(url + "/predict", json={"stay_id": stay_id, "prediction_time": legal_cutoff(stay_id)}, timeout=30.0)
    assert response.status_code == 404
