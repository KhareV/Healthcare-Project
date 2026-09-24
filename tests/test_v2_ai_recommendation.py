"""Focused tests for the V2 AI-recommendation feature (Groq/Llama-family
synthesis) and its supporting API endpoints. Deterministic and network-free:
external calls are exercised only through graceful-degradation paths or a
monkeypatched transport, never a live network request, so this suite never
depends on external reachability or API quota."""

import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

SAMPLE_PREDICTION = {
    "prediction_time": "2101-05-31T06:17:26Z",
    "elapsed_icu_hours": 24,
    "current_sofa": 1.0,
    "recovery": {"sofa_hat_24h": 1.3, "delta_24h": 0.3, "sofa_hat_48h": 3.2, "delta_48h": 2.2},
    "icu_stay_time": {"remaining_hours": 34.4},
    "organ_support": {"probability_24h": 0.42, "raw_probability": 0.39, "threshold": 0.398, "alert": True},
    "explanations": {
        "recovery24": {"top_positive_contributors": [{"label": "Current SOFA", "attribution": 0.5}], "top_negative_contributors": [{"label": "heart rate", "attribution": -0.1}]},
        "recovery48": {"top_positive_contributors": [], "top_negative_contributors": []},
        "icu_stay_time": {"top_positive_contributors": [], "top_negative_contributors": []},
        "organ_support": {"top_positive_contributors": [], "top_negative_contributors": []},
    },
    "data_quality": {"observed_feature_fraction": 0.78},
}


def test_build_user_prompt_contains_key_numbers():
    from serving.v2.ai_recommendation import build_user_prompt

    prompt = build_user_prompt(prediction=SAMPLE_PREDICTION)
    assert "1.0" in prompt
    assert "1.3" in prompt
    assert "34.4" in prompt
    assert "42.0" in prompt or "42.0 %" in prompt
    assert "ABOVE" in prompt
    assert "Current SOFA (+0.500)" in prompt


def test_system_prompt_forbids_clinical_recommendation_language():
    from serving.v2.ai_recommendation import SYSTEM_PROMPT

    lowered = SYSTEM_PROMPT.lower()
    assert "never diagnose" in lowered
    assert "never recommend" in lowered or "recommend or imply a" in lowered
    assert "not a real patient" in lowered
    assert "not a real-time clinical tool" in lowered


def test_generate_recommendation_degrades_gracefully_without_key():
    from serving.v2.ai_recommendation import generate_recommendation

    result = generate_recommendation(prediction=SAMPLE_PREDICTION, api_key=None)
    assert result.status == "UNAVAILABLE"
    assert result.summary is None
    assert result.error is not None
    assert "GROQ_API_KEY" in result.error
    assert result.disclaimer


def test_generate_recommendation_degrades_gracefully_on_transport_error(monkeypatch):
    from serving.v2 import ai_recommendation as module

    def _raise(*args, **kwargs):
        raise ConnectionError("simulated network failure")

    monkeypatch.setattr(module.httpx, "post", _raise)
    result = module.generate_recommendation(prediction=SAMPLE_PREDICTION, api_key="fake-key-for-test")
    assert result.status == "UNAVAILABLE"
    assert result.summary is None
    assert "ConnectionError" in result.error


def test_generate_recommendation_succeeds_with_mocked_transport(monkeypatch):
    from serving.v2 import ai_recommendation as module

    class _FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> Any:
            return {"choices": [{"message": {"content": "A synthetic benchmark research note."}}]}

    def _fake_post(url, *, headers, json, timeout):  # noqa: ANN001
        assert url == module.GROQ_CHAT_COMPLETIONS_URL
        assert headers["Authorization"] == "Bearer fake-key-for-test"
        assert json["model"] == module.DEFAULT_MODEL
        assert json["messages"][0]["role"] == "system"
        return _FakeResponse()

    monkeypatch.setattr(module.httpx, "post", _fake_post)
    result = module.generate_recommendation(prediction=SAMPLE_PREDICTION, api_key="fake-key-for-test")
    assert result.status == "OK"
    assert result.summary == "A synthetic benchmark research note."
    assert result.error is None


def test_generate_recommendation_handles_empty_content(monkeypatch):
    from serving.v2 import ai_recommendation as module

    class _FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> Any:
            return {"choices": [{"message": {"content": "", "reasoning": "hidden chain of thought"}}]}

    monkeypatch.setattr(module.httpx, "post", lambda *a, **k: _FakeResponse())
    result = module.generate_recommendation(prediction=SAMPLE_PREDICTION, api_key="fake-key-for-test")
    assert result.status == "UNAVAILABLE"
    assert result.error is not None


# --- API endpoints -----------------------------------------------------------

@pytest.fixture(scope="module")
def api_client():
    from fastapi.testclient import TestClient

    from api.v2_app import build_v2_app

    return TestClient(build_v2_app(ROOT))


def test_demo_subjects_endpoint(api_client):
    response = api_client.get("/demo-subjects")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "V2_DEMO_MANIFEST_FROZEN"
    assert len(body["demo_subjects"]) >= 3


def test_performance_endpoint_reads_frozen_artifacts_only(api_client):
    import json as _json

    state_before = _json.loads((ROOT / "artifacts/performance_v2/governance/v2_fresh_test_access_state.json").read_text())
    response = api_client.get("/performance")
    assert response.status_code == 200
    body = response.json()
    assert "final_metrics" in body and "bootstrap" in body and "naive_comparison" in body
    assert body["final_metrics"]["recovery24"]["mae"] == pytest.approx(1.0744128806773996)
    state_after = _json.loads((ROOT / "artifacts/performance_v2/governance/v2_fresh_test_access_state.json").read_text())
    assert state_before == state_after


def test_history_endpoint_returns_only_events_at_or_before_cutoff(api_client):
    response = api_client.get("/history", params={"stay_id": "SYN-E-00000001", "prediction_time": "2101-05-31T06:17:26Z"})
    assert response.status_code == 200
    body = response.json()
    assert body["count"] == len(body["events"])
    assert body["count"] > 0
    for event in body["events"]:
        assert event["event_time"] <= "2101-05-31T06:17:26.999999Z"


def test_history_endpoint_rejects_fresh_test_and_unknown_stays(api_client):
    for stay_id in ("SYN-V2-E-00000001", "SYN-E-99999999"):
        response = api_client.get("/history", params={"stay_id": stay_id, "prediction_time": "2101-01-01T00:00:00Z"})
        assert response.status_code == 404


def test_history_endpoint_rejects_illegal_cutoff(api_client):
    response = api_client.get("/history", params={"stay_id": "SYN-E-00000001", "prediction_time": "2099-01-01T00:00:00Z"})
    assert response.status_code == 422


def test_ai_recommendation_endpoint_never_crashes_and_returns_structured_result(api_client, monkeypatch):
    """Exercises the real endpoint wiring end-to-end (prediction ->
    explanations -> AI synthesis), with the outbound Groq call mocked so the
    test is deterministic and network-free."""

    from serving.v2 import ai_recommendation as module

    class _FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self):
            return {"choices": [{"message": {"content": "Deterministic test summary."}}]}

    monkeypatch.setattr(module.httpx, "post", lambda *a, **k: _FakeResponse())
    monkeypatch.setenv("GROQ_API_KEY", "fake-key-for-test")

    response = api_client.post("/ai/recommendation", json={"stay_id": "SYN-E-00000001", "prediction_time": "2101-05-31T06:17:26Z"})
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "OK"
    assert body["summary"] == "Deterministic test summary."
    assert "not clinical advice" in body["disclaimer"].lower()


def test_ai_recommendation_rejects_fresh_test_and_unknown_stays(api_client):
    for stay_id in ("SYN-V2-E-00000001", "SYN-E-99999999"):
        response = api_client.post("/ai/recommendation", json={"stay_id": stay_id, "prediction_time": "2101-01-01T00:00:00Z"})
        assert response.status_code == 404


def test_cors_middleware_present(api_client):
    response = api_client.options(
        "/predict",
        headers={"Origin": "http://localhost:5173", "Access-Control-Request-Method": "POST"},
    )
    assert response.status_code in (200, 204)
    assert response.headers.get("access-control-allow-origin") == "*"


def test_no_test_run_touched_fresh_test_access_state(api_client):
    import json as _json

    state = _json.loads((ROOT / "artifacts/performance_v2/governance/v2_fresh_test_access_state.json").read_text())
    assert state["status"] == "FINAL_V2_RUN_COMPLETED"
    assert len(state["history"]) == 3
