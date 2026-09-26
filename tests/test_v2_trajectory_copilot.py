"""Focused tests for the Trajectory Copilot grounded Q&A layer and its
supporting API endpoint (/assistant) plus the demo-patient alias artifact.

Deterministic and network-free, mirroring tests/test_v2_ai_recommendation.py:
external calls are exercised only through graceful-degradation paths or a
monkeypatched transport, never a live network request."""

import json
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

SAMPLE_CONTEXT = {
    "patient_alias": "DEMO-CARDIAC-003",
    "prediction_time": "2103-01-05T08:02:18Z",
    "episode": {"elapsed_hours": 30, "cardiac_subtype": "ARRHYTHMIA", "age_years": 26, "sex": "FEMALE"},
    "current_state": {"current_sofa": 1.0},
    "forecasts": {
        "delta_sofa_24": 0.05,
        "delta_sofa_48": 0.5,
        "predicted_sofa_24h": 1.05,
        "predicted_sofa_48h": 1.5,
        "remaining_icu_hours": 35.2,
        "support_raw_probability": 0.30,
        "support_calibrated_probability": 0.249,
        "support_threshold": 0.398,
        "support_alert": False,
    },
    "top_contributors": {
        "recovery24": [{"label": "Current SOFA", "attribution": 0.5}],
        "recovery48": [],
        "icu": [],
        "support": [],
    },
    "data_quality": {"observed_feature_fraction": 0.4},
    "system_limitations": ["synthetic research benchmark", "retrospective replay, not real-time"],
}

SAMPLE_PREVIOUS = {
    "prediction_time": "2103-01-05T02:02:18Z",
    "current_sofa": 1.0,
    "predicted_sofa_24h": 1.10,
    "predicted_sofa_48h": 1.26,
    "remaining_icu_hours": 33.8,
    "support_probability": 0.249,
    "support_alert": False,
}


def test_build_assistant_user_prompt_grounds_alias_and_numbers():
    from serving.v2.ai_recommendation import build_assistant_user_prompt

    prompt = build_assistant_user_prompt(context=SAMPLE_CONTEXT)
    assert "DEMO-CARDIAC-003" in prompt
    assert "ARRHYTHMIA" in prompt
    assert "1.05" in prompt
    assert "35.2" in prompt
    assert "first replay" in prompt.lower() or "no previous replay" in prompt.lower()


def test_build_assistant_user_prompt_computes_deltas_against_previous_cutoff():
    from serving.v2.ai_recommendation import build_assistant_user_prompt

    context = {**SAMPLE_CONTEXT, "previous_cutoff": SAMPLE_PREVIOUS}
    prompt = build_assistant_user_prompt(context=context, question="What changed?")
    assert "What changed?" in prompt
    assert "Predicted SOFA +24h" in prompt
    # remaining-ICU-hours values are exact at one-decimal precision: 33.8 -> 35.2 is an increase
    assert "33.8" in prompt and "35.2" in prompt and "up" in prompt


def test_trajectory_copilot_prompt_forbids_treatment_and_diagnosis_language():
    from serving.v2.ai_recommendation import load_trajectory_copilot_prompt

    lowered = " ".join(load_trajectory_copilot_prompt().lower().split())
    assert "never" in lowered
    assert "prescribe" in lowered or "medication" in lowered
    assert "treatment recommendation" in lowered
    assert "synthetic" in lowered and "retrospective" in lowered
    assert "does not provide treatment recommendations" in lowered


def test_generate_assistant_response_degrades_gracefully_without_key():
    from serving.v2.ai_recommendation import generate_assistant_response

    result = generate_assistant_response(context=SAMPLE_CONTEXT, api_key=None)
    assert result.status == "UNAVAILABLE"
    assert result.error is not None


def test_generate_assistant_response_succeeds_with_mocked_transport(monkeypatch):
    from serving.v2 import ai_recommendation as module

    captured = {}

    class _FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> Any:
            return {"choices": [{"message": {"content": "Deterministic copilot answer."}}]}

    def _fake_post(url, headers=None, json=None, timeout=None):  # noqa: A002 - matches httpx.post signature
        captured["payload"] = json
        return _FakeResponse()

    monkeypatch.setattr(module.httpx, "post", _fake_post)
    result = module.generate_assistant_response(context=SAMPLE_CONTEXT, question="Summarize.", api_key="fake-key-for-test")
    assert result.status == "OK"
    assert result.summary == "Deterministic copilot answer."
    assert captured["payload"]["messages"][0]["role"] == "system"
    assert "DEMO-CARDIAC-003" in captured["payload"]["messages"][1]["content"]


# --- demo-patient alias artifact ---------------------------------------------

def test_demo_patients_product_artifact_is_structural_and_covers_manifest():
    artifact_path = ROOT / "artifacts/performance_v2/product/demo_patients_v1.json"
    assert artifact_path.exists(), "run scripts/product_v2_build_demo_patients.py"
    payload = json.loads(artifact_path.read_text())
    assert payload["status"] == "V2_PRODUCT_DEMO_PATIENTS_READY"
    assert payload["selection_independent_of_model_performance"] is True

    manifest = json.loads((ROOT / "configs/performance_v2/v2_demo_manifest_v1.json").read_text())
    manifest_stays = {s["stay_id"] for s in manifest["demo_subjects"]}
    artifact_stays = {p["stay_id"] for p in payload["demo_patients"]}
    assert artifact_stays == manifest_stays

    aliases = [p["patient_alias"] for p in payload["demo_patients"]]
    assert len(aliases) == len(set(aliases)), "aliases must be unique"
    assert all(alias.startswith("DEMO-CARDIAC-") for alias in aliases)


# --- API endpoint -------------------------------------------------------------

@pytest.fixture(scope="module")
def api_client():
    from fastapi.testclient import TestClient

    from api.v2_app import build_v2_app

    return TestClient(build_v2_app(ROOT))


def test_demo_subjects_endpoint_includes_patient_alias(api_client):
    response = api_client.get("/demo-subjects")
    assert response.status_code == 200
    body = response.json()
    assert all("patient_alias" in subject for subject in body["demo_subjects"])
    assert all(subject["patient_alias"].startswith("DEMO-CARDIAC-") for subject in body["demo_subjects"])


def test_assistant_endpoint_default_summary(api_client, monkeypatch):
    from serving.v2 import ai_recommendation as module

    class _FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self):
            return {"choices": [{"message": {"content": "Deterministic default summary."}}]}

    monkeypatch.setattr(module.httpx, "post", lambda *a, **k: _FakeResponse())
    monkeypatch.setenv("GROQ_API_KEY", "fake-key-for-test")

    response = api_client.post("/assistant", json={"stay_id": "SYN-E-00000001", "prediction_time": "2101-05-31T06:17:26Z"})
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "OK"
    assert body["summary"] == "Deterministic default summary."


def test_assistant_endpoint_grounds_previous_cutoff_deltas(api_client, monkeypatch):
    from serving.v2 import ai_recommendation as module

    captured = {}

    class _FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self):
            return {"choices": [{"message": {"content": "ok"}}]}

    def _fake_post(url, headers=None, json=None, timeout=None):  # noqa: A002
        captured["payload"] = json
        return _FakeResponse()

    monkeypatch.setattr(module.httpx, "post", _fake_post)
    monkeypatch.setenv("GROQ_API_KEY", "fake-key-for-test")

    manifest = json.loads((ROOT / "configs/performance_v2/v2_demo_manifest_v1.json").read_text())
    subject = next(s for s in manifest["demo_subjects"] if s["stay_id"] == "SYN-E-00000001")
    cutoffs = subject["legal_cutoffs"]

    response = api_client.post(
        "/assistant",
        json={
            "stay_id": subject["stay_id"],
            "prediction_time": cutoffs[1],
            "previous_prediction_time": cutoffs[0],
            "question": "What changed since the previous cutoff?",
        },
    )
    assert response.status_code == 200
    user_message = captured["payload"]["messages"][1]["content"]
    assert "What changed since the previous cutoff?" in user_message
    assert "No previous replay cutoff" not in user_message  # a real previous cutoff was supplied


def test_assistant_endpoint_rejects_fresh_test_and_unknown_stays(api_client):
    for stay_id in ("SYN-V2-E-00000001", "SYN-E-99999999"):
        response = api_client.post("/assistant", json={"stay_id": stay_id, "prediction_time": "2101-01-01T00:00:00Z"})
        assert response.status_code == 404


def test_assistant_endpoint_ignores_invalid_previous_cutoff_rather_than_failing(api_client, monkeypatch):
    from serving.v2 import ai_recommendation as module

    monkeypatch.setattr(module.httpx, "post", lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not be called without a key")))
    response = api_client.post(
        "/assistant",
        json={"stay_id": "SYN-E-00000001", "prediction_time": "2101-05-31T06:17:26Z", "previous_prediction_time": "2099-01-01T00:00:00Z"},
    )
    assert response.status_code == 200
    assert response.json()["status"] == "UNAVAILABLE"  # no GROQ_API_KEY set in this test


def test_no_test_run_touched_fresh_test_access_state(api_client):
    state = json.loads((ROOT / "artifacts/performance_v2/governance/v2_fresh_test_access_state.json").read_text())
    assert state["status"] == "FINAL_V2_RUN_COMPLETED"
    assert len(state["history"]) == 3
