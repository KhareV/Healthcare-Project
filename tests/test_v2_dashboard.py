"""Focused tests for the V2 dashboard (server-rendered HTML, all five
views). Never mutates fresh-test access state; every panel is populated
through genuine (in-process, real request/response) API calls."""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

FORBIDDEN_TERMS = (
    "discharge eta", "hospital discharge", "recovery time", "survival time",
    "complication risk", "complication probability", "deterioration score", "escalation risk",
    "confidence score", "certainty", "prediction interval",
)
# The data-quality panel's REQUIRED disclaiming sentence must itself name the
# forbidden concepts in order to negate them ("...never a prediction
# confidence, certainty, or quality score."). Strip that one sentence before
# scanning for stray/unguarded usage elsewhere.
DISCLAIMER_SENTENCE = "these are transparent observation counts, never a prediction confidence, certainty, or quality score."
DEMO_STAY = "SYN-E-00000001"


@pytest.fixture(scope="module")
def dashboard_client():
    from fastapi.testclient import TestClient

    from api.v2_app import build_v2_app
    from dashboard.v2_api_client import InProcessV2DashboardAPIClient
    from dashboard.v2_app import build_v2_dashboard_app
    from data.synthetic.validation import load_jsonl

    api_client = InProcessV2DashboardAPIClient(TestClient(build_v2_app(ROOT)))
    demo = json.loads((ROOT / "configs/performance_v2/v2_demo_manifest_v1.json").read_text())
    demo_stays = {item["stay_id"] for item in demo["demo_subjects"]}
    events = load_jsonl(ROOT / "artifacts/data/synthetic/timelines/final/phase9_final_v1/canonical_timeline.jsonl")
    raw_events_by_stay = {}
    for row in events:
        if row["stay_id"] in demo_stays:
            raw_events_by_stay.setdefault(row["stay_id"], []).append(row)
    app = build_v2_dashboard_app(root=ROOT, api_client=api_client, raw_events_by_stay=raw_events_by_stay)
    return TestClient(app)


@pytest.fixture(scope="module")
def first_cutoff():
    demo = json.loads((ROOT / "configs/performance_v2/v2_demo_manifest_v1.json").read_text())
    subject = next(s for s in demo["demo_subjects"] if s["stay_id"] == DEMO_STAY)
    return subject["legal_cutoffs"]


@pytest.mark.parametrize("view", ["replay", "forecast", "performance", "explainability", "dataquality"])
def test_dashboard_smoke_all_views(dashboard_client, first_cutoff, view):
    response = dashboard_client.get("/", params={"view": view, "stay_id": DEMO_STAY, "prediction_time": first_cutoff[-1]})
    assert response.status_code == 200
    text = response.text
    assert "RETROSPECTIVE SEQUENTIAL REPLAY" in text
    assert "SYNTHETIC RESEARCH BENCHMARK" in text
    assert "NOT REAL-TIME CLINICAL PREDICTION" in text


def test_no_forbidden_terminology_anywhere(dashboard_client, first_cutoff):
    for view in ("replay", "forecast", "performance", "explainability", "dataquality"):
        response = dashboard_client.get("/", params={"view": view, "stay_id": DEMO_STAY, "prediction_time": first_cutoff[-1]})
        lowered = response.text.lower()
        remainder = lowered.replace(DISCLAIMER_SENTENCE, "")
        for term in FORBIDDEN_TERMS:
            assert term not in remainder, f"forbidden term {term!r} found in view={view}"
        assert "confidence" not in remainder, f"unguarded 'confidence' usage found in view={view}"
        assert "quality score" not in remainder, f"unguarded 'quality score' usage found in view={view}"


def test_no_discharge_eta_wording(dashboard_client, first_cutoff):
    response = dashboard_client.get("/", params={"view": "replay", "stay_id": DEMO_STAY, "prediction_time": first_cutoff[-1]})
    assert "discharge" not in response.text.lower()


def test_icu_and_support_exact_titles_present(dashboard_client, first_cutoff):
    response = dashboard_client.get("/", params={"view": "replay", "stay_id": DEMO_STAY, "prediction_time": first_cutoff[-1]})
    assert "Remaining ICU stay time" in response.text
    assert "New Organ-Support Initiation Risk" in response.text


def test_replay_recomputation_across_cutoffs(dashboard_client, first_cutoff):
    """Different cutoffs must be genuinely recomputed (independent API
    calls), not read from any cache -- verified by distinct elapsed-hours
    values appearing per cutoff in the rendered page."""

    first_page = dashboard_client.get("/", params={"view": "replay", "stay_id": DEMO_STAY, "prediction_time": first_cutoff[0]})
    last_page = dashboard_client.get("/", params={"view": "replay", "stay_id": DEMO_STAY, "prediction_time": first_cutoff[-1]})
    assert first_cutoff[0] in first_page.text
    assert first_cutoff[-1] in last_page.text
    assert first_page.text != last_page.text


def test_replay_history_accumulates_sequential_calls(dashboard_client, first_cutoff):
    response = dashboard_client.get("/", params={"view": "replay", "stay_id": DEMO_STAY, "prediction_time": first_cutoff[3]})
    assert response.status_code == 200
    assert "Replay history" in response.text
    for i in range(4):
        assert f"t{i}" in response.text


def test_model_metadata_renders_short_and_full_hashes(dashboard_client, first_cutoff):
    response = dashboard_client.get("/", params={"view": "dataquality", "stay_id": DEMO_STAY, "prediction_time": first_cutoff[-1]})
    assert "b355c26339271f0b84c123eb46cfb61c8df0cc9c9c6fcfcbb86c29047d10ad3c" in response.text


def test_data_quality_never_labeled_confidence(dashboard_client, first_cutoff):
    """The panel must EXPLICITLY disclaim confidence (the word appears only
    inside that negation), never present a count/status AS a confidence
    value."""

    response = dashboard_client.get("/", params={"view": "dataquality", "stay_id": DEMO_STAY, "prediction_time": first_cutoff[-1]})
    lowered = response.text.lower()
    assert DISCLAIMER_SENTENCE in lowered
    remainder = lowered.replace(DISCLAIMER_SENTENCE, "")
    assert "confidence" not in remainder


def test_performance_page_reads_frozen_artifacts_and_triggers_no_inference(dashboard_client, first_cutoff):
    state_before = json.loads((ROOT / "artifacts/performance_v2/governance/v2_fresh_test_access_state.json").read_text())
    response = dashboard_client.get("/", params={"view": "performance"})
    assert response.status_code == 200
    assert "1.0744" in response.text or "1.07" in response.text  # recovery24 MAE, from final_metrics_v2.json
    state_after = json.loads((ROOT / "artifacts/performance_v2/governance/v2_fresh_test_access_state.json").read_text())
    assert state_before == state_after


def test_v1_v2_improvement_panel_present(dashboard_client):
    response = dashboard_client.get("/", params={"view": "performance"})
    assert "different independent final-test cohorts" in response.text
    assert "0.6337" in response.text  # v1 calibrated AUPRC, historical context


def test_unknown_stay_dashboard_404(dashboard_client):
    response = dashboard_client.get("/", params={"view": "replay", "stay_id": "SYN-E-99999999"})
    assert response.status_code == 404


def test_fresh_test_stay_dashboard_blocked(dashboard_client):
    response = dashboard_client.get("/", params={"view": "replay", "stay_id": "SYN-V2-E-00000001"})
    assert response.status_code == 404


def test_v2_dashboard_source_never_imports_model_or_scientific_code():
    """V2's equivalent of tests/test_dashboard_contract.py's boundary check:
    the dashboard package must call the API, never import/instantiate model,
    SHAP, isotonic, or SOFA code directly. Checked by import statements
    (not literal text -- V2's dashboard legitimately DISPLAYS the word
    "xgboost" since every V2 model genuinely is XGBoost)."""

    for name in ("v2_app.py", "v2_render.py", "v2_charts.py", "v2_api_client.py"):
        text = (ROOT / "dashboard" / name).read_text()
        for forbidden_import in (
            "import xgboost", "import shap", "from shap", "import torch",
            "from serving.v2.runtime", "from serving.v2.explanations",
            "from performance_v2", "PredictionPipeline(",
        ):
            assert forbidden_import not in text, f"{name} illegally imports/uses {forbidden_import!r}"
