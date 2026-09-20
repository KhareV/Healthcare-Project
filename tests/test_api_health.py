from fastapi.testclient import TestClient

from api.main import app
from api_helpers import build_pipeline, client_for


def test_default_health_is_alive_but_truthfully_not_ready():
    response = TestClient(app).get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "alive",
        "ready": False,
        "serving_scope": "unavailable",
        "mode": "RETROSPECTIVE_SEQUENTIAL_REPLAY",
    }


def test_synthetic_health_accesses_no_patient_history_or_models(tmp_path):
    _, pipeline, runtime = build_pipeline(tmp_path)
    response = client_for(pipeline).get("/health")
    assert response.status_code == 200
    assert response.json()["ready"] is False
    assert response.json()["serving_scope"] == "synthetic"
    assert runtime.input_provider.calls == []
    assert all(model.predict_calls == 0 for model in runtime.predictors.values())
