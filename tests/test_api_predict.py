import math
from pathlib import Path

import pytest

from api.schemas import PredictionResponse
from serving.prediction_schema import validate_response
from api_helpers import REQUEST, build_pipeline, client_for


def test_predict_accepts_exact_request_and_invokes_pipeline_once(tmp_path):
    _, pipeline, runtime = build_pipeline(tmp_path)

    class CountingPipeline:
        def __init__(self, delegate):
            self.delegate = delegate
            self.calls = []

        @property
        def bundle(self):
            return self.delegate.bundle

        def predict(self, request):
            self.calls.append(request)
            return self.delegate.predict(request)

    counted = CountingPipeline(pipeline)
    response = client_for(counted).post("/predict", json=REQUEST)
    assert response.status_code == 200
    assert counted.calls == [REQUEST]
    payload = response.json()
    assert PredictionResponse.model_validate(payload).model_dump() == payload
    assert validate_response(payload, synthetic=True) == payload
    assert payload["prediction_time"] == REQUEST["prediction_time"]
    assert payload["mode"] == "RETROSPECTIVE_SEQUENTIAL_REPLAY"
    assert payload["icu_stay_time_hours"] == pytest.approx(math.expm1(3.0))
    assert {task: predictor.predict_calls for task, predictor in runtime.predictors.items()} == {
        "recovery": 1, "icu_stay_time": 1, "organ_support": 1
    }


@pytest.mark.parametrize(
    "payload",
    (
        {"prediction_time": REQUEST["prediction_time"]},
        {"stay_id": REQUEST["stay_id"]},
        {"stay_id": REQUEST["stay_id"], "prediction_time": "not-a-time"},
        {**REQUEST, "model_family": "gru"},
    ),
)
def test_request_validation_is_422_and_structured(payload, tmp_path):
    _, pipeline, runtime = build_pipeline(tmp_path)
    response = client_for(pipeline).post("/predict", json=payload)
    assert response.status_code == 422
    assert response.json() == {
        "error": {"code": "request_validation_error", "message": "request is invalid"}
    }
    assert all(model.predict_calls == 0 for model in runtime.predictors.values())


def test_repeated_requests_are_deterministic_without_operational_noise(tmp_path):
    _, pipeline, runtime = build_pipeline(tmp_path)
    client = client_for(pipeline)
    loader_calls = list(runtime.loader_calls)
    assert client.post("/predict", json=REQUEST).json() == client.post(
        "/predict", json=REQUEST
    ).json()
    assert runtime.loader_calls == loader_calls
    assert runtime.calibrator.fit_calls == 0


def test_api_source_is_thin_and_phase12_dashboard_are_absent():
    root = Path(__file__).resolve().parents[1]
    source = (root / "api/main.py").read_text().lower()
    prohibited = (
        "build_features", "imputer", "scaler", "tensorize", "torch.load",
        "xgboost.booster", "pickle.load", "joblib.load", "treeexplainer",
        "integratedgradients", ".fit(", "choose_support_threshold", "current_sofa",
        "predictions.csv", "prediction_lookup", "final_predictions",
    )
    for term in prohibited:
        assert term not in source
    assert (root / "dashboard/app.py").is_file()
