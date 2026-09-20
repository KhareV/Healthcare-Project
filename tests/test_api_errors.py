from serving.artifacts import (
    ArtifactCompatibilityError,
    ArtifactHashMismatchError,
    ArtifactMissingError,
)
from serving.history import IllegalCutoffError, UnknownStayError
from serving.pipeline import PredictionPipelineError
from serving.preprocessing import ServingFeatureContractError
from serving_helpers import FixtureRuntime
from api_helpers import REQUEST, build_pipeline, client_for


class FailingInputProvider:
    def __init__(self, error):
        self.error = error
        self.calls = 0

    def get_canonical_input(self, **_kwargs):
        self.calls += 1
        raise self.error

    def data_quality(self, **_kwargs):
        raise AssertionError("quality must not be requested after input failure")


def _input_failure(tmp_path, error):
    runtime = FixtureRuntime()
    runtime.input_provider = FailingInputProvider(error)
    _, pipeline, runtime = build_pipeline(tmp_path, runtime=runtime)
    response = client_for(pipeline).post("/predict", json=REQUEST)
    return response, runtime


def test_unknown_stay_maps_to_nonleaking_404_before_model_calls(tmp_path):
    response, runtime = _input_failure(tmp_path, UnknownStayError("restricted /path and PHI"))
    assert response.status_code == 404
    assert response.json() == {
        "error": {"code": "unknown_stay", "message": "requested stay is unavailable"}
    }
    assert all(model.predict_calls == 0 for model in runtime.predictors.values())
    assert "/path" not in response.text and "PHI" not in response.text


def test_illegal_cutoff_maps_to_422_before_model_calls(tmp_path):
    response, runtime = _input_failure(tmp_path, IllegalCutoffError("off grid"))
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "illegal_prediction_time"
    assert all(model.predict_calls == 0 for model in runtime.predictors.values())


class FailingPipeline:
    def __init__(self, delegate, error):
        self.delegate = delegate
        self.error = error
        self.calls = 0

    @property
    def bundle(self):
        return self.delegate.bundle

    def predict(self, _request):
        self.calls += 1
        raise self.error


def test_known_availability_failures_map_to_safe_503(tmp_path):
    _, pipeline, _ = build_pipeline(tmp_path)
    errors = (
        ArtifactMissingError("/secret/model missing"),
        ArtifactHashMismatchError("/secret/hash mismatch"),
        ArtifactCompatibilityError("preprocessor incompatible"),
        ServingFeatureContractError("feature schema incompatible"),
        PredictionPipelineError("calibrator or explanation unavailable"),
    )
    for error in errors:
        failing = FailingPipeline(pipeline, error)
        response = client_for(failing).post("/predict", json=REQUEST)
        assert response.status_code == 503
        assert response.json() == {
            "error": {
                "code": "serving_unavailable",
                "message": "validated serving dependencies are unavailable",
            }
        }
        assert "/secret" not in response.text


def test_default_production_app_fails_closed_without_real_bundle():
    from fastapi.testclient import TestClient
    from api.main import app

    client = TestClient(app)
    assert client.post("/predict", json=REQUEST).status_code == 503
    assert client.get("/model-metadata").status_code == 503


def test_invalid_or_uncalibrated_support_probability_cannot_succeed(tmp_path):
    _, pipeline, runtime = build_pipeline(tmp_path)

    def invalid(_prepared):
        runtime.predictors["organ_support"].predict_calls += 1
        return 2.0

    runtime.predictors["organ_support"].predict = invalid
    response = client_for(pipeline).post("/predict", json=REQUEST)
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "serving_unavailable"
