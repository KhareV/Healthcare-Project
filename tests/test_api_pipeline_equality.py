import json

from api.schemas import PredictionRequest, PredictionResponse
from serving.prediction_schema import TOP_LEVEL_RESPONSE, load_contract
from serving_helpers import FixtureRuntime
from api_helpers import REQUEST, build_pipeline, client_for
from phase7_helpers import (
    LATER_CUTOFF,
    MAIN_CUTOFF,
    MAIN_STAY,
    TIMELINE_FIXTURE,
    build_provider,
    load_timelines,
)


def test_api_scientific_payload_equals_direct_pipeline_exactly(tmp_path):
    _, pipeline, _ = build_pipeline(tmp_path)
    direct = pipeline.predict(REQUEST)
    http = client_for(pipeline).post("/predict", json=REQUEST)
    assert http.status_code == 200
    assert http.json() == json.loads(json.dumps(direct))


def test_pydantic_and_phase5_required_fields_are_in_parity():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    contract = load_contract(root / "configs/prediction_schema_v1.json")
    request_schema = PredictionRequest.model_json_schema()
    response_schema = PredictionResponse.model_json_schema()
    assert set(request_schema["required"]) == set(contract["request"]["required"])
    assert set(response_schema["required"]) == set(contract["response"]["required"])
    assert set(response_schema["required"]) == TOP_LEVEL_RESPONSE
    assert request_schema["additionalProperties"] is False
    assert response_schema["additionalProperties"] is False


def test_openapi_exposes_only_required_public_routes_and_contracts(tmp_path):
    _, pipeline, _ = build_pipeline(tmp_path)
    client = client_for(pipeline)
    openapi = client.get("/openapi.json").json()
    assert {"/predict", "/health", "/model-metadata"}.issubset(openapi["paths"])
    request_ref = openapi["paths"]["/predict"]["post"]["requestBody"]["content"]["application/json"]["schema"]["$ref"]
    assert request_ref.endswith("/PredictionRequest")
    assert "real-time clinical inference" in openapi["info"]["description"]


class FixedRecoveryPayload:
    def recovery(self, _raw_output, _prepared_input, **_context):
        return {
            "delta_24h": 0.25,
            "delta_48h": -0.5,
            "reconstructed_sofa_24h": 4.25,
            "reconstructed_sofa_48h": 3.5,
        }


def _phase7_api(tmp_path, timelines):
    runtime = FixtureRuntime()
    runtime.input_provider = build_provider(timelines=timelines)[0]
    _, pipeline, runtime = build_pipeline(
        tmp_path,
        runtime=runtime,
        recovery_provider=FixedRecoveryPayload(),
    )
    return client_for(pipeline), runtime


def test_future_row_mutation_deletion_and_addition_cannot_change_api_output_at_t(tmp_path):
    payload = json.loads(TIMELINE_FIXTURE.read_text())
    baseline_client, _ = _phase7_api(tmp_path / "baseline", load_timelines(payload))
    baseline_request = {"stay_id": MAIN_STAY, "prediction_time": MAIN_CUTOFF}
    baseline = baseline_client.post("/predict", json=baseline_request)
    assert baseline.status_code == 200

    events = payload["timelines"][0]["events"]
    retained = [row for row in events if row["synthetic_available_time"] <= MAIN_CUTOFF]
    hostile_future = {
        "synthetic_stay_key": MAIN_STAY,
        "synthetic_available_time": "2026-01-05T18:00:00+00:00",
        "synthetic_feature": "SYNTHETIC_SIGNAL_A",
        "synthetic_value": -1e300,
        "synthetic_sequence": 1000,
        "future_sofa_48h": 1e300,
        "support_target": 1,
    }
    payload["timelines"][0]["events"] = [hostile_future] + list(reversed(retained))
    changed_client, _ = _phase7_api(tmp_path / "changed", load_timelines(payload))
    changed = changed_client.post("/predict", json=baseline_request)
    assert changed.status_code == 200
    assert changed.json() == baseline.json()


def test_later_legal_cutoff_is_valid_without_requiring_output_to_change(tmp_path):
    client, _ = _phase7_api(tmp_path, load_timelines())
    early = client.post(
        "/predict", json={"stay_id": MAIN_STAY, "prediction_time": MAIN_CUTOFF}
    )
    later = client.post(
        "/predict", json={"stay_id": MAIN_STAY, "prediction_time": LATER_CUTOFF}
    )
    assert early.status_code == later.status_code == 200


def test_actual_phase7_provider_maps_illegal_cutoffs_and_unknown_stay(tmp_path):
    client, runtime = _phase7_api(tmp_path, load_timelines())
    for cutoff in (
        "2026-01-02T06:00:00+00:00",
        "2026-01-04T12:00:01+00:00",
        "2026-01-06T12:00:00+00:00",
    ):
        response = client.post(
            "/predict", json={"stay_id": MAIN_STAY, "prediction_time": cutoff}
        )
        assert response.status_code == 422
    unknown = client.post(
        "/predict",
        json={"stay_id": "SYNTHETIC_UNKNOWN", "prediction_time": MAIN_CUTOFF},
    )
    assert unknown.status_code == 404
    assert all(model.predict_calls == 0 for model in runtime.predictors.values())
