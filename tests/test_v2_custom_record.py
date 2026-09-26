"""Focused tests for the "bring your own data" custom-record feature:
serving.v2.custom_record and the /custom-records* API routes.

Deterministic and network-free (AI-endpoint tests mock the outbound Groq
call), mirroring the conventions in tests/test_v2_ai_recommendation.py."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

FULL_OBSERVATIONS = [
    {"concept": "heart_rate", "hours_since_admission": 2, "value": 88},
    {"concept": "mean_arterial_pressure", "hours_since_admission": 2, "value": 72},
    {"concept": "systolic_blood_pressure", "hours_since_admission": 2, "value": 110},
    {"concept": "diastolic_blood_pressure", "hours_since_admission": 2, "value": 65},
    {"concept": "respiratory_rate", "hours_since_admission": 2, "value": 18},
    {"concept": "oxygen_saturation", "hours_since_admission": 2, "value": 96},
    {"concept": "temperature", "hours_since_admission": 2, "value": 37.1},
    {"concept": "pao2", "hours_since_admission": 2, "value": 85},
    {"concept": "fio2", "hours_since_admission": 2, "value": 0.3},
    {"concept": "creatinine", "hours_since_admission": 2, "value": 1.0},
    {"concept": "platelet_count", "hours_since_admission": 2, "value": 210},
    {"concept": "bilirubin_total", "hours_since_admission": 2, "value": 0.7},
    {"concept": "glasgow_coma_scale", "hours_since_admission": 2, "value": 15},
    {"concept": "lactate", "hours_since_admission": 2, "value": 1.4},
    {"concept": "heart_rate", "hours_since_admission": 26, "value": 92},
    {"concept": "mean_arterial_pressure", "hours_since_admission": 26, "value": 68},
    {"concept": "oxygen_saturation", "hours_since_admission": 26, "value": 94},
    {"concept": "respiratory_rate", "hours_since_admission": 26, "value": 22},
    {"concept": "lactate", "hours_since_admission": 26, "value": 2.1},
]


@pytest.fixture(scope="module")
def runtime():
    from serving.v2.runtime import V2ServingRuntime

    return V2ServingRuntime(ROOT)


def test_canonical_concepts_excludes_support_and_urine_channels(runtime):
    from serving.v2.custom_record import canonical_concepts

    concepts = {item["concept"] for item in canonical_concepts(runtime.root)}
    assert concepts == {
        "heart_rate", "mean_arterial_pressure", "systolic_blood_pressure", "diastolic_blood_pressure",
        "respiratory_rate", "oxygen_saturation", "temperature", "pao2", "fio2", "creatinine",
        "platelet_count", "bilirubin_total", "glasgow_coma_scale", "lactate",
    }
    assert "urine_output_volume" not in concepts
    assert "vasopressor_on" not in concepts and "invasive_ventilation_on" not in concepts


def test_build_custom_record_end_to_end_produces_a_real_prediction(runtime):
    from serving.v2.custom_record import build_custom_record

    record = build_custom_record(
        runtime, patient_alias="  Test Patient  ", age_years=58, sex_category="MALE", observations=FULL_OBSERVATIONS,
    )
    assert record["patient_alias"] == "Test Patient"
    assert record["stay_id"].startswith("CUSTOM-")
    assert record["data_readiness"] == "READY"
    assert record["n_legal_cutoffs"] >= 1
    assert record["concept_coverage"]["observed"] == 14

    cutoff = record["legal_cutoffs"][0]
    prediction = runtime.predict(record["stay_id"], cutoff)
    assert prediction.stay_id == record["stay_id"]
    assert prediction.current_sofa >= 0.0
    assert 0.0 <= prediction.support_calibrated_probability <= 1.0
    assert prediction.recovery24_sofa == pytest.approx(max(0.0, min(24.0, prediction.current_sofa + prediction.recovery24_delta)))


def test_history_truncation_respects_the_selected_cutoff(runtime):
    from serving.v2.custom_record import build_custom_record

    record = build_custom_record(
        runtime, patient_alias="History Test", age_years=40, sex_category="FEMALE", observations=FULL_OBSERVATIONS,
    )
    events = runtime._events_by_stay[record["stay_id"]]
    cutoff = record["legal_cutoffs"][0]  # elapsed 24h -> only the hour=2 observations are <= cutoff
    visible = [e for e in events if e["event_time"] <= cutoff]
    assert len(visible) == 14
    assert all(e["event_time"] <= cutoff for e in visible)


def test_glasgow_coma_scale_must_be_a_whole_number(runtime):
    from serving.v2.custom_record import CustomRecordError, build_custom_record

    with pytest.raises(CustomRecordError, match="whole number"):
        build_custom_record(
            runtime, patient_alias="Bad GCS", age_years=40, sex_category="FEMALE",
            observations=[{"concept": "glasgow_coma_scale", "hours_since_admission": 2, "value": 12.5}],
        )


def test_unsupported_concept_is_rejected(runtime):
    from serving.v2.custom_record import CustomRecordError, build_custom_record

    with pytest.raises(CustomRecordError, match="unsupported canonical concept"):
        build_custom_record(
            runtime, patient_alias="Bad concept", age_years=40, sex_category="FEMALE",
            observations=[{"concept": "urine_output_volume", "hours_since_admission": 2, "value": 40}],
        )


def test_a_single_early_observation_is_ready_but_warns_about_sparse_coverage(runtime):
    from serving.v2.custom_record import build_custom_record

    record = build_custom_record(
        runtime, patient_alias="Sparse Patient", age_years=40, sex_category="FEMALE",
        observations=[{"concept": "heart_rate", "hours_since_admission": 2, "value": 80}],
    )
    assert record["data_readiness"] == "READY"
    assert record["n_legal_cutoffs"] >= 1  # the +24h cutoff is always legal once >=1 observation exists
    assert any("missing" in w for w in record["warnings"])  # only 1 of 14 concepts covered


def test_observation_at_or_before_admission_is_rejected(runtime):
    from serving.v2.custom_record import CustomRecordError, build_custom_record

    with pytest.raises(CustomRecordError, match="greater than 0"):
        build_custom_record(
            runtime, patient_alias="At Admission", age_years=40, sex_category="FEMALE",
            observations=[{"concept": "heart_rate", "hours_since_admission": 0, "value": 80}],
        )


def test_observation_beyond_the_grid_ceiling_is_rejected(runtime):
    from serving.v2.custom_record import CustomRecordError, build_custom_record

    with pytest.raises(CustomRecordError, match="at most 90"):
        build_custom_record(
            runtime, patient_alias="Too Late", age_years=40, sex_category="FEMALE",
            observations=[{"concept": "heart_rate", "hours_since_admission": 91, "value": 80}],
        )


def test_no_evidence_near_first_cutoff_warns_but_still_predicts(runtime):
    from serving.v2.custom_record import build_custom_record

    record = build_custom_record(
        runtime, patient_alias="Late Data Only", age_years=40, sex_category="FEMALE",
        observations=[{"concept": "heart_rate", "hours_since_admission": 60, "value": 95}],
    )
    assert record["data_readiness"] == "READY"
    assert any("first legal cutoff" in w for w in record["warnings"])
    first_cutoff = record["legal_cutoffs"][0]
    prediction = runtime.predict(record["stay_id"], first_cutoff)  # must not raise
    assert prediction.current_sofa >= 0.0


def test_empty_alias_and_empty_observations_are_rejected(runtime):
    from serving.v2.custom_record import CustomRecordError, build_custom_record

    with pytest.raises(CustomRecordError):
        build_custom_record(runtime, patient_alias="   ", age_years=40, sex_category="FEMALE", observations=FULL_OBSERVATIONS)
    with pytest.raises(CustomRecordError):
        build_custom_record(runtime, patient_alias="Nobody", age_years=40, sex_category="FEMALE", observations=[])


def test_custom_stay_ids_never_collide_with_frozen_or_fresh_test_namespaces(runtime):
    from serving.v2.custom_record import build_custom_record

    record = build_custom_record(
        runtime, patient_alias="Namespace Check", age_years=40, sex_category="FEMALE", observations=FULL_OBSERVATIONS,
    )
    assert record["stay_id"].startswith("CUSTOM-")
    assert not record["stay_id"].startswith("SYN-")


def test_custom_record_module_never_writes_to_disk():
    source = (ROOT / "src" / "serving" / "v2" / "custom_record.py").read_text(encoding="utf-8")
    for forbidden in ("write_text(", "json.dump(", "open(", ".mkdir("):
        assert forbidden not in source, f"custom_record.py must never write to disk (found {forbidden!r})"


def test_get_custom_record_round_trips(runtime):
    from serving.v2.custom_record import build_custom_record, get_custom_record

    record = build_custom_record(
        runtime, patient_alias="Round Trip", age_years=45, sex_category="MALE", observations=FULL_OBSERVATIONS,
    )
    fetched = get_custom_record(runtime, record["stay_id"])
    assert fetched is not None
    assert fetched["patient_alias"] == "Round Trip"
    assert fetched["legal_cutoffs"] == record["legal_cutoffs"]
    assert get_custom_record(runtime, "CUSTOM-DOES-NOT-EXIST") is None


# --- API endpoints -------------------------------------------------------------

@pytest.fixture(scope="module")
def api_client():
    from fastapi.testclient import TestClient

    from api.v2_app import build_v2_app

    return TestClient(build_v2_app(ROOT))


def test_custom_records_schema_endpoint(api_client):
    response = api_client.get("/custom-records/schema")
    assert response.status_code == 200
    concepts = response.json()["concepts"]
    assert len(concepts) == 14
    assert all({"concept", "label", "hint", "unit", "provenance_id"} <= set(item) for item in concepts)


def test_create_and_predict_via_api_end_to_end(api_client):
    response = api_client.post(
        "/custom-records",
        json={"patient_alias": "API Patient", "age_years": 50, "sex_category": "FEMALE", "observations": FULL_OBSERVATIONS},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["data_readiness"] == "READY"
    stay_id, cutoff = body["stay_id"], body["legal_cutoffs"][0]

    predict_response = api_client.post("/predict", json={"stay_id": stay_id, "prediction_time": cutoff})
    assert predict_response.status_code == 200
    prediction = predict_response.json()
    assert prediction["stay_id"] == stay_id
    assert "recovery" in prediction and "icu_stay_time" in prediction and "organ_support" in prediction
    assert all(prediction["explanations"][task]["diagnostics"]["additivity_check_passed"] for task in prediction["explanations"])

    history_response = api_client.get("/history", params={"stay_id": stay_id, "prediction_time": cutoff})
    assert history_response.status_code == 200
    assert history_response.json()["count"] == 14

    get_response = api_client.get(f"/custom-records/{stay_id}")
    assert get_response.status_code == 200
    assert get_response.json()["patient_alias"] == "API Patient"


def test_create_custom_record_rejects_invalid_input_with_422(api_client):
    response = api_client.post(
        "/custom-records",
        json={"patient_alias": "Bad", "age_years": 50, "sex_category": "FEMALE", "observations": [{"concept": "not_a_real_concept", "hours_since_admission": 2, "value": 1}]},
    )
    assert response.status_code == 422


def test_unknown_custom_record_returns_404(api_client):
    response = api_client.get("/custom-records/CUSTOM-NEVER-CREATED")
    assert response.status_code == 404


def test_demo_subjects_and_fresh_test_guard_unaffected_by_custom_records(api_client):
    demo_response = api_client.get("/demo-subjects")
    assert demo_response.status_code == 200
    assert len(demo_response.json()["demo_subjects"]) == 3

    for stay_id in ("SYN-V2-E-00000001", "SYN-E-99999999"):
        rejected = api_client.post("/predict", json={"stay_id": stay_id, "prediction_time": "2101-01-01T00:00:00Z"})
        assert rejected.status_code == 404


def test_assistant_endpoint_grounds_custom_record_alias(api_client, monkeypatch):
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

    create_response = api_client.post(
        "/custom-records",
        json={"patient_alias": "Copilot Alias Check", "age_years": 33, "sex_category": "MALE", "observations": FULL_OBSERVATIONS},
    )
    body = create_response.json()

    response = api_client.post("/assistant", json={"stay_id": body["stay_id"], "prediction_time": body["legal_cutoffs"][0]})
    assert response.status_code == 200
    user_message = captured["payload"]["messages"][1]["content"]
    assert "Copilot Alias Check" in user_message
    assert "custom record" in user_message.lower()


def test_no_test_run_touched_fresh_test_access_state(api_client):
    import json as _json

    state = _json.loads((ROOT / "artifacts/performance_v2/governance/v2_fresh_test_access_state.json").read_text())
    assert state["status"] == "FINAL_V2_RUN_COMPLETED"
    assert len(state["history"]) == 3
