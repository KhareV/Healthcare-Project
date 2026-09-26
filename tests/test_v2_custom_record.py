"""Focused tests for the "bring your own data" custom-record feature:
serving.v2.custom_record, organ-support entry, per-user ownership
(serving.v2.auth), and the /custom-records* API routes.

Deterministic and network-free (AI-endpoint tests mock the outbound Groq
call; auth tests mock the Clerk JWKS key fetch with a locally-generated RSA
keypair), mirroring the conventions in tests/test_v2_ai_recommendation.py
and tests/test_v2_auth.py."""

import sys
import time
from pathlib import Path

import jwt as pyjwt
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

REAL_PUBLISHABLE_KEY = "pk_test_dmVyaWZpZWQta2luZ2Zpc2gtNTQuY2xlcmsuYWNjb3VudHMuZGV2JA"
ISSUER = "https://verified-kingfish-54.clerk.accounts.dev"

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
        runtime, owner_user_id="user_a", patient_alias="  Test Patient  ", age_years=58, sex_category="MALE", observations=FULL_OBSERVATIONS,
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
        runtime, owner_user_id="user_a", patient_alias="History Test", age_years=40, sex_category="FEMALE", observations=FULL_OBSERVATIONS,
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
            runtime, owner_user_id="user_a", patient_alias="Bad GCS", age_years=40, sex_category="FEMALE",
            observations=[{"concept": "glasgow_coma_scale", "hours_since_admission": 2, "value": 12.5}],
        )


def test_unsupported_concept_is_rejected(runtime):
    from serving.v2.custom_record import CustomRecordError, build_custom_record

    with pytest.raises(CustomRecordError, match="unsupported canonical concept"):
        build_custom_record(
            runtime, owner_user_id="user_a", patient_alias="Bad concept", age_years=40, sex_category="FEMALE",
            observations=[{"concept": "urine_output_volume", "hours_since_admission": 2, "value": 40}],
        )


def test_a_single_early_observation_is_ready_but_warns_about_sparse_coverage(runtime):
    from serving.v2.custom_record import build_custom_record

    record = build_custom_record(
        runtime, owner_user_id="user_a", patient_alias="Sparse Patient", age_years=40, sex_category="FEMALE",
        observations=[{"concept": "heart_rate", "hours_since_admission": 2, "value": 80}],
    )
    assert record["data_readiness"] == "READY"
    assert record["n_legal_cutoffs"] >= 1  # the +24h cutoff is always legal once >=1 observation exists
    assert any("missing" in w for w in record["warnings"])  # only 1 of 14 concepts covered
    assert record["readiness_summary"]["observations_entered"] == 1
    assert record["readiness_summary"]["support_state_entered"] == {"vasopressor": False, "invasive_ventilation": False}


def test_observation_at_or_before_admission_is_rejected(runtime):
    from serving.v2.custom_record import CustomRecordError, build_custom_record

    with pytest.raises(CustomRecordError, match="greater than 0"):
        build_custom_record(
            runtime, owner_user_id="user_a", patient_alias="At Admission", age_years=40, sex_category="FEMALE",
            observations=[{"concept": "heart_rate", "hours_since_admission": 0, "value": 80}],
        )


def test_observation_beyond_the_grid_ceiling_is_rejected(runtime):
    from serving.v2.custom_record import CustomRecordError, build_custom_record

    with pytest.raises(CustomRecordError, match="at most 90"):
        build_custom_record(
            runtime, owner_user_id="user_a", patient_alias="Too Late", age_years=40, sex_category="FEMALE",
            observations=[{"concept": "heart_rate", "hours_since_admission": 91, "value": 80}],
        )


def test_no_evidence_near_first_cutoff_warns_but_still_predicts(runtime):
    from serving.v2.custom_record import build_custom_record

    record = build_custom_record(
        runtime, owner_user_id="user_a", patient_alias="Late Data Only", age_years=40, sex_category="FEMALE",
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
        build_custom_record(runtime, owner_user_id="user_a", patient_alias="   ", age_years=40, sex_category="FEMALE", observations=FULL_OBSERVATIONS)
    with pytest.raises(CustomRecordError):
        build_custom_record(runtime, owner_user_id="user_a", patient_alias="Nobody", age_years=40, sex_category="FEMALE", observations=[])
    with pytest.raises(CustomRecordError, match="owner_user_id"):
        build_custom_record(runtime, owner_user_id="", patient_alias="No Owner", age_years=40, sex_category="FEMALE", observations=FULL_OBSERVATIONS)


def test_custom_stay_ids_never_collide_with_frozen_or_fresh_test_namespaces(runtime):
    from serving.v2.custom_record import build_custom_record

    record = build_custom_record(
        runtime, owner_user_id="user_a", patient_alias="Namespace Check", age_years=40, sex_category="FEMALE", observations=FULL_OBSERVATIONS,
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
        runtime, owner_user_id="user_a", patient_alias="Round Trip", age_years=45, sex_category="MALE", observations=FULL_OBSERVATIONS,
    )
    fetched = get_custom_record(runtime, record["stay_id"])
    assert fetched is not None
    assert fetched["patient_alias"] == "Round Trip"
    assert fetched["legal_cutoffs"] == record["legal_cutoffs"]
    assert get_custom_record(runtime, "CUSTOM-DOES-NOT-EXIST") is None


# --- organ-support entry -------------------------------------------------------

def test_ongoing_vasopressor_and_bounded_ventilation_have_correct_off_on_transitions(runtime):
    """The scientifically load-bearing test: verifies real OFF->ON->OFF
    transitions against the frozen query_vasopressor_state/
    query_invasive_ventilation_state functions, inspecting the actual
    per-bin feature VALUES (not the observation mask, which is always
    True-for-non-padding on support channels by design)."""

    from serving.v2.custom_record import build_custom_record

    record = build_custom_record(
        runtime, owner_user_id="user_a", patient_alias="Support Test", age_years=50, sex_category="MALE",
        observations=[{"concept": "heart_rate", "hours_since_admission": 6, "value": 90}],
        support_intervals=[
            {"kind": "vasopressor", "agent": "norepinephrine", "rate": 0.15, "start_hour": 4, "end_hour": None},
            {"kind": "ventilation", "start_hour": 10, "end_hour": 20},
        ],
    )
    assert record["readiness_summary"]["support_state_entered"] == {"vasopressor": True, "invasive_ventilation": True}

    prediction = runtime.predict(record["stay_id"], "2024-01-02T12:00:00Z")  # hour 36
    names = list(prediction.feature_row["temporal_feature_names"])
    values = prediction.feature_row["history_values"]
    vaso = [row[names.index("vasopressor_on")] for row in values]
    vent = [row[names.index("invasive_ventilation_on")] for row in values]

    # bins 0-1 are pre-admission padding (None); bins 2-7 cover hours 6..36.
    assert vaso == [None, None, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]  # started hour 4, never ends -> ON throughout
    assert vent == [None, None, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0]  # [10,20) -> OFF, ON, ON, OFF, OFF, OFF


def test_support_already_active_at_cutoff_is_not_flagged_as_new(runtime):
    """'Already active at the prediction cutoff' vs 'newly initiated' is
    training-label semantics computed only during Phase-7/9 label
    generation, never in this serving path -- the frozen model already
    learned "new" from the temporal ON/OFF pattern across bins. This test
    only confirms the serving feature itself is a plain, correct ON/OFF
    state (no fabricated 'is_new' signal is invented anywhere here)."""

    from serving.v2.custom_record import build_custom_record

    record = build_custom_record(
        runtime, owner_user_id="user_a", patient_alias="Already Active", age_years=50, sex_category="MALE",
        observations=[{"concept": "heart_rate", "hours_since_admission": 6, "value": 90}],
        support_intervals=[{"kind": "vasopressor", "agent": "epinephrine", "rate": 0.1, "start_hour": 1, "end_hour": None}],
    )
    prediction = runtime.predict(record["stay_id"], record["legal_cutoffs"][0])
    assert "is_new" not in str(prediction.feature_row)  # no fabricated initiation flag anywhere in the feature row
    names = list(prediction.feature_row["temporal_feature_names"])
    last_bin_value = prediction.feature_row["history_values"][-1][names.index("vasopressor_on")]
    assert last_bin_value == 1.0  # correctly ON (started before cutoff), nothing more claimed


def test_future_support_intervals_never_leak_into_earlier_cutoffs(runtime):
    from serving.v2.custom_record import build_custom_record

    record = build_custom_record(
        runtime, owner_user_id="user_a", patient_alias="Future Leak Check", age_years=50, sex_category="MALE",
        observations=[{"concept": "heart_rate", "hours_since_admission": 6, "value": 90}],
        support_intervals=[{"kind": "vasopressor", "agent": "dopamine", "rate": 5.0, "start_hour": 80, "end_hour": None}],
    )
    early_cutoff = record["legal_cutoffs"][0]  # hour 24, well before the support interval starts at hour 80
    prediction = runtime.predict(record["stay_id"], early_cutoff)
    names = list(prediction.feature_row["temporal_feature_names"])
    values = [row[names.index("vasopressor_on")] for row in prediction.feature_row["history_values"]]
    assert all(v in (None, 0.0) for v in values)  # never ON before the interval starts


def test_vasopressor_requires_a_known_agent_and_a_positive_rate(runtime):
    from serving.v2.custom_record import CustomRecordError, build_custom_record

    with pytest.raises(CustomRecordError, match="agent"):
        build_custom_record(
            runtime, owner_user_id="user_a", patient_alias="Bad Agent", age_years=50, sex_category="MALE",
            observations=[{"concept": "heart_rate", "hours_since_admission": 6, "value": 90}],
            support_intervals=[{"kind": "vasopressor", "agent": "adrenaline_but_wrong_name", "rate": 0.1, "start_hour": 4, "end_hour": None}],
        )
    with pytest.raises(CustomRecordError, match="rate"):
        build_custom_record(
            runtime, owner_user_id="user_a", patient_alias="Bad Rate", age_years=50, sex_category="MALE",
            observations=[{"concept": "heart_rate", "hours_since_admission": 6, "value": 90}],
            support_intervals=[{"kind": "vasopressor", "agent": "norepinephrine", "rate": -1.0, "start_hour": 4, "end_hour": None}],
        )


def test_support_interval_end_must_be_after_start(runtime):
    from serving.v2.custom_record import CustomRecordError, build_custom_record

    with pytest.raises(CustomRecordError, match="end_hour"):
        build_custom_record(
            runtime, owner_user_id="user_a", patient_alias="Bad Interval", age_years=50, sex_category="MALE",
            observations=[{"concept": "heart_rate", "hours_since_admission": 6, "value": 90}],
            support_intervals=[{"kind": "ventilation", "start_hour": 20, "end_hour": 10}],
        )


# --- API endpoints -------------------------------------------------------------

@pytest.fixture(scope="module")
def api_client():
    """Built without a Clerk key in the environment -> local-dev auth mode
    (see serving.v2.auth): every request is treated as the fixed local-dev
    owner with no Authorization header required. This matches how most of
    this suite (and the rest of the existing test suite) already runs."""

    from fastapi.testclient import TestClient

    from api.v2_app import build_v2_app

    return TestClient(build_v2_app(ROOT))


@pytest.fixture()
def authed_client_factory(monkeypatch):
    """Builds a fresh TestClient with real Clerk verification switched on
    (against the project's actual Clerk instance issuer), with the JWKS
    fetch mocked to a locally-generated RSA keypair so no network call is
    made. Returns (client, private_key) so each test can sign its own
    tokens for whichever simulated user it needs."""

    from cryptography.hazmat.primitives.asymmetric import rsa
    from fastapi.testclient import TestClient

    from api.v2_app import build_v2_app
    from serving.v2 import auth as auth_module

    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public_key = private_key.public_key()

    class _FakeSigningKey:
        def __init__(self, key):
            self.key = key

    monkeypatch.setattr(auth_module.PyJWKClient, "get_signing_key_from_jwt", lambda self, token: _FakeSigningKey(public_key))
    monkeypatch.setenv("PUBLIC_CLERK_PUBLISHABLE_KEY", REAL_PUBLISHABLE_KEY)

    def _make():
        return TestClient(build_v2_app(ROOT)), private_key

    return _make


def _token_for(private_key, user_id: str) -> str:
    now = int(time.time())
    return pyjwt.encode({"sub": user_id, "iss": ISSUER, "iat": now, "exp": now + 60}, private_key, algorithm="RS256")


def test_custom_records_schema_endpoint(api_client):
    response = api_client.get("/custom-records/schema")
    assert response.status_code == 200
    concepts = response.json()["concepts"]
    assert len(concepts) == 14
    assert all({"concept", "label", "hint", "unit", "provenance_id"} <= set(item) for item in concepts)


def test_health_reports_auth_mode(api_client, authed_client_factory):
    assert api_client.get("/health").json()["auth_mode"] == "local_dev_no_auth"
    authed_client, _ = authed_client_factory()
    assert authed_client.get("/health").json()["auth_mode"] == "clerk_verified"


def test_create_and_predict_via_api_end_to_end_in_local_dev_mode(api_client):
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


def test_demo_stay_predict_requires_no_auth_even_in_clerk_configured_mode(authed_client_factory):
    """Demo patients remain globally available -- the ownership check is a
    no-op for any stay_id that isn't an ephemeral custom record."""

    client, _ = authed_client_factory()
    response = client.post("/predict", json={"stay_id": "SYN-E-00000001", "prediction_time": "2101-05-31T06:17:26Z"})
    assert response.status_code == 200


def test_creating_a_custom_record_requires_authentication_when_clerk_is_configured(authed_client_factory):
    client, private_key = authed_client_factory()

    unauthenticated = client.post(
        "/custom-records",
        json={"patient_alias": "No Token", "age_years": 50, "sex_category": "FEMALE", "observations": FULL_OBSERVATIONS},
    )
    assert unauthenticated.status_code == 401

    token = _token_for(private_key, "user_alice")
    authenticated = client.post(
        "/custom-records",
        json={"patient_alias": "Alice's Patient", "age_years": 50, "sex_category": "FEMALE", "observations": FULL_OBSERVATIONS},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert authenticated.status_code == 200


def test_cross_user_isolation_user_b_cannot_see_user_as_custom_record(authed_client_factory):
    client, private_key = authed_client_factory()
    alice_token = _token_for(private_key, "user_alice")
    bob_token = _token_for(private_key, "user_bob")

    created = client.post(
        "/custom-records",
        json={"patient_alias": "Alice Private", "age_years": 50, "sex_category": "FEMALE", "observations": FULL_OBSERVATIONS},
        headers={"Authorization": f"Bearer {alice_token}"},
    )
    stay_id, cutoff = created.json()["stay_id"], created.json()["legal_cutoffs"][0]

    # Alice can read/predict/history her own record.
    assert client.get(f"/custom-records/{stay_id}", headers={"Authorization": f"Bearer {alice_token}"}).status_code == 200
    assert client.post("/predict", json={"stay_id": stay_id, "prediction_time": cutoff}, headers={"Authorization": f"Bearer {alice_token}"}).status_code == 200
    assert client.get("/history", params={"stay_id": stay_id, "prediction_time": cutoff}, headers={"Authorization": f"Bearer {alice_token}"}).status_code == 200

    # Bob gets 404 (not 403 -- indistinguishable from "never existed") on every one of the same calls.
    assert client.get(f"/custom-records/{stay_id}", headers={"Authorization": f"Bearer {bob_token}"}).status_code == 404
    assert client.post("/predict", json={"stay_id": stay_id, "prediction_time": cutoff}, headers={"Authorization": f"Bearer {bob_token}"}).status_code == 404
    assert client.get("/history", params={"stay_id": stay_id, "prediction_time": cutoff}, headers={"Authorization": f"Bearer {bob_token}"}).status_code == 404

    # No token at all also gets 404 for the same record.
    assert client.get(f"/custom-records/{stay_id}").status_code == 404
    assert client.post("/predict", json={"stay_id": stay_id, "prediction_time": cutoff}).status_code == 404


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


def test_assistant_endpoint_respects_cross_user_isolation(authed_client_factory, monkeypatch):
    from serving.v2 import ai_recommendation as module

    monkeypatch.setattr(module.httpx, "post", lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not reach the LLM for an unauthorized caller")))

    client, private_key = authed_client_factory()
    alice_token = _token_for(private_key, "user_alice")
    bob_token = _token_for(private_key, "user_bob")

    created = client.post(
        "/custom-records",
        json={"patient_alias": "Alice Copilot", "age_years": 50, "sex_category": "FEMALE", "observations": FULL_OBSERVATIONS},
        headers={"Authorization": f"Bearer {alice_token}"},
    )
    stay_id, cutoff = created.json()["stay_id"], created.json()["legal_cutoffs"][0]

    response = client.post("/assistant", json={"stay_id": stay_id, "prediction_time": cutoff}, headers={"Authorization": f"Bearer {bob_token}"})
    assert response.status_code == 404


def test_no_test_run_touched_fresh_test_access_state(api_client):
    import json as _json

    state = _json.loads((ROOT / "artifacts/performance_v2/governance/v2_fresh_test_access_state.json").read_text())
    assert state["status"] == "FINAL_V2_RUN_COMPLETED"
    assert len(state["history"]) == 3
