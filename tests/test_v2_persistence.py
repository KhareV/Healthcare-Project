"""Real, live-MongoDB integration tests for the persistent longitudinal
health-record layer (src/serving/v2/persistence.py): conditions, custom-
record durability, prediction-run history, and owner scoping.

Skipped entirely if MONGODB_URI is not configured -- a genuine optional
external service, exactly like this project's existing Groq/Kokoro tests.
Deliberately NOT mocked: this project's testing philosophy insists on
proving persistence claims with actual data insertion against the real
configured cluster, not a simulated one (see docs/product_v2/
CUSTOM_RECORD_FLOW.md and the project history of catching real bugs only
by testing with genuine data). Every document these tests create is
deleted again at teardown so repeated runs never accumulate garbage in a
shared cluster.

Clerk verification is still mocked (a locally-generated RSA keypair, same
pattern as tests/test_v2_custom_record.py's authed_client_factory) -- that
part of the stack is unrelated to what this file is proving.
"""

import os
import sys
import time
import uuid
from pathlib import Path

import jwt as pyjwt
import pytest
from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

# dotenv_values (not load_dotenv) deliberately: this must NOT mutate the
# process-wide os.environ at import time, or every other test module that
# calls build_v2_app() in the same pytest session would silently pick up a
# live MONGODB_URI too and start writing real documents to the shared
# cluster under test users that never clean up after themselves.
# authed_client_factory below scopes MONGODB_URI with monkeypatch.setenv
# instead, which pytest automatically reverts after each test.
MONGODB_URI = os.environ.get("MONGODB_URI") or dotenv_values(ROOT / ".env").get("MONGODB_URI")
pytestmark = pytest.mark.skipif(not MONGODB_URI, reason="MONGODB_URI not configured; persistence tests are skipped")

REAL_PUBLISHABLE_KEY = "pk_test_dmVyaWZpZWQta2luZ2Zpc2gtNTQuY2xlcmsuYWNjb3VudHMuZGV2JA"
ISSUER = "https://verified-kingfish-54.clerk.accounts.dev"


def _unique_owner(label: str) -> str:
    return f"pytest-persistence-{label}-{uuid.uuid4().hex[:10]}"


def _token_for(private_key, user_id: str) -> str:
    now = int(time.time())
    return pyjwt.encode({"sub": user_id, "iss": ISSUER, "iat": now, "exp": now + 300}, private_key, algorithm="RS256")


def _wipe_owner(persistence, owner_user_id: str) -> None:
    """Best-effort cleanup so this shared, real cluster never accumulates
    test garbage, even if a test fails partway through."""

    persistence._db.conditions.delete_many({"owner_user_id": owner_user_id})
    persistence._db.encounters.delete_many({"owner_user_id": owner_user_id})
    persistence._db.prediction_runs.delete_many({"owner_user_id": owner_user_id})


@pytest.fixture()
def persistence():
    from serving.v2.persistence import MongoPersistence

    instance = MongoPersistence(uri=MONGODB_URI)
    assert instance.enabled, "MONGODB_URI is set but the cluster is unreachable -- check connectivity"
    return instance


@pytest.fixture()
def authed_client_factory(monkeypatch):
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
    monkeypatch.setenv("MONGODB_URI", MONGODB_URI)

    def _make():
        # A fresh build_v2_app call constructs a brand-new V2ServingRuntime
        # (empty in-memory ephemeral state) and a brand-new MongoPersistence
        # -- connected to the SAME real cluster -- so calling this twice
        # genuinely simulates an API restart with durable storage intact.
        return TestClient(build_v2_app(ROOT)), private_key

    return _make


OBSERVATIONS = [
    {"concept": "heart_rate", "hours_since_admission": 2, "value": 88},
    {"concept": "mean_arterial_pressure", "hours_since_admission": 2, "value": 74},
    {"concept": "glasgow_coma_scale", "hours_since_admission": 2, "value": 15},
    # A second, later observation so outtime_hours (max(30, last_hour+6, ...))
    # clears 36h -- otherwise this record would have exactly one legal
    # cutoff (hour 24) and any test needing >=2 cutoffs would silently only
    # ever see one.
    {"concept": "lactate", "hours_since_admission": 32, "value": 1.2},
]


def test_direct_layer_condition_round_trip(persistence):
    owner = _unique_owner("condition")
    try:
        assert persistence.list_conditions(owner_user_id=owner) == []
        created = persistence.add_condition(owner_user_id=owner, label="Ischemic heart disease", diagnosed_year=2022, status="active")
        assert created["label"] == "Ischemic heart disease"
        assert created["condition_id"]

        listed = persistence.list_conditions(owner_user_id=owner)
        assert len(listed) == 1
        assert listed[0]["condition_id"] == created["condition_id"]

        assert persistence.delete_condition(owner_user_id=owner, condition_id=created["condition_id"]) is True
        assert persistence.list_conditions(owner_user_id=owner) == []
        assert persistence.delete_condition(owner_user_id=owner, condition_id=created["condition_id"]) is False
    finally:
        _wipe_owner(persistence, owner)


def test_direct_layer_encounter_save_and_get(persistence):
    owner = _unique_owner("encounter")
    stay_id = f"CUSTOM-TEST{uuid.uuid4().hex[:8].upper()}"
    try:
        assert persistence.get_encounter(stay_id=stay_id) is None
        persistence.save_encounter(
            owner_user_id=owner, stay_id=stay_id, subject_id="CUSTOM-SUBJ-TEST", patient_alias="Direct Layer Test",
            age_years=61, sex_category="FEMALE", intime="2024-01-01T00:00:00Z", outtime="2024-01-03T00:00:00Z",
            observations=OBSERVATIONS, support_intervals=[],
        )
        doc = persistence.get_encounter(stay_id=stay_id)
        assert doc is not None
        assert doc["owner_user_id"] == owner
        assert doc["patient_alias"] == "Direct Layer Test"
        assert doc["observations"] == OBSERVATIONS

        listed = persistence.list_encounters(owner_user_id=owner)
        assert [d["stay_id"] for d in listed] == [stay_id]
    finally:
        _wipe_owner(persistence, owner)


def test_custom_record_persists_and_survives_simulated_restart(authed_client_factory, persistence):
    owner = _unique_owner("restart")
    try:
        client1, private_key = authed_client_factory()
        token = _token_for(private_key, owner)
        headers = {"Authorization": f"Bearer {token}"}

        assert client1.get("/health").json()["persistence_mode"] == "mongodb"

        create = client1.post(
            "/custom-records",
            json={"patient_alias": "Restart Test Patient", "age_years": 58, "sex_category": "MALE", "observations": OBSERVATIONS, "support_intervals": []},
            headers=headers,
        )
        assert create.status_code == 200, create.text
        stay_id = create.json()["stay_id"]
        cutoff = create.json()["legal_cutoffs"][0]

        # Prove it was actually written to the real cluster, not just held
        # in this process's memory.
        assert persistence.get_encounter(stay_id=stay_id) is not None

        original = client1.post("/predict", json={"stay_id": stay_id, "prediction_time": cutoff}, headers=headers)
        assert original.status_code == 200, original.text
        original_body = original.json()

        # Simulate an API restart: a fresh runtime with empty in-memory
        # ephemeral state, backed by the same real MongoDB cluster.
        client2, _ = authed_client_factory()
        assert client2.get(f"/custom-records/{stay_id}", headers=headers).status_code == 200

        rehydrated = client2.post("/predict", json={"stay_id": stay_id, "prediction_time": cutoff}, headers=headers)
        assert rehydrated.status_code == 200, rehydrated.text
        rehydrated_body = rehydrated.json()

        # Rehydration must reconstruct byte-for-byte-equivalent derived
        # state from the same raw input -- not merely "some" response.
        assert rehydrated_body["current_sofa"] == original_body["current_sofa"]
        assert rehydrated_body["recovery"] == original_body["recovery"]
        assert rehydrated_body["icu_stay_time"] == original_body["icu_stay_time"]
        assert rehydrated_body["organ_support"] == original_body["organ_support"]

        # Cross-user isolation still holds after a rehydration: it restores
        # the stay under its true recorded owner, not the current caller.
        other_token = _token_for(private_key, _unique_owner("intruder"))
        forbidden = client2.get(f"/custom-records/{stay_id}", headers={"Authorization": f"Bearer {other_token}"})
        assert forbidden.status_code == 404
    finally:
        _wipe_owner(persistence, owner)


def test_prediction_history_accumulates_and_dedupes(authed_client_factory, persistence):
    owner = _unique_owner("history")
    try:
        client, private_key = authed_client_factory()
        token = _token_for(private_key, owner)
        headers = {"Authorization": f"Bearer {token}"}

        create = client.post(
            "/custom-records",
            json={"patient_alias": "History Test Patient", "age_years": 47, "sex_category": "FEMALE", "observations": OBSERVATIONS, "support_intervals": []},
            headers=headers,
        )
        assert create.status_code == 200, create.text
        stay_id = create.json()["stay_id"]
        all_cutoffs = create.json()["legal_cutoffs"]
        assert len(all_cutoffs) >= 2, f"test fixture needs >=2 legal cutoffs, got {all_cutoffs!r}"
        cutoffs = all_cutoffs[:2]

        empty = client.get(f"/custom-records/{stay_id}/predictions", headers=headers)
        assert empty.status_code == 200
        assert empty.json()["predictions"] == []

        for cutoff in cutoffs:
            resp = client.post("/predict", json={"stay_id": stay_id, "prediction_time": cutoff}, headers=headers)
            assert resp.status_code == 200, resp.text

        history = client.get(f"/custom-records/{stay_id}/predictions", headers=headers).json()["predictions"]
        assert [p["prediction_time"] for p in history] == cutoffs

        # Re-visiting the first cutoff must upsert, not duplicate.
        again = client.post("/predict", json={"stay_id": stay_id, "prediction_time": cutoffs[0]}, headers=headers)
        assert again.status_code == 200
        history_after = client.get(f"/custom-records/{stay_id}/predictions", headers=headers).json()["predictions"]
        assert len(history_after) == 2

        # A different user gets no history for someone else's record.
        other_token = _token_for(private_key, _unique_owner("intruder"))
        forbidden = client.get(f"/custom-records/{stay_id}/predictions", headers={"Authorization": f"Bearer {other_token}"})
        assert forbidden.status_code == 404
    finally:
        _wipe_owner(persistence, owner)


def test_list_custom_records_is_owner_scoped(authed_client_factory, persistence):
    owner_a = _unique_owner("lista")
    owner_b = _unique_owner("listb")
    try:
        client, private_key = authed_client_factory()
        token_a = _token_for(private_key, owner_a)
        token_b = _token_for(private_key, owner_b)

        create_a = client.post(
            "/custom-records",
            json={"patient_alias": "Owner A Patient", "age_years": 40, "sex_category": "MALE", "observations": OBSERVATIONS, "support_intervals": []},
            headers={"Authorization": f"Bearer {token_a}"},
        )
        create_b = client.post(
            "/custom-records",
            json={"patient_alias": "Owner B Patient", "age_years": 50, "sex_category": "FEMALE", "observations": OBSERVATIONS, "support_intervals": []},
            headers={"Authorization": f"Bearer {token_b}"},
        )
        stay_a, stay_b = create_a.json()["stay_id"], create_b.json()["stay_id"]

        list_a = client.get("/custom-records", headers={"Authorization": f"Bearer {token_a}"}).json()["records"]
        list_b = client.get("/custom-records", headers={"Authorization": f"Bearer {token_b}"}).json()["records"]

        assert {r["stay_id"] for r in list_a} == {stay_a}
        assert {r["stay_id"] for r in list_b} == {stay_b}
    finally:
        _wipe_owner(persistence, owner_a)
        _wipe_owner(persistence, owner_b)


def test_conditions_endpoints_via_api_are_owner_scoped(authed_client_factory, persistence):
    owner_a = _unique_owner("conda")
    owner_b = _unique_owner("condb")
    try:
        client, private_key = authed_client_factory()
        token_a = _token_for(private_key, owner_a)
        token_b = _token_for(private_key, owner_b)
        headers_a = {"Authorization": f"Bearer {token_a}"}
        headers_b = {"Authorization": f"Bearer {token_b}"}

        created = client.post("/health-record/conditions", json={"label": "Hypertension", "diagnosed_year": 2019, "status": "active"}, headers=headers_a)
        assert created.status_code == 200, created.text
        condition_id = created.json()["condition_id"]

        assert [c["condition_id"] for c in client.get("/health-record/conditions", headers=headers_a).json()["conditions"]] == [condition_id]
        assert client.get("/health-record/conditions", headers=headers_b).json()["conditions"] == []

        # Owner B cannot delete owner A's condition.
        assert client.delete(f"/health-record/conditions/{condition_id}", headers=headers_b).status_code == 404
        assert len(client.get("/health-record/conditions", headers=headers_a).json()["conditions"]) == 1

        assert client.delete(f"/health-record/conditions/{condition_id}", headers=headers_a).json()["deleted"] is True
        assert client.get("/health-record/conditions", headers=headers_a).json()["conditions"] == []
    finally:
        _wipe_owner(persistence, owner_a)
        _wipe_owner(persistence, owner_b)
