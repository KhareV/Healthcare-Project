"""Real, live-MongoDB integration tests for the persistent longitudinal
Health Record layer (src/serving/v2/persistence.py, src/serving/v2/
report_storage.py): patient profile, conditions (incl. edit), encounter-
to-profile linkage, observation/support history views, reports (metadata +
file storage + security), prediction history, export, and audit events.

Skipped entirely if MONGODB_URI is not configured -- a genuine optional
external service, exactly like this project's existing Groq/Kokoro tests.
Deliberately NOT mocked: this project's testing philosophy insists on
proving persistence claims with actual data insertion against the real
configured cluster, not a simulated one. Every document (and every locally
stored report file) these tests create is deleted again at teardown so
repeated runs never accumulate garbage in a shared cluster or on disk.

Clerk verification is still mocked (a locally-generated RSA keypair, same
pattern as tests/test_v2_custom_record.py's authed_client_factory) -- that
part of the stack is unrelated to what this file is proving.
"""

import os
import shutil
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
    """Best-effort cleanup so this shared, real cluster (and the local
    report-storage directory) never accumulate test garbage, even if a test
    fails partway through."""

    for coll in ("patient_profiles", "conditions", "encounters", "prediction_runs", "reports", "audit_events"):
        persistence._db[coll].delete_many({"owner_user_id": owner_user_id})
    shutil.rmtree(ROOT / "runtime" / "uploads" / owner_user_id, ignore_errors=True)


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

SUPPORT_INTERVALS = [
    {"kind": "vasopressor", "agent": "norepinephrine", "rate": 0.1, "start_hour": 4, "end_hour": None},
]

TINY_PDF_BYTES = b"%PDF-1.4\n%fake-pdf-for-tests\n"


def _create_record(client, headers, alias="Test Patient", observations=None, support_intervals=None):
    resp = client.post(
        "/custom-records",
        json={
            "patient_alias": alias, "age_years": 55, "sex_category": "FEMALE",
            "observations": observations if observations is not None else OBSERVATIONS,
            "support_intervals": support_intervals if support_intervals is not None else [],
        },
        headers=headers,
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


# --------------------------------------------------------------------------
# Patient profile
# --------------------------------------------------------------------------

def test_profile_get_or_create_and_update_roundtrip(persistence):
    owner = _unique_owner("profile")
    try:
        assert persistence.get_profile(owner_user_id=owner) is None
        created = persistence.get_or_create_profile(owner_user_id=owner)
        assert created["patient_id"].startswith("PAT-")
        assert created["owner_user_id"] == owner

        # get_or_create is idempotent -- the same profile, not a new one.
        again = persistence.get_or_create_profile(owner_user_id=owner)
        assert again["patient_id"] == created["patient_id"]

        updated = persistence.update_profile(owner_user_id=owner, fields={
            "display_name_or_alias": "Jordan Doe", "age_years": 39, "sex_category": "FEMALE",
            "blood_group": "O+", "height_cm": 168.0, "weight_kg": 61.5,
        })
        assert updated["display_name_or_alias"] == "Jordan Doe"
        assert updated["age_years"] == 39
        assert updated["blood_group"] == "O+"
        assert updated["patient_id"] == created["patient_id"]
    finally:
        _wipe_owner(persistence, owner)


def test_profile_api_endpoints_are_owner_isolated(authed_client_factory, persistence):
    owner_a = _unique_owner("profa")
    owner_b = _unique_owner("profb")
    try:
        client, private_key = authed_client_factory()
        headers_a = {"Authorization": f"Bearer {_token_for(private_key, owner_a)}"}
        headers_b = {"Authorization": f"Bearer {_token_for(private_key, owner_b)}"}

        put_a = client.put("/health-record/profile", json={"display_name_or_alias": "Owner A"}, headers=headers_a)
        assert put_a.status_code == 200, put_a.text
        put_b = client.put("/health-record/profile", json={"display_name_or_alias": "Owner B"}, headers=headers_b)
        assert put_b.status_code == 200, put_b.text

        get_a = client.get("/health-record/profile", headers=headers_a).json()
        get_b = client.get("/health-record/profile", headers=headers_b).json()
        assert get_a["display_name_or_alias"] == "Owner A"
        assert get_b["display_name_or_alias"] == "Owner B"
        assert get_a["patient_id"] != get_b["patient_id"]
    finally:
        _wipe_owner(persistence, owner_a)
        _wipe_owner(persistence, owner_b)


# --------------------------------------------------------------------------
# Conditions: create, edit, delete, owner isolation
# --------------------------------------------------------------------------

def test_direct_layer_condition_round_trip_with_edit(persistence):
    owner = _unique_owner("condition")
    try:
        assert persistence.list_conditions(owner_user_id=owner) == []
        profile = persistence.get_or_create_profile(owner_user_id=owner)
        created = persistence.add_condition(
            owner_user_id=owner, patient_id=profile["patient_id"], name="Ischemic heart disease",
            code=None, diagnosed_date="2022-06-01", status="active", notes="stable on medication",
        )
        assert created["name"] == "Ischemic heart disease"
        assert created["patient_id"] == profile["patient_id"]
        assert created["condition_id"]

        listed = persistence.list_conditions(owner_user_id=owner)
        assert len(listed) == 1
        assert listed[0]["condition_id"] == created["condition_id"]

        edited = persistence.update_condition(
            owner_user_id=owner, condition_id=created["condition_id"], fields={"status": "resolved", "notes": "fully resolved"}
        )
        assert edited["status"] == "resolved"
        assert edited["notes"] == "fully resolved"
        assert edited["name"] == "Ischemic heart disease"  # untouched fields survive a partial edit

        assert persistence.delete_condition(owner_user_id=owner, condition_id=created["condition_id"]) is True
        assert persistence.list_conditions(owner_user_id=owner) == []
        assert persistence.delete_condition(owner_user_id=owner, condition_id=created["condition_id"]) is False
    finally:
        _wipe_owner(persistence, owner)


def test_legacy_condition_shape_is_readable_and_deletable(persistence):
    """A condition inserted the way the original (pre-Health-Record-
    Completion) implementation wrote it -- addressed by Mongo _id, no
    condition_id/name/patient_id fields at all -- must still be listed,
    normalized, and deletable. Backward compatibility, not a hypothetical."""

    owner = _unique_owner("legacycond")
    try:
        result = persistence._db.conditions.insert_one({
            "owner_user_id": owner, "label": "Old Shape Condition", "diagnosed_year": 2020,
            "status": "active", "created_at": "2025-01-01T00:00:00Z",
        })
        legacy_id = str(result.inserted_id)

        listed = persistence.list_conditions(owner_user_id=owner)
        assert len(listed) == 1
        assert listed[0]["condition_id"] == legacy_id
        assert listed[0]["name"] == "Old Shape Condition"
        assert listed[0]["diagnosed_date"] == "2020-01-01"
        assert listed[0]["patient_id"] is None

        fetched = persistence.get_condition(owner_user_id=owner, condition_id=legacy_id)
        assert fetched is not None and fetched["name"] == "Old Shape Condition"

        assert persistence.delete_condition(owner_user_id=owner, condition_id=legacy_id) is True
        assert persistence.list_conditions(owner_user_id=owner) == []
    finally:
        _wipe_owner(persistence, owner)


def test_condition_api_create_edit_and_owner_isolation(authed_client_factory, persistence):
    owner_a = _unique_owner("conda")
    owner_b = _unique_owner("condb")
    try:
        client, private_key = authed_client_factory()
        headers_a = {"Authorization": f"Bearer {_token_for(private_key, owner_a)}"}
        headers_b = {"Authorization": f"Bearer {_token_for(private_key, owner_b)}"}

        created = client.post("/health-record/conditions", json={"name": "Hypertension", "status": "active"}, headers=headers_a)
        assert created.status_code == 200, created.text
        condition_id = created.json()["condition_id"]

        bad_status = client.post("/health-record/conditions", json={"name": "Bad", "status": "not-a-real-status"}, headers=headers_a)
        assert bad_status.status_code == 422

        edited = client.patch(f"/health-record/conditions/{condition_id}", json={"status": "resolved"}, headers=headers_a)
        assert edited.status_code == 200, edited.text
        assert edited.json()["status"] == "resolved"

        assert [c["condition_id"] for c in client.get("/health-record/conditions", headers=headers_a).json()["conditions"]] == [condition_id]
        assert client.get("/health-record/conditions", headers=headers_b).json()["conditions"] == []

        # Owner B cannot edit or delete owner A's condition.
        assert client.patch(f"/health-record/conditions/{condition_id}", json={"status": "active"}, headers=headers_b).status_code == 404
        assert client.delete(f"/health-record/conditions/{condition_id}", headers=headers_b).status_code == 404
        assert len(client.get("/health-record/conditions", headers=headers_a).json()["conditions"]) == 1

        assert client.delete(f"/health-record/conditions/{condition_id}", headers=headers_a).json()["deleted"] is True
        assert client.get("/health-record/conditions", headers=headers_a).json()["conditions"] == []
    finally:
        _wipe_owner(persistence, owner_a)
        _wipe_owner(persistence, owner_b)


# --------------------------------------------------------------------------
# Encounters: linkage to profile, migration-on-read, observation/support views
# --------------------------------------------------------------------------

def test_encounter_created_via_custom_records_links_to_profile(authed_client_factory, persistence):
    owner = _unique_owner("enclink")
    try:
        client, private_key = authed_client_factory()
        headers = {"Authorization": f"Bearer {_token_for(private_key, owner)}"}

        record = _create_record(client, headers, support_intervals=SUPPORT_INTERVALS)
        stay_id = record["stay_id"]

        profile = client.get("/health-record/profile", headers=headers).json()
        encounters = client.get("/health-record/encounters", headers=headers).json()["encounters"]
        assert len(encounters) == 1
        assert encounters[0]["stay_id"] == stay_id
        assert encounters[0]["patient_id"] == profile["patient_id"]
        assert encounters[0]["encounter_type"]
        assert encounters[0]["source"] == "MANUAL_STRUCTURED_ENTRY"

        single = client.get(f"/health-record/encounters/{stay_id}", headers=headers)
        assert single.status_code == 200
        assert single.json()["stay_id"] == stay_id
    finally:
        _wipe_owner(persistence, owner)


def test_legacy_encounter_without_patient_id_is_backfilled_on_read(persistence):
    """An encounter persisted before patient_id existed (the shape the
    previous phase's save_encounter wrote) must still load, and must be
    transparently linked to the caller's profile the first time it is
    listed -- migration-on-read, not a destructive rewrite."""

    owner = _unique_owner("legacyenc")
    stay_id = f"CUSTOM-LEGACY{uuid.uuid4().hex[:8].upper()}"
    try:
        persistence._db.encounters.insert_one({
            "owner_user_id": owner, "stay_id": stay_id, "subject_id": "CUSTOM-SUBJ-LEGACY",
            "patient_alias": "Legacy Encounter", "age_years": 70, "sex_category": "MALE",
            "intime": "2024-01-01T00:00:00Z", "outtime": "2024-01-03T00:00:00Z",
            "observations": OBSERVATIONS, "support_intervals": [], "created_at": "2025-01-01T00:00:00Z",
        })
        assert persistence.get_encounter(stay_id=stay_id)["patient_id"] is None

        listed = persistence.list_encounters(owner_user_id=owner)
        assert len(listed) == 1
        assert listed[0]["patient_id"] is not None
        assert listed[0]["encounter_type"] == "MANUAL_RECOVERY_EPISODE"

        profile = persistence.get_or_create_profile(owner_user_id=owner)
        assert listed[0]["patient_id"] == profile["patient_id"]

        # The backfill was a real, persisted write, not just an in-memory patch.
        assert persistence.get_encounter(stay_id=stay_id)["patient_id"] == profile["patient_id"]
    finally:
        _wipe_owner(persistence, owner)


def test_observation_and_support_history_views(authed_client_factory, persistence):
    owner = _unique_owner("obshist")
    try:
        client, private_key = authed_client_factory()
        headers = {"Authorization": f"Bearer {_token_for(private_key, owner)}"}

        _create_record(client, headers, alias="Obs History Patient", support_intervals=SUPPORT_INTERVALS)

        observations = client.get("/health-record/observations", headers=headers).json()["observations"]
        assert len(observations) == len(OBSERVATIONS)
        assert {o["concept"] for o in observations} == {o["concept"] for o in OBSERVATIONS}
        assert all(o["patient_alias"] == "Obs History Patient" for o in observations)

        support = client.get("/health-record/support", headers=headers).json()["support_intervals"]
        assert len(support) == 1
        assert support[0]["kind"] == "vasopressor"
        assert support[0]["agent"] == "norepinephrine"
    finally:
        _wipe_owner(persistence, owner)


# --------------------------------------------------------------------------
# Prediction history: linkage + persistence (existing behavior, kept)
# --------------------------------------------------------------------------

def test_custom_record_persists_and_survives_simulated_restart(authed_client_factory, persistence):
    owner = _unique_owner("restart")
    try:
        client1, private_key = authed_client_factory()
        headers = {"Authorization": f"Bearer {_token_for(private_key, owner)}"}

        assert client1.get("/health").json()["persistence_mode"] == "mongodb"

        record = _create_record(client1, headers, alias="Restart Test Patient")
        stay_id, cutoff = record["stay_id"], record["legal_cutoffs"][0]

        # Prove it was actually written to the real cluster, not just held
        # in this process's memory.
        assert persistence.get_encounter(stay_id=stay_id) is not None

        original = client1.post("/predict", json={"stay_id": stay_id, "prediction_time": cutoff}, headers=headers)
        assert original.status_code == 200, original.text
        original_body = original.json()

        # Also upload a report and set a profile field before "restarting",
        # to prove Part 21's requirement that profile + encounter + report
        # metadata + prediction history all survive an API restart.
        client1.put("/health-record/profile", json={"display_name_or_alias": "Restart Profile"}, headers=headers)
        files = {"file": ("restart-report.pdf", TINY_PDF_BYTES, "application/pdf")}
        data = {"title": "Restart Report", "document_type": "lab_report"}
        upload = client1.post("/health-record/reports", data=data, files=files, headers=headers)
        assert upload.status_code == 200, upload.text
        report_id = upload.json()["report_id"]

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

        # Profile, prediction history, and report metadata all survive too.
        assert client2.get("/health-record/profile", headers=headers).json()["display_name_or_alias"] == "Restart Profile"
        history = client2.get(f"/custom-records/{stay_id}/predictions", headers=headers).json()["predictions"]
        assert len(history) == 1
        report_after_restart = client2.get(f"/health-record/reports/{report_id}", headers=headers)
        assert report_after_restart.status_code == 200
        download_after_restart = client2.get(f"/health-record/reports/{report_id}/download", headers=headers)
        assert download_after_restart.status_code == 200
        assert download_after_restart.content == TINY_PDF_BYTES

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
        headers = {"Authorization": f"Bearer {_token_for(private_key, owner)}"}

        record = _create_record(client, headers, alias="History Test Patient")
        stay_id = record["stay_id"]
        all_cutoffs = record["legal_cutoffs"]
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

        # The consolidated /health-record/predictions view sees it too.
        consolidated = client.get("/health-record/predictions", headers=headers).json()["predictions"]
        assert len(consolidated) == 2

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
        headers_a = {"Authorization": f"Bearer {_token_for(private_key, owner_a)}"}
        headers_b = {"Authorization": f"Bearer {_token_for(private_key, owner_b)}"}

        record_a = _create_record(client, headers_a, alias="Owner A Patient")
        record_b = _create_record(client, headers_b, alias="Owner B Patient")

        list_a = client.get("/custom-records", headers=headers_a).json()["records"]
        list_b = client.get("/custom-records", headers=headers_b).json()["records"]

        assert {r["stay_id"] for r in list_a} == {record_a["stay_id"]}
        assert {r["stay_id"] for r in list_b} == {record_b["stay_id"]}
    finally:
        _wipe_owner(persistence, owner_a)
        _wipe_owner(persistence, owner_b)


# --------------------------------------------------------------------------
# Reports: metadata, file storage, security (owner isolation, path
# traversal, unsupported types, size limit)
# --------------------------------------------------------------------------

def test_report_upload_list_download_delete(authed_client_factory, persistence):
    owner = _unique_owner("report")
    try:
        client, private_key = authed_client_factory()
        headers = {"Authorization": f"Bearer {_token_for(private_key, owner)}"}

        files = {"file": ("blood-test.pdf", TINY_PDF_BYTES, "application/pdf")}
        data = {"title": "Blood Test", "document_type": "lab_report", "report_date": "2026-01-15"}
        upload = client.post("/health-record/reports", data=data, files=files, headers=headers)
        assert upload.status_code == 200, upload.text
        body = upload.json()
        assert body["title"] == "Blood Test"
        assert body["processing_status"] == "NOT_PARSED"
        assert "storage_key" not in body
        report_id = body["report_id"]

        listed = client.get("/health-record/reports", headers=headers).json()["reports"]
        assert len(listed) == 1
        assert all("storage_key" not in r for r in listed)

        fetched = client.get(f"/health-record/reports/{report_id}", headers=headers)
        assert fetched.status_code == 200
        assert "storage_key" not in fetched.json()

        downloaded = client.get(f"/health-record/reports/{report_id}/download", headers=headers)
        assert downloaded.status_code == 200
        assert downloaded.content == TINY_PDF_BYTES
        assert downloaded.headers["content-type"] == "application/pdf"

        deleted = client.delete(f"/health-record/reports/{report_id}", headers=headers)
        assert deleted.status_code == 200
        assert deleted.json()["deleted"] is True
        assert client.get(f"/health-record/reports/{report_id}", headers=headers).status_code == 404
        assert client.get(f"/health-record/reports/{report_id}/download", headers=headers).status_code == 404
        assert client.get("/health-record/reports", headers=headers).json()["reports"] == []
    finally:
        _wipe_owner(persistence, owner)


def test_report_owner_isolation(authed_client_factory, persistence):
    owner_a = _unique_owner("repa")
    owner_b = _unique_owner("repb")
    try:
        client, private_key = authed_client_factory()
        headers_a = {"Authorization": f"Bearer {_token_for(private_key, owner_a)}"}
        headers_b = {"Authorization": f"Bearer {_token_for(private_key, owner_b)}"}

        files = {"file": ("private.pdf", TINY_PDF_BYTES, "application/pdf")}
        data = {"title": "Private Report", "document_type": "lab_report"}
        upload = client.post("/health-record/reports", data=data, files=files, headers=headers_a)
        report_id = upload.json()["report_id"]

        assert client.get(f"/health-record/reports/{report_id}", headers=headers_b).status_code == 404
        assert client.get(f"/health-record/reports/{report_id}/download", headers=headers_b).status_code == 404
        assert client.delete(f"/health-record/reports/{report_id}", headers=headers_b).status_code == 404

        # Owner A's report is untouched by B's failed attempts.
        assert client.get(f"/health-record/reports/{report_id}", headers=headers_a).status_code == 200
        assert client.get("/health-record/reports", headers=headers_b).json()["reports"] == []
    finally:
        _wipe_owner(persistence, owner_a)
        _wipe_owner(persistence, owner_b)


def test_report_rejects_unsupported_file_type(authed_client_factory, persistence):
    owner = _unique_owner("badtype")
    try:
        client, private_key = authed_client_factory()
        headers = {"Authorization": f"Bearer {_token_for(private_key, owner)}"}

        files = {"file": ("script.exe", b"not a real executable, just bytes", "application/x-msdownload")}
        data = {"title": "Malicious", "document_type": "other"}
        resp = client.post("/health-record/reports", data=data, files=files, headers=headers)
        assert resp.status_code == 415
        assert client.get("/health-record/reports", headers=headers).json()["reports"] == []
    finally:
        _wipe_owner(persistence, owner)


def test_report_rejects_oversized_file(authed_client_factory, persistence):
    from serving.v2.report_storage import MAX_REPORT_SIZE_BYTES

    owner = _unique_owner("bigfile")
    try:
        client, private_key = authed_client_factory()
        headers = {"Authorization": f"Bearer {_token_for(private_key, owner)}"}

        oversized = b"\x00" * (MAX_REPORT_SIZE_BYTES + 1)
        files = {"file": ("huge.png", oversized, "image/png")}
        data = {"title": "Too Big", "document_type": "other"}
        resp = client.post("/health-record/reports", data=data, files=files, headers=headers)
        assert resp.status_code == 413
        assert client.get("/health-record/reports", headers=headers).json()["reports"] == []
    finally:
        _wipe_owner(persistence, owner)


def test_local_report_storage_rejects_path_traversal(tmp_path):
    from serving.v2.report_storage import LocalReportStorage, ReportStorageError

    storage = LocalReportStorage(tmp_path / "uploads")
    for malicious_key in ("../../etc/passwd", "owner/../../secret.txt", "/etc/passwd", "a\\b"):
        with pytest.raises(ReportStorageError):
            storage.save(key=malicious_key, data=b"pwned")
    # Nothing was written anywhere by any of the rejected attempts.
    assert list((tmp_path / "uploads").iterdir()) == []


def test_sanitize_filename_strips_directories_and_unsafe_characters():
    from serving.v2.report_storage import sanitize_filename

    assert sanitize_filename("../../etc/passwd") == "passwd"
    assert sanitize_filename("normal report (final).pdf") == "normal_report__final_.pdf"
    assert sanitize_filename("") == "report"
    assert len(sanitize_filename("a" * 500 + ".pdf")) <= 120


# --------------------------------------------------------------------------
# Export and audit events
# --------------------------------------------------------------------------

def test_export_contains_every_section_without_report_binaries(authed_client_factory, persistence):
    owner = _unique_owner("export")
    try:
        client, private_key = authed_client_factory()
        headers = {"Authorization": f"Bearer {_token_for(private_key, owner)}"}

        client.put("/health-record/profile", json={"display_name_or_alias": "Export Test"}, headers=headers)
        client.post("/health-record/conditions", json={"name": "Diabetes", "status": "active"}, headers=headers)
        record = _create_record(client, headers, alias="Export Patient")
        client.post("/predict", json={"stay_id": record["stay_id"], "prediction_time": record["legal_cutoffs"][0]}, headers=headers)
        files = {"file": ("export-report.pdf", TINY_PDF_BYTES, "application/pdf")}
        client.post("/health-record/reports", data={"title": "Export Report", "document_type": "other"}, files=files, headers=headers)

        export = client.get("/health-record/export", headers=headers)
        assert export.status_code == 200
        body = export.json()
        assert body["profile"]["display_name_or_alias"] == "Export Test"
        assert len(body["conditions"]) == 1
        assert len(body["encounters"]) == 1
        assert body["encounters"][0]["observations"]
        assert len(body["reports"]) == 1
        assert len(body["prediction_history"]) == 1
        # No binary content anywhere in the export.
        assert "storage_key" not in body["reports"][0]
        assert b"%PDF" not in str(body).encode("utf-8", errors="ignore")
    finally:
        _wipe_owner(persistence, owner)


def test_audit_events_recorded_for_meaningful_mutations(authed_client_factory, persistence):
    owner = _unique_owner("audit")
    try:
        client, private_key = authed_client_factory()
        headers = {"Authorization": f"Bearer {_token_for(private_key, owner)}"}

        client.put("/health-record/profile", json={"display_name_or_alias": "Audit Test"}, headers=headers)
        created = client.post("/health-record/conditions", json={"name": "Asthma", "status": "active"}, headers=headers)
        condition_id = created.json()["condition_id"]
        client.patch(f"/health-record/conditions/{condition_id}", json={"status": "resolved"}, headers=headers)
        client.delete(f"/health-record/conditions/{condition_id}", headers=headers)
        _create_record(client, headers, alias="Audit Patient")
        files = {"file": ("audit-report.pdf", TINY_PDF_BYTES, "application/pdf")}
        upload = client.post("/health-record/reports", data={"title": "Audit Report", "document_type": "other"}, files=files, headers=headers)
        client.delete(f"/health-record/reports/{upload.json()['report_id']}", headers=headers)

        events = client.get("/health-record/audit-events", headers=headers).json()["events"]
        actions = [e["action"] for e in events]
        for expected in (
            "profile_updated", "condition_created", "condition_updated", "condition_deleted",
            "encounter_created", "report_uploaded", "report_deleted",
        ):
            assert expected in actions, f"missing audit action {expected!r} in {actions!r}"
        # No secrets or raw content leak into the audit trail.
        assert all("storage_key" not in e and "content" not in e for e in events)
    finally:
        _wipe_owner(persistence, owner)
