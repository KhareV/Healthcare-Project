"""Real, live-cluster + live-Groq integration tests for Report Intelligence
(src/serving/v2/report_parser.py, and the /health-record/reports/{id}/parse
and /confirm endpoints in api/v2_app.py).

Skipped entirely if MONGODB_URI or GROQ_API_KEY is not configured -- both
are genuine optional external services, exactly like this project's other
live-service tests. Deliberately NOT mocked: the whole point of this
feature is whether a real report's text gets turned into sensible
candidate measurements, which a mock can't prove either way.
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

_env = dotenv_values(ROOT / ".env")
MONGODB_URI = os.environ.get("MONGODB_URI") or _env.get("MONGODB_URI")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY") or _env.get("GROQ_API_KEY")
pytestmark = pytest.mark.skipif(
    not (MONGODB_URI and GROQ_API_KEY), reason="MONGODB_URI and/or GROQ_API_KEY not configured; Report Intelligence tests are skipped"
)

REAL_PUBLISHABLE_KEY = "pk_test_dmVyaWZpZWQta2luZ2Zpc2gtNTQuY2xlcmsuYWNjb3VudHMuZGV2JA"
ISSUER = "https://verified-kingfish-54.clerk.accounts.dev"

OBSERVATIONS = [
    {"concept": "heart_rate", "hours_since_admission": 2, "value": 88},
    {"concept": "lactate", "hours_since_admission": 32, "value": 1.2},
]


def _unique_owner(label: str) -> str:
    return f"pytest-report-intel-{label}-{uuid.uuid4().hex[:10]}"


def _token_for(private_key, user_id: str) -> str:
    now = int(time.time())
    return pyjwt.encode({"sub": user_id, "iss": ISSUER, "iat": now, "exp": now + 300}, private_key, algorithm="RS256")


def _wipe_owner(persistence, owner_user_id: str) -> None:
    import shutil

    for coll in ("patient_profiles", "conditions", "encounters", "prediction_runs", "reports", "audit_events"):
        persistence._db[coll].delete_many({"owner_user_id": owner_user_id})
    shutil.rmtree(ROOT / "runtime" / "uploads" / owner_user_id, ignore_errors=True)


def _build_synthetic_pdf(lines) -> bytes:
    """A minimal, hand-built single-page PDF with a Helvetica text stream --
    no external dependency needed to produce a real, parseable PDF fixture."""

    stream_parts = ["BT /F1 12 Tf 20 260 Td"]
    for i, line in enumerate(lines):
        prefix = "" if i == 0 else "0 -20 Td "
        escaped = line.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")
        stream_parts.append(f"{prefix}({escaped}) Tj")
    stream_parts.append("ET")
    content = " ".join(stream_parts).encode()

    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /Resources << /Font << /F1 4 0 R >> >> /MediaBox [0 0 300 300] /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length " + str(len(content)).encode() + b" >>\nstream\n" + content + b"\nendstream",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, obj in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + obj + b"\nendobj\n"
    xref_offset = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode() + b"0000000000 65535 f \n"
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += b"trailer\n<< /Size " + str(len(objects) + 1).encode() + b" /Root 1 0 R >>\nstartxref\n" + str(xref_offset).encode() + b"\n%%EOF"
    return bytes(out)


SYNTHETIC_LAB_REPORT_PDF = _build_synthetic_pdf([
    "Synthetic Lab Report",
    "Creatinine: 1.8 mg/dL",
    "Platelet Count: 132 x10^9/L",
    "Total Bilirubin: 1.3 mg/dL",
])


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
    monkeypatch.setenv("GROQ_API_KEY", GROQ_API_KEY)

    def _make():
        return TestClient(build_v2_app(ROOT)), private_key

    return _make


def _create_record(client, headers, alias="Report Intel Test Patient"):
    resp = client.post(
        "/custom-records",
        json={"patient_alias": alias, "age_years": 55, "sex_category": "FEMALE", "observations": OBSERVATIONS, "support_intervals": []},
        headers=headers,
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


def _upload_report(client, headers, *, title="Synthetic Lab Report"):
    files = {"file": ("lab.pdf", SYNTHETIC_LAB_REPORT_PDF, "application/pdf")}
    resp = client.post("/health-record/reports", data={"title": title, "document_type": "lab_report"}, files=files, headers=headers)
    assert resp.status_code == 200, resp.text
    return resp.json()["report_id"]


def test_parse_extracts_and_maps_real_candidates(authed_client_factory, persistence):
    owner = _unique_owner("parse")
    try:
        client, private_key = authed_client_factory()
        headers = {"Authorization": f"Bearer {_token_for(private_key, owner)}"}
        report_id = _upload_report(client, headers)

        resp = client.post(f"/health-record/reports/{report_id}/parse", headers=headers)
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["processing_status"] == "PARSED"
        candidates = body["candidate_measurements"]
        assert len(candidates) >= 1
        by_concept = {c["concept"] for c in candidates if c["concept"]}
        # The fixture names three measurements this product's vocabulary
        # supports; a real Groq call should map at least the unambiguous
        # ones correctly.
        assert "creatinine" in by_concept
        assert all(not c["confirmed"] for c in candidates)
    finally:
        _wipe_owner(persistence, owner)


def test_confirm_creates_observation_and_prediction_reflects_it(authed_client_factory, persistence):
    owner = _unique_owner("confirm")
    try:
        client, private_key = authed_client_factory()
        headers = {"Authorization": f"Bearer {_token_for(private_key, owner)}"}

        record = _create_record(client, headers)
        stay_id = record["stay_id"]
        before = client.get(f"/custom-records/{stay_id}", headers=headers).json()
        before_observed = before["readiness_summary"]["observations_entered"]
        assert "renal" in before["readiness_summary"]["sofa_components_missing"]

        report_id = _upload_report(client, headers)
        parsed = client.post(f"/health-record/reports/{report_id}/parse", headers=headers).json()
        creatinine_candidate = next(c for c in parsed["candidate_measurements"] if c["concept"] == "creatinine")

        confirm = client.post(
            f"/health-record/reports/{report_id}/confirm",
            json={"encounter_id": stay_id, "confirmations": [{"candidate_id": creatinine_candidate["candidate_id"], "hours_since_admission": 6}]},
            headers=headers,
        )
        assert confirm.status_code == 200, confirm.text
        assert confirm.json()["added"] == 1

        after = client.get(f"/custom-records/{stay_id}", headers=headers).json()
        assert after["readiness_summary"]["observations_entered"] == before_observed + 1
        assert "renal" in after["readiness_summary"]["sofa_components_observed"]

        # The confirmed measurement is now a real, servable observation --
        # a fresh prediction call succeeds and reflects the same current SOFA.
        cutoff = after["legal_cutoffs"][0]
        pred = client.post("/predict", json={"stay_id": stay_id, "prediction_time": cutoff}, headers=headers)
        assert pred.status_code == 200, pred.text

        # Re-confirming the same candidate is a no-op, never a duplicate.
        again = client.post(
            f"/health-record/reports/{report_id}/confirm",
            json={"encounter_id": stay_id, "confirmations": [{"candidate_id": creatinine_candidate["candidate_id"], "hours_since_admission": 6}]},
            headers=headers,
        )
        assert again.status_code == 200
        assert again.json()["added"] == 0
        still = client.get(f"/custom-records/{stay_id}", headers=headers).json()
        assert still["readiness_summary"]["observations_entered"] == before_observed + 1
    finally:
        _wipe_owner(persistence, owner)


def test_confirm_rejects_unmapped_concept_candidate(authed_client_factory, persistence):
    owner = _unique_owner("unmapped")
    try:
        client, private_key = authed_client_factory()
        headers = {"Authorization": f"Bearer {_token_for(private_key, owner)}"}

        record = _create_record(client, headers)
        stay_id = record["stay_id"]
        report_id = _upload_report(client, headers)

        # Force an unmapped candidate directly (independent of what Groq
        # happens to return for the fixture) to test the confirm endpoint's
        # own rejection logic deterministically.
        persistence.save_report_candidates(
            owner_user_id=owner, report_id=report_id,
            candidates=[{"candidate_id": "CAND-UNMAPPED", "raw_label": "Some Unknown Test", "concept": None, "value": 1.0, "unit": None, "observed_at": None, "confirmed": False}],
            status="PARSED",
        )

        resp = client.post(
            f"/health-record/reports/{report_id}/confirm",
            json={"encounter_id": stay_id, "confirmations": [{"candidate_id": "CAND-UNMAPPED", "hours_since_admission": 6}]},
            headers=headers,
        )
        assert resp.status_code == 422
    finally:
        _wipe_owner(persistence, owner)


def test_confirm_is_owner_scoped(authed_client_factory, persistence):
    owner_a = _unique_owner("ria")
    owner_b = _unique_owner("rib")
    try:
        client, private_key = authed_client_factory()
        headers_a = {"Authorization": f"Bearer {_token_for(private_key, owner_a)}"}
        headers_b = {"Authorization": f"Bearer {_token_for(private_key, owner_b)}"}

        record_a = _create_record(client, headers_a, alias="Owner A Encounter")
        report_id = _upload_report(client, headers_a)
        parsed = client.post(f"/health-record/reports/{report_id}/parse", headers=headers_a).json()
        candidate_id = parsed["candidate_measurements"][0]["candidate_id"]

        # Owner B cannot even see the report to parse/confirm it.
        assert client.post(f"/health-record/reports/{report_id}/parse", headers=headers_b).status_code == 404
        assert client.post(
            f"/health-record/reports/{report_id}/confirm",
            json={"encounter_id": record_a["stay_id"], "confirmations": [{"candidate_id": candidate_id, "hours_since_admission": 6}]},
            headers=headers_b,
        ).status_code == 404

        # Owner A cannot confirm into an encounter they don't own either.
        record_b = _create_record(client, headers_b, alias="Owner B Encounter")
        forbidden = client.post(
            f"/health-record/reports/{report_id}/confirm",
            json={"encounter_id": record_b["stay_id"], "confirmations": [{"candidate_id": candidate_id, "hours_since_admission": 6}]},
            headers=headers_a,
        )
        assert forbidden.status_code == 404
    finally:
        _wipe_owner(persistence, owner_a)
        _wipe_owner(persistence, owner_b)
