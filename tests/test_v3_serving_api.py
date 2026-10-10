"""V3 product API + prediction-history versioning tests (network-free; Groq/Mongo are faked or absent)."""
import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

FULL = [{"concept": c, "hours_since_admission": h, "value": v} for h in (2, 26) for c, v in (
    ("heart_rate", 88), ("mean_arterial_pressure", 72), ("systolic_blood_pressure", 110), ("diastolic_blood_pressure", 65),
    ("respiratory_rate", 18), ("oxygen_saturation", 96), ("temperature", 37.1), ("pao2", 85), ("fio2", 0.3),
    ("creatinine", 1.0), ("platelet_count", 210), ("bilirubin_total", 0.7), ("glasgow_coma_scale", 15), ("lactate", 1.4))]


@pytest.fixture(scope="module")
def client():
    from api.product_app import build_product_app

    return TestClient(build_product_app(ROOT))


@pytest.fixture(scope="module")
def demo():
    return json.loads((ROOT / "configs/performance_v2/v2_demo_manifest_v1.json").read_text())["demo_subjects"][0]


def test_health_and_model_metadata_report_v3_only(client):
    h = client.get("/health").json()
    m = client.get("/model-metadata").json()
    assert h["release"] == m["release"] == "V3"
    assert m["support_threshold"] == 0.3021480061849302 and m["calibrator"]["method"] == "platt"
    assert {t: v["family"] for t, v in m["model_versions"].items()} == {"recovery24": "xgboost", "recovery48": "catboost", "icu_stay_time": "xgboost", "organ_support": "xgboost"}
    assert "6fed05f3" not in json.dumps(m)  # V2 selected-models hash never shown with V3


def test_predict_payload_is_v3_complete_with_four_explanations(client, demo):
    r = client.post("/predict", json={"stay_id": demo["stay_id"], "prediction_time": demo["legal_cutoffs"][4]})
    assert r.status_code == 200
    p = r.json()
    assert p["schema_version"] == "v3_prediction_response_v1" and p["release"] == "V3"
    assert p["model_bundle"]["release"] == "V3" and p["model_bundle"]["status"] == "CURRENT"
    assert p["organ_support"]["threshold"] == 0.3021480061849302
    assert set(p["explanations"]) == {"recovery24", "recovery48", "icu_stay_time", "organ_support"}
    for task, e in p["explanations"].items():
        assert e["diagnostics"]["additivity_check_passed"] is True, task
        assert e["method"] == "tree_shap" and e["diagnostics"]["causal_claim"] is False
        assert e["top_positive_contributors"] and e["by_feature_group"] and len(e["by_temporal_bin"]) == 8
        assert e["identity"] == {"stay_id": demo["stay_id"], "prediction_time": demo["legal_cutoffs"][4], "task": task}
    assert "raw model margin" in p["explanations"]["organ_support"]["diagnostics"]["explains"].lower()
    assert p["explanations"]["recovery48"]["diagnostics"]["model_family"] == "catboost"
    # independence of horizons, display clip only
    assert p["recovery"]["sofa_hat_48h"] == pytest.approx(min(24, max(0, p["current_sofa"] + p["recovery"]["delta_48h"])))
    assert "performance_v2" not in json.dumps(p) and p["data_revision"]


def test_unknown_sealed_and_illegal_requests_rejected(client, demo):
    assert client.post("/predict", json={"stay_id": "SYN-V3-T-E-00000001", "prediction_time": demo["legal_cutoffs"][0]}).status_code == 404
    assert client.post("/predict", json={"stay_id": demo["stay_id"], "prediction_time": "2100-01-01T00:00:00Z"}).status_code == 422


def test_performance_serves_v3_and_labels_v2_as_evaluated_compromised(client):
    p = client.get("/performance").json()
    final = json.loads((ROOT / "artifacts/performance_v3/phase4/final_metrics_v3.json").read_text())
    assert p["release"] == "V3" and p["final_metrics"]["recovery24"]["mae"] == final["models"]["V3"]["recovery24"]["mae"]
    assert p["comparators"]["V2_AS_EVALUATED"]["role"] == "HISTORICAL_COMPROMISED"
    assert p["comparators"]["V2_AS_SERVED"]["role"] == "operational comparison"
    assert p["paired_v2_v3"]["primary_comparator"] == "V2_AS_SERVED"
    assert "artifacts/performance_v2" not in json.dumps(p)


def test_demo_subjects_and_history(client, demo):
    d = client.get("/demo-subjects").json()
    assert d["release"] == "V3" and len(d["demo_subjects"]) == 3 and all(s["training_cohort_demo"] for s in d["demo_subjects"])
    t = demo["legal_cutoffs"][2]
    h = client.get("/history", params={"stay_id": demo["stay_id"], "prediction_time": t}).json()
    assert h["count"] > 0 and all(e["event_time"] <= t.replace("Z", ".000000Z") or e["event_time"][:19] <= t[:19] for e in h["events"])


def test_ai_endpoints_degrade_gracefully_without_groq_and_ground_on_v3(client, demo, monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    body = {"stay_id": demo["stay_id"], "prediction_time": demo["legal_cutoffs"][1]}
    assert client.post("/ai/recommendation", json=body).json()["status"] == "UNAVAILABLE"
    assert client.post("/assistant", json={**body, "question": "what changed?"}).status_code == 200
    from serving.v2.ai_recommendation import build_user_prompt

    payload = client.post("/predict", json=body).json()
    prompt = build_user_prompt(prediction=payload)
    assert "Model release: V3" in prompt and "raw output" in prompt and "XGBoost" not in prompt


# ---------------- prediction-history versioning (fake Mongo) ----------------

def test_prediction_run_versioning_never_overwrites_legacy_v2():
    from fake_mongo import fake_persistence

    p = fake_persistence()
    p._db.prediction_runs.insert_one({"owner_user_id": "u", "stay_id": "CUSTOM-1", "prediction_time": "t1", "current_sofa": 2.0, "remaining_icu_hours": 40.0})
    p.save_prediction_run(owner_user_id="u", stay_id="CUSTOM-1", prediction_time="t1", snapshot={"current_sofa": 2.0, "remaining_icu_hours": 22.0},
                          bundle_id="performance_v3_v3", release="V3", data_revision="r1")
    p.save_prediction_run(owner_user_id="u", stay_id="CUSTOM-1", prediction_time="t1", snapshot={"current_sofa": 2.0, "remaining_icu_hours": 23.0},
                          bundle_id="performance_v3_v3", release="V3", data_revision="r1")  # same bundle -> upsert, not duplicate
    runs = p.list_prediction_runs(owner_user_id="u", stay_id="CUSTOM-1")
    assert len(runs) == 2
    legacy = next(r for r in runs if r["release"] == "V2")
    v3 = next(r for r in runs if r["release"] == "V3")
    assert legacy["bundle_id"] == "performance_v2_frozen" and legacy["legacy"] is True and legacy["remaining_icu_hours"] == 40.0
    assert v3["bundle_id"] == "performance_v3_v3" and v3["legacy"] is False and v3["remaining_icu_hours"] == 23.0 and v3["data_revision"] == "r1"
    # a legacy-style (V2 app) save still only touches legacy documents
    p.save_prediction_run(owner_user_id="u", stay_id="CUSTOM-1", prediction_time="t1", snapshot={"remaining_icu_hours": 41.0})
    runs = p.list_prediction_runs(owner_user_id="u", stay_id="CUSTOM-1")
    assert {r["release"]: r["remaining_icu_hours"] for r in runs} == {"V2": 41.0, "V3": 23.0}


def test_confirmed_report_measurement_marks_stored_predictions_stale(monkeypatch):
    from fake_mongo import fake_persistence
    import api.v2_app as v2_app
    from api.product_app import build_product_app

    fake = fake_persistence()
    monkeypatch.setattr(v2_app, "MongoPersistence", lambda uri=None: fake)
    c = TestClient(build_product_app(ROOT))
    rec = c.post("/custom-records", json={"patient_alias": "Stale Check", "age_years": 61, "sex_category": "MALE", "observations": FULL}).json()
    sid = rec["stay_id"]; t = rec["legal_cutoffs"][0]
    first = c.post("/predict", json={"stay_id": sid, "prediction_time": t}).json()
    runs = c.get(f"/custom-records/{sid}/predictions").json()["predictions"]
    assert len(runs) == 1 and runs[0]["release"] == "V3" and runs[0]["stale"] is False and runs[0]["data_revision"] == first["data_revision"]

    from serving.v2.custom_record import append_observations_to_encounter

    runtime = c.app.state.v2_runtime
    owner = runtime.ephemeral_owner(sid)
    append_observations_to_encounter(runtime, fake, owner_user_id=owner, stay_id=sid,
                                     observations=[{"concept": "lactate", "hours_since_admission": 3, "value": 3.9}])
    runs = c.get(f"/custom-records/{sid}/predictions").json()["predictions"]
    assert runs[0]["stale"] is True and runs[0]["current_data_revision"] != runs[0]["data_revision"]
    again = c.post("/predict", json={"stay_id": sid, "prediction_time": t}).json()
    runs = c.get(f"/custom-records/{sid}/predictions").json()["predictions"]
    assert again["data_revision"] != first["data_revision"]
    assert len(runs) == 2 and sorted(r["stale"] for r in runs) == [False, True]
