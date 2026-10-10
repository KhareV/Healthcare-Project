import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.research_router import build_research_router
from serving.v3 import research_data as rd

ROOT = Path(__file__).resolve().parents[1]
needs = pytest.mark.skipif(not (ROOT / "artifacts/performance_v3/phase4/final_metrics_v3.json").exists(), reason="frozen V3 phase-4 artifacts absent")
needs_pred = pytest.mark.skipif(not (ROOT / rd.PRED).exists(), reason="saved test predictions absent (gitignored/regenerable)")


@pytest.fixture(scope="module")
def client():
    app = FastAPI()
    app.include_router(build_research_router(ROOT))
    return TestClient(app)


@needs
def test_index_lists_every_dataset_with_status_and_hashes(client):
    idx = client.get("/research/index").json()
    assert idx["count"] == len(idx["datasets"]) >= 30
    for it in idx["datasets"]:
        assert it["evidence_status"] in {rd.STATUS_FINAL, rd.STATUS_DEV, rd.STATUS_HIST, rd.STATUS_DEFECT, rd.STATUS_PRIV}
        assert it["metric_definition"] and it["group"] and it["chart_hint"]
        for s in it["sources"]:
            if s["present"]:
                assert s["sha256"] == hashlib.sha256((ROOT / s["path"]).read_bytes()).hexdigest()


@needs
def test_every_indexed_dataset_resolves_with_rows_and_provenance(client):
    idx = client.get("/research/index").json()["datasets"]
    for it in idx:
        if not it["available"]:
            continue
        r = client.get(f"/research/datasets/{it['id']}")
        assert r.status_code == 200, (it["id"], r.text[:200])
        body = r.json()
        assert body["provenance"]["id"] == it["id"] and isinstance(body["data"]["rows"], list)
        assert len(r.content) < 2_500_000, it["id"]


@needs
def test_summary_numbers_equal_frozen_artifacts_exactly(client):
    s = client.get("/research/summary").json()
    fm = json.loads((ROOT / "artifacts/performance_v3/phase4/final_metrics_v3.json").read_text())["models"]["V3"]
    h = s["headline"]
    assert h["recovery24_mae"]["value"] == fm["recovery24"]["mae"] and round(h["recovery24_mae"]["value"], 4) == 1.0261
    assert round(h["icu_weighted_median_ae_hours"]["value"], 4) == 9.1011
    assert round(h["support_calibrated_auroc"]["value"], 4) == 0.7210 and round(h["support_calibrated_auprc"]["value"], 4) == 0.6238
    assert h["support_f1_at_frozen_threshold"]["threshold"] == 0.3021480061849302
    assert s["bundle"] == "V3" and s["evidence_status"] == rd.STATUS_FINAL
    assert "V2_AS_EVALUATED" in s["comparison"]["model_notes"] and "D1" in s["comparison"]["model_notes"]["V2_AS_EVALUATED"]


@needs
@needs_pred
def test_curve_auc_matches_frozen_metrics_and_downsampling_is_recorded(client):
    c = client.get("/research/curves/support").json()
    fm = json.loads((ROOT / "artifacts/performance_v3/phase4/final_metrics_v3.json").read_text())["models"]
    roc, pr = c["roc"]["data"], c["pr"]["data"]
    for m in ("V3", "V2_AS_SERVED", "V2_AS_EVALUATED", "CONTROL"):
        assert abs(roc["metric_by_model"][m] - fm[m]["organ_support"]["auroc"]) < 1e-3
        assert abs(pr["metric_by_model"][m] - fm[m]["organ_support"]["auprc"]) < 1e-3
        s = roc["series"][m]
        assert len(s["x"]) <= 500 and "downsampling" in s
        # AUC recomputed from the (possibly downsampled) points stays close
        assert abs(float(np.trapz(s["y"], s["x"])) - roc["metric_by_model"][m]) < 5e-3
    assert roc["n_rows"] == 5703 and roc["n_stays"] == 1167


@needs
@needs_pred
def test_confusion_and_threshold_sensitivity_match_frozen_threshold(client):
    c = client.get("/research/curves/support").json()
    fm = json.loads((ROOT / "artifacts/performance_v3/phase4/final_metrics_v3.json").read_text())
    rows = {r["model"]: r for r in c["confusion"]["data"]["rows"]}
    for m, r in rows.items():
        assert abs(r["f1"] - fm["models"][m]["organ_support"]["f1"]) < 1e-6
        assert r["threshold"] == fm["thresholds"][m]
    assert rows["V3"]["threshold"] == 0.3021480061849302
    ts = c["threshold_sensitivity"]["data"]
    frozen = [r for r in ts["rows"] if r["model"] == "V3" and r.get("frozen")][0]
    assert frozen["threshold"] == 0.3021480061849302 and abs(frozen["f1"] - fm["models"]["V3"]["organ_support"]["f1"]) < 1e-3
    assert "does not change" in ts["statement"]


@needs
@needs_pred
def test_regression_endpoints_have_units_and_denominators(client):
    r = client.get("/research/regression").json()
    res = r["residuals"]["data"]
    assert set(res["units"]) == {"recovery24", "recovery48", "icu_stay_time"} and res["units"]["icu_stay_time"] == "hours"
    n = {k: v["n_rows"] for k, v in res["summary"].items() if k.endswith(":V3")}
    assert n["recovery24:V3"] == 5130 and n["recovery48:V3"] == 1816 and n["icu_stay_time:V3"] == 9008
    assert r["error_slices"]["data"]["rows"] and r["paired"]["data"]["promotion"]


@needs
def test_defect_diagnostic_and_status_separation(client):
    d = client.get("/research/defect-d1").json()
    assert d["provenance"]["evidence_status"] == rd.STATUS_DEFECT
    modes = {(x["partition"], x["mode"], x["task"]) for x in d["data"]["rows"]}
    assert ("LCE (development)", "V2_AS_SERVED", "icu_stay_time") in modes
    g = client.get("/research/diagnostic").json()
    assert g["status"] == "PRIVILEGED_DIAGNOSTIC_NOT_DEPLOYABLE"
    rows = g["latent_state"]["data"]["rows"]
    assert all(r["deployable"] is False for r in rows) and any(r["model"].startswith("ORACLE") for r in rows)
    assert g["candidate_development"]["data"]["gate_c"]["passes_gate_c"] is False


@needs
def test_learning_curve_ablation_leaderboard_robustness_explain_provenance(client):
    for path in ("learning-curve", "ablation", "leaderboard", "robustness", "explainability/global", "provenance", "cohort"):
        r = client.get(f"/research/{path}")
        assert r.status_code == 200, path
    lc = client.get("/research/learning-curve").json()["data"]["rows"]
    assert {int(r["size_subjects"]) for r in lc} == {425, 850, 1700, 3800, 5900, 8000}
    lb = client.get("/research/leaderboard").json()["data"]
    assert lb["n_failed"] >= 0 and lb["n_trials"] == len(lb["rows"])
    prov = client.get("/research/provenance").json()["data"]
    assert prov["threshold"]["value"] == 0.3021480061849302 and prov["calibrator"]["method"] == "platt"
    assert set(prov["tasks"]) == {"recovery24", "recovery48", "icu_stay_time", "organ_support"}


@needs
def test_export_formats_and_missing_dataset(client):
    csv_text = client.get("/research/datasets/model_comparison/export?format=csv").text
    lines = [l for l in csv_text.splitlines() if not l.startswith("#")]
    assert lines[0].startswith("task,") and len(lines) >= 5
    tex = client.get("/research/datasets/paired_differences/export?format=tex").text
    assert r"\toprule" in tex and r"\begin{table}" in tex and r"\bottomrule" in tex
    assert client.get("/research/datasets/paired_differences/export?format=json").json()["provenance"]["evidence_status"] == rd.STATUS_FINAL
    assert client.get("/research/datasets/not_a_dataset").status_code == 404
    assert client.get("/research/datasets/model_comparison/export?format=xlsx").status_code == 422


def test_export_helpers_escape_and_roundtrip():
    rows = [{"a": 1, "b": "x_y & 50%"}, {"a": 2.5, "b": None}]
    assert rd.to_csv(rows).splitlines() == ["a,b", "1,x_y & 50%", "2.5,"]
    tex = rd.to_latex_table(rows, "Cap_1", "tab:t")
    assert r"x\_y \& 50\%" in tex and r"\caption{Cap\_1}" in tex


def test_missing_artifact_is_404_not_fabricated(tmp_path):
    app = FastAPI()
    app.include_router(build_research_router(tmp_path))
    c = TestClient(app)
    assert c.get("/research/summary").status_code == 404
    idx = c.get("/research/index").json()
    assert all(not d["available"] for d in idx["datasets"])
