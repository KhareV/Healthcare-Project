"""Focused tests for the V2 real, model-backed serving stack. Never
triggers fresh-V2-test inference; never mutates any frozen Phase-3/4
scientific artifact."""

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))


def _load(path):
    return json.loads((ROOT / path).read_text())


# --- V2 serving-bundle / model hashes ----------------------------------------

def test_v2_serving_bundle_frozen_and_binds_scientific_hashes():
    bundle = _load("artifacts/performance_v2/serving/v2_dashboard_serving_bundle_v1.json")
    assert bundle["status"] == "V2_DASHBOARD_SERVING_FROZEN"
    assert bundle["parents"]["selected_models_v2"]["sha256"] == "6fed05f3c71ce648658b2f864e69fa9f80f3ec8538f53a4e370fa8a4eaf176a8"
    assert bundle["parents"]["v2_model_freeze"]["sha256"] == "f5feb29cab2e25d6e8bff860af52ac505f40951578aa73547ec420ba8fcda570"


def test_selected_models_v2_loads_and_recovery48_is_b_min():
    sm = _load("artifacts/performance_v2/phase3/selected_models_v2.json")
    assert sm["tasks"]["recovery48"]["feature_variant"] == "B_MIN"
    assert sm["tasks"]["recovery24"]["feature_variant"] == "B_MIN"
    assert sm["tasks"]["icu_stay_time"]["feature_variant"] == "B_PLUS_F"
    assert sm["tasks"]["organ_support"]["feature_variant"] == "B_FULL"


@pytest.fixture(scope="module")
def runtime():
    from serving.v2.runtime import V2ServingRuntime

    return V2ServingRuntime(ROOT)


@pytest.fixture(scope="module")
def api_app():
    from api.v2_app import build_v2_app

    return build_v2_app(ROOT)


@pytest.fixture(scope="module")
def api_client(api_app):
    from fastapi.testclient import TestClient

    return TestClient(api_app)


def test_v2_feature_variant_per_task(runtime):
    from performance_v2.final_evaluation import TASK_VARIANT

    assert TASK_VARIANT == {"recovery24": "B_MIN", "recovery48": "B_MIN", "icu_stay_time": "B_PLUS_F", "organ_support": "B_FULL"}


def test_runtime_hash_verification_fails_closed(tmp_path):
    from serving.v2.runtime import V2ServingError, V2ServingRuntime

    fake_root = tmp_path
    (fake_root / "artifacts/performance_v2/phase3/models").mkdir(parents=True)
    (fake_root / "artifacts/performance_v2/phase3/models/recovery24_xgb_v2.json").write_text("not a real model")
    with pytest.raises((V2ServingError, FileNotFoundError, OSError)):
        V2ServingRuntime(fake_root)


# --- Demo manifest / guard ---------------------------------------------------

def test_demo_manifest_frozen_and_structural():
    manifest = _load("configs/performance_v2/v2_demo_manifest_v1.json")
    assert manifest["status"] == "V2_DEMO_MANIFEST_FROZEN"
    assert manifest["selection_criteria"]["selection_independent_of_model_performance"] is True
    assert manifest["n_demo_subjects"] >= 3
    for subject in manifest["demo_subjects"]:
        assert subject["n_legal_cutoffs"] >= 5
        assert subject["subject_id"].startswith("SYN-S-")  # DEV TRAIN population, never fresh-test


def test_demo_subjects_span_distinct_cardiac_subtypes():
    manifest = _load("configs/performance_v2/v2_demo_manifest_v1.json")
    subtypes = {s["cardiac_condition_group"] for s in manifest["demo_subjects"]}
    assert len(subtypes) >= 3


def test_fresh_test_stay_rejected_by_guard(runtime):
    from serving.v2.guard import UnknownDemoStayError

    with pytest.raises(UnknownDemoStayError):
        runtime.predict("SYN-V2-E-00000001", "2101-01-01T00:00:00Z")


def test_unknown_stay_rejected_identically(runtime):
    from serving.v2.guard import UnknownDemoStayError

    with pytest.raises(UnknownDemoStayError):
        runtime.predict("SYN-E-99999999", "2101-01-01T00:00:00Z")


def test_illegal_cutoff_rejected(runtime):
    from serving.v2.runtime import IllegalCutoffError

    with pytest.raises(IllegalCutoffError):
        runtime.predict("SYN-E-00000001", "2099-01-01T00:00:00Z")


# --- History truncation / feature equivalence --------------------------------

def test_history_truncation_matches_offline_batch_features(runtime):
    from performance_v2.data_loading import load_dev_rows
    from performance_v2.final_evaluation import build_inference_matrix, load_frozen_normalization, reconstruct_group_a_names

    train = load_dev_rows(ROOT, splits=("train",))
    target = sorted([r for r in train if r["stay_id"] == "SYN-E-00000001"], key=lambda r: r["grid_index"])[0]
    names = reconstruct_group_a_names(ROOT, "recovery24")
    norm = load_frozen_normalization(ROOT, "B_MIN")
    offline_matrix = build_inference_matrix([target], group_a_names=names, normalization=norm)

    cutoffs = runtime.legal_cutoffs("SYN-E-00000001")
    online = runtime.predict("SYN-E-00000001", cutoffs[0])
    assert np.allclose(offline_matrix, online.matrices["recovery24"], atol=1e-6, equal_nan=True)


def test_future_row_perturbation_does_not_change_earlier_features(runtime):
    """Truncation must ignore events strictly after the cutoff."""
    cutoffs = runtime.legal_cutoffs("SYN-E-00000001")
    early_cutoff = cutoffs[0]
    before = runtime.predict("SYN-E-00000001", early_cutoff)

    # Inject a synthetic future event far after the early cutoff.
    future_time = (datetime.fromisoformat(cutoffs[-1].replace("Z", "+00:00")) + timedelta(hours=100)).isoformat().replace("+00:00", "Z")
    injected = dict(runtime._events_by_stay["SYN-E-00000001"][0])
    injected = {**injected, "event_time": future_time, "event_id": injected["event_id"] + "-INJECTED-TEST"}
    runtime._events_by_stay["SYN-E-00000001"] = list(runtime._events_by_stay["SYN-E-00000001"]) + [injected]
    try:
        after = runtime.predict("SYN-E-00000001", early_cutoff)
        assert np.array_equal(before.matrices["recovery24"], after.matrices["recovery24"], equal_nan=True)
        assert before.current_sofa == after.current_sofa
    finally:
        runtime._events_by_stay["SYN-E-00000001"] = [r for r in runtime._events_by_stay["SYN-E-00000001"] if r.get("event_id") != injected["event_id"]]


def test_current_sofa_equality_across_repeated_calls(runtime):
    cutoffs = runtime.legal_cutoffs("SYN-E-00000001")
    a = runtime.predict("SYN-E-00000001", cutoffs[1])
    b = runtime.predict("SYN-E-00000001", cutoffs[1])
    assert a.current_sofa == b.current_sofa
    assert a.recovery24_delta == b.recovery24_delta


# --- Recovery reconstruction --------------------------------------------------

def test_independent_recovery_reconstruction_no_chaining(runtime):
    cutoffs = runtime.legal_cutoffs("SYN-E-00000001")
    pred = runtime.predict("SYN-E-00000001", cutoffs[2])
    expected_24 = max(0.0, min(24.0, pred.current_sofa + pred.recovery24_delta))
    expected_48 = max(0.0, min(24.0, pred.current_sofa + pred.recovery48_delta))
    assert pred.recovery24_sofa == pytest.approx(expected_24)
    assert pred.recovery48_sofa == pytest.approx(expected_48)
    # explicit non-chaining: 48h must NOT be reconstructed from the 24h prediction
    wrong_chained = max(0.0, min(24.0, pred.recovery24_sofa + pred.recovery48_delta))
    if pred.recovery24_delta != 0:
        assert pred.recovery48_sofa != pytest.approx(wrong_chained) or pred.recovery24_sofa == pred.current_sofa


def test_recovery_clip_bounds(runtime):
    cutoffs = runtime.legal_cutoffs("SYN-E-00000001")
    pred = runtime.predict("SYN-E-00000001", cutoffs[0])
    assert 0.0 <= pred.recovery24_sofa <= 24.0
    assert 0.0 <= pred.recovery48_sofa <= 24.0


# --- ICU postprocessing -------------------------------------------------------

def test_icu_hours_postprocess_matches_frozen_formula(runtime):
    cutoffs = runtime.legal_cutoffs("SYN-E-00000001")
    pred = runtime.predict("SYN-E-00000001", cutoffs[0])
    expected = np.expm1(max(0.0, pred.icu_raw_log_prediction))
    assert pred.icu_remaining_hours == pytest.approx(expected)
    assert pred.icu_remaining_hours >= 0.0


# --- Support: raw -> isotonic -> threshold -----------------------------------

def test_support_raw_to_calibrated_to_threshold(runtime):
    cutoffs = runtime.legal_cutoffs("SYN-E-00000001")
    pred = runtime.predict("SYN-E-00000001", cutoffs[0])
    assert 0.0 <= pred.support_raw_probability <= 1.0
    assert 0.0 <= pred.support_calibrated_probability <= 1.0
    assert pred.support_threshold == pytest.approx(0.39781983118092895)
    assert pred.support_alert == (pred.support_calibrated_probability >= pred.support_threshold)


# --- TreeSHAP -----------------------------------------------------------------

@pytest.mark.parametrize("task", ["recovery24", "recovery48", "icu_stay_time", "organ_support"])
def test_tree_shap_routing_for_all_v2_tasks(runtime, task):
    from serving.v2.explanations import explain

    cutoffs = runtime.legal_cutoffs("SYN-E-00000001")
    pred = runtime.predict("SYN-E-00000001", cutoffs[2])
    positive, negative, diagnostics = explain(runtime.models[task], pred.matrices[task], pred.feature_names[task])
    assert diagnostics["additivity_check_passed"] is True
    assert isinstance(positive, list) and isinstance(negative, list)


def test_explanation_feature_name_mapping_is_readable(runtime):
    from serving.v2.explanations import explain

    cutoffs = runtime.legal_cutoffs("SYN-E-00000001")
    pred = runtime.predict("SYN-E-00000001", cutoffs[2])
    positive, negative, _ = explain(runtime.models["recovery24"], pred.matrices["recovery24"], pred.feature_names["recovery24"])
    for item in positive + negative:
        assert not item.readable_label.startswith("f") or not item.readable_label[1:].isdigit()
        assert item.readable_label  # never empty


# --- API / direct equality ----------------------------------------------------

def test_api_matches_direct_pipeline_exactly(runtime, api_client):
    cutoffs = runtime.legal_cutoffs("SYN-E-00000001")
    direct = runtime.predict("SYN-E-00000001", cutoffs[1])
    response = api_client.post("/predict", json={"stay_id": "SYN-E-00000001", "prediction_time": cutoffs[1]})
    assert response.status_code == 200
    body = response.json()
    assert body["recovery"]["delta_24h"] == pytest.approx(direct.recovery24_delta)
    assert body["recovery"]["delta_48h"] == pytest.approx(direct.recovery48_delta)
    assert body["icu_stay_time"]["remaining_hours"] == pytest.approx(direct.icu_remaining_hours)
    assert body["organ_support"]["probability_24h"] == pytest.approx(direct.support_calibrated_probability)


def test_repeated_api_calls_deterministic(runtime, api_client):
    cutoffs = runtime.legal_cutoffs("SYN-E-00000001")
    a = api_client.post("/predict", json={"stay_id": "SYN-E-00000001", "prediction_time": cutoffs[0]}).json()
    b = api_client.post("/predict", json={"stay_id": "SYN-E-00000001", "prediction_time": cutoffs[0]}).json()
    assert a["recovery"]["delta_24h"] == b["recovery"]["delta_24h"]


def test_unknown_stay_is_404(api_client):
    response = api_client.post("/predict", json={"stay_id": "SYN-E-99999999", "prediction_time": "2101-01-01T00:00:00Z"})
    assert response.status_code == 404


def test_fresh_test_stay_via_api_is_404_like_unknown(api_client):
    response = api_client.post("/predict", json={"stay_id": "SYN-V2-E-00000001", "prediction_time": "2101-01-01T00:00:00Z"})
    assert response.status_code == 404


def test_illegal_cutoff_is_422(api_client):
    response = api_client.post("/predict", json={"stay_id": "SYN-E-00000001", "prediction_time": "2099-01-01T00:00:00Z"})
    assert response.status_code == 422


def test_health_endpoint(api_client):
    response = api_client.get("/health")
    assert response.status_code == 200
    assert response.json()["mode"] == "RETROSPECTIVE_SEQUENTIAL_REPLAY"


# --- No fresh-V2-test model inference anywhere in this file -----------------

def test_this_module_never_touches_fresh_test_access_state():
    state = _load("artifacts/performance_v2/governance/v2_fresh_test_access_state.json")
    assert state["status"] == "FINAL_V2_RUN_COMPLETED"
    assert len(state["history"]) == 3  # unchanged by any V2 serving/dashboard test
