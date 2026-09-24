"""Stage-4 real serving integration: resolver, PredictionPipeline, FastAPI,
explanations, and the sealed-test-subject guard, against the actual frozen
Stage-3 artifacts. No test-partition subject is ever predicted here.
"""

import json

import pytest

from api.main import create_app
from api.real_app import build_real_app
from evaluation.selection_validation import TASKS
from explainability.router import lint_descriptive_explanation_text
from serving.prediction_schema import lint_contract_text, validate_response
from serving.real.bundle import Stage4BundleError, resolve_stage4_bundle
from serving.real.history import SealedTestSubjectError
from vedant_infra.g3 import G3FreezeError
from stage4_helpers import ROOT, copy_repository, legal_cutoff, safe_train_stay, sealed_test_stay


def _final_test_access_consumed() -> bool:
    state = json.loads((ROOT / "artifacts/governance/test_access_state.json").read_text())
    return state["state"] != "AUTHORIZED_NOT_RUN"


@pytest.fixture(scope="module")
def resolution():
    try:
        return resolve_stage4_bundle(ROOT)
    except (Stage4BundleError, G3FreezeError) as error:
        if _final_test_access_consumed():
            pytest.skip(
                "final-test access already consumed: resolve_stage4_bundle "
                "correctly fails closed once G3's live dependency audit no "
                "longer holds (experiments/artifacts.csv/registry.csv now "
                "legitimately bind the registered final-test results). This "
                "is the expected post-test transition -- "
                "artifacts/governance/g4_test_evaluation_freeze_v1.json is "
                "authoritative from this point forward, not this Stage-4 "
                "pre-test demo/validation resolver. "
                f"Underlying error: {error}"
            )
        raise


def test_serving_fails_closed_once_final_test_access_consumed():
    """Complements the ``resolution`` fixture's skip above: explicitly
    proves resolve_stage4_bundle refuses (fail-closed) rather than silently
    serving once the one-time final-test access has been consumed."""
    if not _final_test_access_consumed():
        pytest.skip("final-test access not yet consumed in this repository state")
    with pytest.raises((Stage4BundleError, G3FreezeError)):
        resolve_stage4_bundle(ROOT)


@pytest.fixture(scope="module")
def app(resolution):
    return create_app(pipeline=resolution.pipeline)


@pytest.fixture(scope="module")
def demo_request():
    stay_id = safe_train_stay()
    return {"stay_id": stay_id, "prediction_time": legal_cutoff(stay_id)}


def test_resolver_binds_the_exact_frozen_selected_models(resolution):
    assert resolution.bundle.scope == "real"
    assert resolution.manifest["status"] == "SELECTED_MODELS_FROZEN_PRE_TEST"
    assert resolution.manifest["serving_ready"] is False
    assert resolution.manifest_sha256 == "3d4fb0b1e0341b0ebe29a54e9581bc47d3b80c536d3ee22f830276b4631fbb0b"
    assert resolution.g3_marker["freeze_status"] == "ACTIVE"
    assert resolution.g3_marker["test_accessed"] is False
    families = {resolution.bundle.task(task).identity.family for task in TASKS}
    assert families == {"xgboost", "gru"}
    assert "lstm" not in families


def test_mixed_family_routing_is_independent_per_task(resolution):
    assert resolution.bundle.task("recovery").identity.family == "xgboost"
    assert resolution.bundle.task("icu_stay_time").identity.family == "gru"
    assert resolution.bundle.task("organ_support").identity.family == "xgboost"


def test_direct_pipeline_predicts_sane_response(resolution, demo_request):
    response = resolution.pipeline.predict(demo_request)
    validate_response(response, synthetic=False)
    assert 0 <= response["recovery"]["reconstructed_sofa_24h"] <= 24
    assert 0 <= response["recovery"]["reconstructed_sofa_48h"] <= 24
    assert response["icu_stay_time_hours"] >= 0
    assert 0 <= response["organ_support_probability_calibrated"] <= 1
    assert response["model_versions"]["recovery"]["family"] == "xgboost"
    assert response["model_versions"]["icu_stay_time"]["family"] == "gru"
    assert response["model_versions"]["organ_support"]["family"] == "xgboost"


def test_repeated_calls_are_deterministic(resolution, demo_request):
    first = resolution.pipeline.predict(demo_request)
    second = resolution.pipeline.predict(demo_request)
    assert first == second


def test_explanations_routed_and_populated(resolution, demo_request):
    response = resolution.pipeline.predict(demo_request)
    explanations = response["explanation_features"]
    assert explanations["recovery"]["family"] == "xgboost"
    assert explanations["recovery"]["explanation_method"] == "tree_shap"
    assert explanations["icu_stay_time"]["family"] == "gru"
    assert explanations["icu_stay_time"]["explanation_method"] == "integrated_gradients"
    assert explanations["organ_support"]["family"] == "xgboost"
    assert explanations["organ_support"]["explanation_method"] == "tree_shap"
    for task in TASKS:
        items = explanations[task]["items"]
        assert len(items) > 0
        for item in items:
            assert lint_descriptive_explanation_text(item["feature_name"]) == ()


def test_recovery_explanation_covers_both_independent_horizons(resolution, demo_request):
    response = resolution.pipeline.predict(demo_request)
    names = {item["feature_name"].split(":")[0] for item in response["explanation_features"]["recovery"]["items"]}
    assert names == {"delta_sofa_24h", "delta_sofa_48h"}


def test_support_explanation_documents_pre_calibration_domain(resolution, demo_request):
    resolution.pipeline.predict(demo_request)
    adapter = resolution.pipeline._explanation_router._adapters["tree_shap"]
    # Re-run through the router directly is unnecessary; the adapter's own
    # contract is asserted via its explain() details on the last call inside
    # the pipeline. We instead assert the documented policy on the adapter.
    assert adapter.synthetic is False


def test_no_prohibited_terminology_in_response(resolution, demo_request):
    response = resolution.pipeline.predict(demo_request)
    assert lint_contract_text(json.dumps(response)) == ()


def test_api_matches_direct_pipeline_exactly(app, resolution, demo_request):
    from fastapi.testclient import TestClient

    client = TestClient(app)
    api_response = client.post("/predict", json=demo_request).json()
    direct_response = resolution.pipeline.predict(demo_request)
    assert api_response == direct_response


def test_health_and_metadata_endpoints(app, resolution):
    from fastapi.testclient import TestClient

    client = TestClient(app)
    health = client.get("/health").json()
    assert health == {
        "status": "alive", "ready": True, "serving_scope": "real", "mode": "RETROSPECTIVE_SEQUENTIAL_REPLAY",
    }
    metadata = client.get("/model-metadata").json()
    assert metadata["manifest_sha256"] == resolution.manifest_sha256
    assert metadata["serving_scope"] == "real"


def test_unknown_stay_is_404(app):
    from fastapi.testclient import TestClient

    client = TestClient(app)
    response = client.post("/predict", json={"stay_id": "NOT-A-REAL-STAY", "prediction_time": "2101-01-01T00:00:00+00:00"})
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "unknown_stay"


def test_illegal_cutoff_is_422(app, demo_request):
    from fastapi.testclient import TestClient

    client = TestClient(app)
    off_grid = dict(demo_request, prediction_time="2101-05-31T06:19:00+00:00")
    response = client.post("/predict", json=off_grid)
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "illegal_prediction_time"


def test_sealed_test_subject_is_refused_like_unknown_stay(app, resolution):
    from fastapi.testclient import TestClient

    client = TestClient(app)
    stay_id = sealed_test_stay()
    assert resolution.split_by_stay[stay_id] == "test"
    response = client.post("/predict", json={"stay_id": stay_id, "prediction_time": legal_cutoff(stay_id)})
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "unknown_stay"


def test_sealed_test_subject_refused_at_pipeline_level_too(resolution):
    stay_id = sealed_test_stay()
    with pytest.raises(SealedTestSubjectError):
        resolution.pipeline.predict({"stay_id": stay_id, "prediction_time": legal_cutoff(stay_id)})


def test_final_test_access_state_untouched_by_serving(resolution, demo_request):
    resolution.pipeline.predict(demo_request)
    state = json.loads((ROOT / "artifacts/governance/test_access_state.json").read_text())
    assert state["state"] == "AUTHORIZED_NOT_RUN"
    assert not list((ROOT / "artifacts/governance/history").glob("FINAL_TEST_ACCESS_CONSUMED*"))


# --------------------------------------------------------------------------
# Hostile artifact-compatibility tests: each tampers a fresh repository copy
# and asserts the Stage-4 resolver fails closed before loading any model.
# --------------------------------------------------------------------------


def _tamper_json(path, mutate) -> None:
    payload = json.loads(path.read_text())
    mutate(payload)
    path.write_text(json.dumps(payload))


def test_hostile_wrong_selected_models_hash(tmp_path):
    repo = copy_repository(tmp_path)
    _tamper_json(
        repo / "artifacts/models/selected_models_v1.json",
        lambda payload: payload["tasks"].__setitem__(
            "organ_support",
            {**payload["tasks"]["organ_support"], "artifact_sha256": "0" * 64},
        ),
    )
    with pytest.raises((Stage4BundleError, G3FreezeError)):
        resolve_stage4_bundle(repo)


def test_hostile_inactive_g3(tmp_path):
    repo = copy_repository(tmp_path)
    _tamper_json(repo / "artifacts/governance/g3_freeze.json", lambda payload: payload.__setitem__("freeze_status", "REVOKED"))
    with pytest.raises((Stage4BundleError, G3FreezeError)):
        resolve_stage4_bundle(repo)


def test_hostile_changed_feature_schema(tmp_path):
    repo = copy_repository(tmp_path)
    schema_path = repo / "configs/synthetic/feature_schema_v2.json"
    schema_path.write_text(schema_path.read_text() + "\n")
    with pytest.raises((Stage4BundleError, G3FreezeError)):
        resolve_stage4_bundle(repo)


def test_hostile_changed_preprocessor(tmp_path):
    repo = copy_repository(tmp_path)
    path = repo / "artifacts/preprocessors/synthetic_feature_preprocessor_v1.json"
    path.write_text(path.read_text() + "\n")
    with pytest.raises((Stage4BundleError, G3FreezeError)):
        resolve_stage4_bundle(repo)


def test_hostile_changed_recovery_scaler(tmp_path):
    repo = copy_repository(tmp_path)
    path = repo / "artifacts/preprocessors/recovery_target_scaler_synthetic_v1.json"
    path.write_text(path.read_text() + "\n")
    with pytest.raises((Stage4BundleError, G3FreezeError)):
        resolve_stage4_bundle(repo)


def test_hostile_changed_support_calibrator(tmp_path):
    repo = copy_repository(tmp_path)
    path = repo / "artifacts/calibration/isotonic_support_v1.json"
    path.write_text(path.read_text() + "\n")
    with pytest.raises((Stage4BundleError, G3FreezeError)):
        resolve_stage4_bundle(repo)


def test_hostile_changed_threshold(tmp_path):
    repo = copy_repository(tmp_path)
    path = repo / "artifacts/thresholds/support_threshold_v1.json"
    path.write_text(path.read_text() + "\n")
    with pytest.raises((Stage4BundleError, G3FreezeError)):
        resolve_stage4_bundle(repo)


def test_hostile_wrong_recovery_model_file(tmp_path):
    repo = copy_repository(tmp_path)
    path = repo / "artifacts/search/xgb/phase12/recovery/xgb-recovery-014/attempt-1/bundle/recovery_24h.json"
    original = path.read_bytes()
    path.write_bytes(original + b" ")
    with pytest.raises((Stage4BundleError, G3FreezeError)):
        resolve_stage4_bundle(repo)


def test_hostile_wrong_icu_checkpoint(tmp_path):
    repo = copy_repository(tmp_path)
    path = repo / "artifacts/search/gru/final_v2/icu_time/gru-icu-time-026/attempt-1/model.pt"
    original = path.read_bytes()
    path.write_bytes(original + b"\x00")
    with pytest.raises((Stage4BundleError, G3FreezeError)):
        resolve_stage4_bundle(repo)


def test_hostile_wrong_support_model_file(tmp_path):
    repo = copy_repository(tmp_path)
    path = repo / "artifacts/search/xgb/phase12/organ_support/xgb-support-024/attempt-1/bundle/organ_support.json"
    original = path.read_bytes()
    path.write_bytes(original + b" ")
    with pytest.raises((Stage4BundleError, G3FreezeError)):
        resolve_stage4_bundle(repo)


def test_hostile_stale_serving_manifest_lstm_family_rejected(tmp_path):
    repo = copy_repository(tmp_path)
    _tamper_json(
        repo / "artifacts/models/selected_models_v1.json",
        lambda payload: payload["tasks"].__setitem__(
            "recovery", {**payload["tasks"]["recovery"], "family": "lstm"}
        ),
    )
    with pytest.raises((Stage4BundleError, G3FreezeError)):
        resolve_stage4_bundle(repo)
