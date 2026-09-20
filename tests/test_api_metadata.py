from api_helpers import REQUEST, StaleMetadataProvider, build_pipeline, client_for
from serving.pipeline import PredictionPipeline
from serving.postprocessing import CanonicalICUTimeServingPostprocessor
from serving_helpers import FixtureRuntime, MockPostprocessor, build_kwargs
from test_prediction_pipeline import _switch_second_mixed_family


def test_model_metadata_uses_exact_bundle_for_all_tasks(tmp_path):
    _, pipeline, _ = build_pipeline(tmp_path)
    response = client_for(pipeline).get("/model-metadata")
    assert response.status_code == 200
    payload = response.json()
    assert payload["mode"] == "RETROSPECTIVE_SEQUENTIAL_REPLAY"
    assert payload["manifest_version"] == pipeline.bundle.manifest_version
    assert payload["manifest_sha256"] == pipeline.bundle.manifest_file_sha256
    assert payload["split_version"] == pipeline.bundle.split_version
    assert set(payload["tasks"]) == {"recovery", "icu_stay_time", "organ_support"}
    assert {task: payload["tasks"][task]["family"] for task in payload["tasks"]} == {
        "recovery": "gru", "icu_stay_time": "xgboost", "organ_support": "gru"
    }
    assert payload["tasks"]["recovery"]["explanation_method"] == "integrated_gradients"
    assert payload["tasks"]["icu_stay_time"]["explanation_method"] == "tree_shap"
    assert payload["organ_support"]["calibrator_sha256"] == pipeline.bundle.support_calibrator_sha256
    assert payload["organ_support"]["threshold"] == pipeline.bundle.support_threshold


def test_prediction_and_metadata_endpoint_share_exact_identities(tmp_path):
    _, pipeline, _ = build_pipeline(tmp_path)
    client = client_for(pipeline)
    metadata = client.get("/model-metadata").json()
    prediction = client.post("/predict", json=REQUEST).json()
    for task in metadata["tasks"]:
        assert prediction["model_versions"][task]["family"] == metadata["tasks"][task]["family"]
        assert prediction["model_versions"][task]["model_version"] == metadata["tasks"][task]["model_version"]
        assert prediction["model_versions"][task]["artifact_sha256"] == metadata["tasks"][task]["artifact_sha256"]
        assert prediction["explanation_features"][task]["explanation_method"] == metadata["tasks"][task]["explanation_method"]


def test_stale_manifest_or_wrong_explanation_metadata_is_503_before_prediction(tmp_path):
    _, pipeline, runtime = build_pipeline(tmp_path)
    mutations = (
        lambda value: value.update(manifest_sha256="1" * 64),
        lambda value: value["tasks"]["recovery"].update(explanation_method="tree_shap"),
    )
    for mutation in mutations:
        client = client_for(pipeline, StaleMetadataProvider(mutation))
        assert client.get("/model-metadata").status_code == 503
        assert client.post("/predict", json=REQUEST).status_code == 503
    assert all(model.predict_calls == 0 for model in runtime.predictors.values())


def test_metadata_has_no_generic_model_version_or_evaluation_metrics(tmp_path):
    _, pipeline, _ = build_pipeline(tmp_path)
    payload = client_for(pipeline).get("/model-metadata").json()
    assert "model_version" not in payload
    forbidden = {"test_mae", "validation_mae", "test_auprc", "bootstrap_ci"}
    assert forbidden.isdisjoint(str(payload).lower().split())


def test_reciprocal_mixed_family_metadata_uses_selected_bundle(tmp_path):
    root, _, _ = build_pipeline(tmp_path)
    _switch_second_mixed_family(root)
    runtime = FixtureRuntime()
    pipeline = PredictionPipeline.build(
        **build_kwargs(root, runtime),
        input_provider=runtime.input_provider,
        postprocessor=CanonicalICUTimeServingPostprocessor(MockPostprocessor()),
        explanation_adapters=runtime.explanation_adapters,
    )
    payload = client_for(pipeline).get("/model-metadata").json()
    assert {task: payload["tasks"][task]["family"] for task in payload["tasks"]} == {
        "recovery": "xgboost",
        "icu_stay_time": "gru",
        "organ_support": "xgboost",
    }
    assert {task: payload["tasks"][task]["explanation_method"] for task in payload["tasks"]} == {
        "recovery": "tree_shap",
        "icu_stay_time": "integrated_gradients",
        "organ_support": "tree_shap",
    }
