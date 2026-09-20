from pathlib import Path

import pytest

import evaluation.threshold
import training.engine
from serving.artifacts import ArtifactMissingError
from serving.pipeline import PredictionPipeline, PredictionPipelineError
from serving.prediction_schema import validate_response
from serving_helpers import (
    FIXTURE_RELATIVE,
    FixtureRuntime,
    build_kwargs,
    copy_serving_fixture,
    read_json,
    rehash_manifest,
    write_json,
)


REQUEST = {
    "stay_id": "SYNTHETIC_PHASE6_STAY",
    "prediction_time": "2026-01-02T12:00:00+00:00",
}


def _pipeline(root, runtime=None):
    runtime = runtime or FixtureRuntime()
    return (
        PredictionPipeline.build(
            **build_kwargs(root, runtime),
            input_provider=runtime.input_provider,
            postprocessor=runtime.postprocessor,
            explanation_adapters=runtime.explanation_adapters,
        ),
        runtime,
    )


def test_mixed_family_pipeline_produces_phase5_valid_response(tmp_path):
    root = copy_serving_fixture(tmp_path)
    pipeline, runtime = _pipeline(root)
    response = pipeline.predict(REQUEST)
    assert validate_response(response, synthetic=True) == response
    assert response["model_versions"]["recovery"]["family"] == "gru"
    assert response["model_versions"]["icu_stay_time"]["family"] == "xgboost"
    assert response["model_versions"]["organ_support"]["family"] == "gru"
    assert response["organ_support_probability_calibrated"] == 0.25


def test_task_routing_calls_only_each_selected_adapter(tmp_path):
    root = copy_serving_fixture(tmp_path)
    pipeline, runtime = _pipeline(root)
    pipeline.predict(REQUEST)
    assert {task: predictor.predict_calls for task, predictor in runtime.predictors.items()} == {
        "recovery": 1, "icu_stay_time": 1, "organ_support": 1
    }
    assert runtime.loader_calls == [
        ("gru", "recovery"), ("xgboost", "icu_stay_time"), ("gru", "organ_support")
    ]
    assert [context.task for context in runtime.explanation_adapters["integrated_gradients"].collector.contexts if context.family == "gru"] == ["recovery", "organ_support"]
    assert [context.task for context in runtime.explanation_adapters["tree_shap"].collector.contexts if context.family == "xgboost"] == ["icu_stay_time"]


def _switch_second_mixed_family(root):
    directory = root / FIXTURE_RELATIVE
    manifest = read_json(directory / "selected_models_synthetic_phase6_v1.json")
    changes = {
        "recovery": ("xgboost", "SYNTHETIC_RECOVERY_XGB_PHASE6_NOT_REAL", "tree_shap"),
        "icu_stay_time": ("gru", "SYNTHETIC_ICU_GRU_PHASE6_NOT_REAL", "integrated_gradients"),
        "organ_support": ("xgboost", "SYNTHETIC_SUPPORT_XGB_PHASE6_NOT_REAL", "tree_shap"),
    }
    metadata_names = {
        "recovery": "recovery_model.metadata.json",
        "icu_stay_time": "icu_model.metadata.json",
        "organ_support": "support_model.metadata.json",
    }
    for task, (family, version, method) in changes.items():
        entry = manifest["tasks"][task]
        entry.update(family=family, model_version=version, explanation_method=method)
        metadata_path = directory / metadata_names[task]
        metadata = read_json(metadata_path)
        metadata.update(family=family, model_version=version)
        entry["artifact_metadata_sha256"] = write_json(metadata_path, metadata)
    calibrator_path = directory / "support_calibrator.json"
    calibrator = read_json(calibrator_path)
    calibrator["selected_family"] = "xgboost"
    calibrator_hash = write_json(calibrator_path, calibrator)
    manifest["tasks"]["organ_support"]["calibrator_sha256"] = calibrator_hash
    threshold_path = directory / "support_threshold.json"
    threshold = read_json(threshold_path)
    threshold["calibrator_sha256"] = calibrator_hash
    manifest["tasks"]["organ_support"]["threshold_sha256"] = write_json(
        threshold_path, threshold
    )
    def replace(payload):
        payload.clear()
        payload.update(manifest)
    rehash_manifest(root, replace)


def test_second_mixed_family_configuration_routes_independently(tmp_path):
    root = copy_serving_fixture(tmp_path)
    _switch_second_mixed_family(root)
    pipeline, runtime = _pipeline(root)
    response = pipeline.predict(REQUEST)
    assert runtime.loader_calls == [
        ("xgboost", "recovery"), ("gru", "icu_stay_time"), ("xgboost", "organ_support")
    ]
    assert {
        task: response["model_versions"][task]["family"]
        for task in ("recovery", "icu_stay_time", "organ_support")
    } == {"recovery": "xgboost", "icu_stay_time": "gru", "organ_support": "xgboost"}
    assert [context.task for context in runtime.explanation_adapters["tree_shap"].collector.contexts if context.family == "xgboost"] == ["recovery", "organ_support"]
    assert [context.task for context in runtime.explanation_adapters["integrated_gradients"].collector.contexts if context.family == "gru"] == ["icu_stay_time"]


def test_response_metadata_comes_from_bundle_not_hard_coded(tmp_path):
    root = copy_serving_fixture(tmp_path)
    pipeline, _ = _pipeline(root)
    response = pipeline.predict(REQUEST)
    assert response["model_metadata"]["manifest_version"] == "selected_models_synthetic_phase6_v1"
    assert response["model_metadata"]["split_version"] == "SYNTHETIC_SPLIT_PHASE6_NOT_REAL"
    assert response["model_metadata"]["label_version"]["organ_support"] == "SYNTHETIC_SUPPORT_LABELS_PHASE6_NOT_REAL"
    assert response["model_versions"]["organ_support"]["artifact_sha256"] == pipeline.bundle.task("organ_support").identity.artifact_sha256


def test_prepared_input_seam_and_frozen_preprocessors_are_invoked_per_task(tmp_path):
    root = copy_serving_fixture(tmp_path)
    pipeline, runtime = _pipeline(root)
    pipeline.predict(REQUEST)
    assert [item[2] for item in runtime.input_provider.calls] == [
        "recovery", "icu_stay_time", "organ_support"
    ]
    assert {task: value.transform_calls for task, value in runtime.preprocessors.items()} == {
        "recovery": 1, "icu_stay_time": 1, "organ_support": 1
    }


def test_support_calibrator_transform_only_and_never_fit(tmp_path):
    root = copy_serving_fixture(tmp_path)
    pipeline, runtime = _pipeline(root)
    pipeline.predict(REQUEST)
    assert runtime.calibrator.transform_calls == 1
    assert runtime.calibrator.fit_calls == 0


def test_training_and_threshold_search_entrypoints_are_never_called(tmp_path, monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("training or threshold search called from serving")
    monkeypatch.setattr(training.engine, "train_with_early_stopping", forbidden)
    monkeypatch.setattr(evaluation.threshold, "choose_support_threshold", forbidden)
    root = copy_serving_fixture(tmp_path)
    pipeline, _ = _pipeline(root)
    pipeline.predict(REQUEST)


def test_mock_explanation_provider_receives_selected_hash_without_future_state(tmp_path):
    root = copy_serving_fixture(tmp_path)
    pipeline, runtime = _pipeline(root)
    pipeline.predict(REQUEST)
    assert runtime.explanations.synthetic is True
    assert runtime.explanations.calls == [
        (
            task,
            pipeline.bundle.task(task).identity.family,
            pipeline.bundle.task(task).identity.artifact_sha256,
        )
        for task in ("recovery", "icu_stay_time", "organ_support")
    ]


def test_repeated_calls_are_deterministic(tmp_path):
    root = copy_serving_fixture(tmp_path)
    pipeline, _ = _pipeline(root)
    assert pipeline.predict(REQUEST) == pipeline.predict(REQUEST)


def test_real_pipeline_missing_selected_models_v1_fails_before_prediction(tmp_path):
    root = copy_serving_fixture(tmp_path)
    runtime = FixtureRuntime()
    kwargs = build_kwargs(root, runtime)
    kwargs.update(
        manifest_ref="artifacts/models/selected_models_v1.json",
        expected_manifest_sha256="1" * 64,
        scope="real",
    )
    with pytest.raises(ArtifactMissingError):
        PredictionPipeline.build(
            **kwargs,
            input_provider=runtime.input_provider,
            postprocessor=runtime.postprocessor,
            explanation_adapters=runtime.explanation_adapters,
        )
    assert runtime.loader_calls == []
    assert runtime.input_provider.calls == []


def test_phase12_and_later_modules_are_not_created():
    root = Path(__file__).resolve().parents[1]
    assert (root / "api/main.py").is_file()
    assert (root / "api/schemas.py").is_file()
    assert (root / "dashboard/app.py").is_file()
    assert (root / "src/explainability/ig.py").is_file()
    assert (root / "src/explainability/tree_shap.py").is_file()


def test_pipeline_source_has_no_test_data_or_model_selection_dependency():
    root = Path(__file__).resolve().parents[1]
    source = (root / "src/serving/pipeline.py").read_text().lower()
    assert "final_test" not in source
    assert "choose_best" not in source
    assert "selected_models_v1" not in source
    assert ".fit(" not in source
