import pytest

from phase7_helpers import MAIN_CUTOFF, MAIN_STAY, build_provider
from serving.artifacts import ArtifactHashMismatchError
from serving.history import StoredStayTimeline, TruncatedStayHistory
from serving.pipeline import PredictionPipeline
from serving.prediction_schema import validate_response
from serving_helpers import (
    FIXTURE_RELATIVE,
    FixtureRuntime,
    build_kwargs,
    copy_serving_fixture,
)


class RouterPostprocessor:
    def recovery(self, raw_output, prepared_input, **_context):
        return {
            "delta_24h": raw_output[0],
            "delta_48h": raw_output[1],
            "reconstructed_sofa_24h": 4.25,
            "reconstructed_sofa_48h": 3.5,
        }

    def icu_stay_time_hours(self, raw_output):
        return 42.0


def test_pipeline_reuses_prediction_input_and_response_schema_remains_valid(tmp_path):
    root = copy_serving_fixture(tmp_path)
    runtime = FixtureRuntime()
    provider, builder = build_provider()
    pipeline = PredictionPipeline.build(
        **build_kwargs(root, runtime),
        input_provider=provider,
        postprocessor=RouterPostprocessor(),
        explanation_adapters=runtime.explanation_adapters,
    )
    response = pipeline.predict({"stay_id": MAIN_STAY, "prediction_time": MAIN_CUTOFF})
    assert validate_response(response, synthetic=True) == response
    # One canonical build per task; the router performs no fourth build.
    assert len(builder.calls) == 3
    assert len(runtime.explanations.contexts) == 3
    for context in runtime.explanations.contexts:
        predictor = runtime.predictors[context.task]
        assert context.prepared_input is predictor.last_prepared_input
        assert context.prediction_time == MAIN_CUTOFF
        assert context.input_prediction_time == MAIN_CUTOFF
        assert context.model is predictor
        assert not isinstance(context.prepared_input, (StoredStayTimeline, TruncatedStayHistory))
        assert context.feature_schema_version == pipeline.bundle.task(context.task).identity.feature_version
        assert context.manifest_sha256 == pipeline.bundle.manifest_file_sha256


def test_mixed_family_routes_are_independent_in_pipeline(tmp_path):
    root = copy_serving_fixture(tmp_path)
    runtime = FixtureRuntime()
    pipeline = PredictionPipeline.build(
        **build_kwargs(root, runtime),
        input_provider=runtime.input_provider,
        postprocessor=runtime.postprocessor,
        explanation_adapters=runtime.explanation_adapters,
    )
    pipeline.predict(
        {"stay_id": "SYNTHETIC_PHASE6_STAY", "prediction_time": "2026-01-02T12:00:00+00:00"}
    )
    ig_tasks = [context.task for context in runtime.explanation_adapters["integrated_gradients"].collector.contexts if context.family == "gru"]
    tree_tasks = [context.task for context in runtime.explanation_adapters["tree_shap"].collector.contexts if context.family == "xgboost"]
    assert ig_tasks == ["recovery", "organ_support"]
    assert tree_tasks == ["icu_stay_time"]


def test_mutated_model_is_rejected_before_router_or_adapter(tmp_path):
    root = copy_serving_fixture(tmp_path)
    model = root / FIXTURE_RELATIVE / "recovery_gru.mock"
    model.write_text(model.read_text() + "mutated")
    runtime = FixtureRuntime()
    with pytest.raises(ArtifactHashMismatchError, match="model"):
        PredictionPipeline.build(
            **build_kwargs(root, runtime),
            input_provider=runtime.input_provider,
            postprocessor=runtime.postprocessor,
            explanation_adapters=runtime.explanation_adapters,
        )
    assert runtime.explanations.calls == []


def test_real_selected_manifest_has_frozen_family_specific_explanations():
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    selected = json.loads(
        (root / "artifacts/models/selected_models_v1.json").read_text(encoding="utf-8")
    )
    assert selected["status"] == "SELECTED_MODELS_FROZEN_PRE_TEST"
    assert selected["test_accessed"] is False
    assert {
        task: (item["family"], item["explanation_method"])
        for task, item in selected["tasks"].items()
    } == {
        "recovery": ("xgboost", "tree_shap"),
        "icu_stay_time": ("gru", "integrated_gradients"),
        "organ_support": ("xgboost", "tree_shap"),
    }
