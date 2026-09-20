import math

import pytest

from serving.artifacts import ArtifactCompatibilityError, ArtifactHashMismatchError
from serving.pipeline import PredictionPipeline
from serving_helpers import (
    FIXTURE_RELATIVE,
    FixtureRuntime,
    build_kwargs,
    copy_serving_fixture,
    read_json,
    rehash_manifest,
    write_json,
)
from phase7_helpers import (
    MAIN_CUTOFF,
    MAIN_STAY,
    FrozenSyntheticPreprocessor,
    build_provider,
    load_schema,
)


def test_transform_only_preprocessor_fit_trap_remains_zero():
    provider, _ = build_provider()
    canonical = provider.get_canonical_input(
        stay_id=MAIN_STAY,
        prediction_time=MAIN_CUTOFF,
        task="recovery",
        family="gru",
        feature_version=load_schema().version,
    )
    preprocessor = FrozenSyntheticPreprocessor()
    assert preprocessor.transform(canonical) == canonical
    assert preprocessor.transform_calls == 1
    assert preprocessor.fit_calls == 0
    assert preprocessor.fit_transform_calls == 0


class _PipelinePostprocessor:
    def recovery(self, raw_output, prepared_input, **_context):
        return {
            "delta_24h": raw_output[0],
            "delta_48h": raw_output[1],
            "reconstructed_sofa_24h": 4.25,
            "reconstructed_sofa_48h": 3.5,
        }

    def icu_stay_time_hours(self, raw_output):
        return 42.0


def test_phase6_pipeline_consumes_cutoff_built_inputs_and_frozen_transforms(tmp_path):
    root = copy_serving_fixture(tmp_path)
    runtime = FixtureRuntime()
    provider, builder = build_provider()
    pipeline = PredictionPipeline.build(
        **build_kwargs(root, runtime),
        input_provider=provider,
        postprocessor=_PipelinePostprocessor(),
        explanation_adapters=runtime.explanation_adapters,
    )
    response = pipeline.predict({"stay_id": MAIN_STAY, "prediction_time": MAIN_CUTOFF})
    assert len(builder.calls) == 3
    assert {task: item.transform_calls for task, item in runtime.preprocessors.items()} == {
        "recovery": 1,
        "icu_stay_time": 1,
        "organ_support": 1,
    }
    assert response["data_quality"]["total_bins"] == 8


def test_missing_train_fit_provenance_blocks_before_load_or_predict(tmp_path):
    root = copy_serving_fixture(tmp_path)
    directory = root / FIXTURE_RELATIVE
    preprocessor_path = directory / "recovery_preprocessor.json"
    preprocessor = read_json(preprocessor_path)
    preprocessor.pop("fit_partition")
    new_preprocessor_hash = write_json(preprocessor_path, preprocessor)
    model_meta_path = directory / "recovery_model.metadata.json"
    model_meta = read_json(model_meta_path)
    model_meta["preprocessor_sha256"] = new_preprocessor_hash
    new_model_meta_hash = write_json(model_meta_path, model_meta)

    def mutate(manifest):
        entry = manifest["tasks"]["recovery"]
        entry["preprocessor_sha256"] = new_preprocessor_hash
        entry["artifact_metadata_sha256"] = new_model_meta_hash

    rehash_manifest(root, mutate)
    runtime = FixtureRuntime()
    with pytest.raises(ArtifactCompatibilityError, match="fit_partition"):
        PredictionPipeline.build(
            **build_kwargs(root, runtime),
            input_provider=runtime.input_provider,
            postprocessor=runtime.postprocessor,
            explanation_adapters=runtime.explanation_adapters,
        )
    assert runtime.loader_calls == []
    assert runtime.input_provider.calls == []


def test_preprocessor_byte_drift_blocks_before_model_input(tmp_path):
    root = copy_serving_fixture(tmp_path)
    path = root / FIXTURE_RELATIVE / "recovery_preprocessor.json"
    path.write_text(path.read_text() + " ")
    runtime = FixtureRuntime()
    with pytest.raises(ArtifactHashMismatchError, match="preprocessor"):
        PredictionPipeline.build(
            **build_kwargs(root, runtime),
            input_provider=runtime.input_provider,
            postprocessor=runtime.postprocessor,
            explanation_adapters=runtime.explanation_adapters,
        )
    assert runtime.loader_calls == []
    assert runtime.input_provider.calls == []
