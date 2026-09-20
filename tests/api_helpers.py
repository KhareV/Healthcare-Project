from copy import deepcopy

from fastapi.testclient import TestClient

from api.main import BundleMetadataProvider, create_app
from serving.pipeline import PredictionPipeline
from serving.postprocessing import CanonicalICUTimeServingPostprocessor
from serving_helpers import FixtureRuntime, MockPostprocessor, build_kwargs, copy_serving_fixture


REQUEST = {
    "stay_id": "SYNTHETIC_PHASE11_STAY",
    "prediction_time": "2026-01-02T12:00:00+00:00",
}


def build_pipeline(
    tmp_path, *, runtime=None, canonical_icu=True, recovery_provider=None
):
    root = copy_serving_fixture(tmp_path)
    runtime = runtime or FixtureRuntime()
    postprocessor = runtime.postprocessor
    if canonical_icu:
        postprocessor = CanonicalICUTimeServingPostprocessor(
            recovery_provider or MockPostprocessor()
        )
    pipeline = PredictionPipeline.build(
        **build_kwargs(root, runtime),
        input_provider=runtime.input_provider,
        postprocessor=postprocessor,
        explanation_adapters=runtime.explanation_adapters,
    )
    return root, pipeline, runtime


def client_for(pipeline, metadata_provider=None):
    return TestClient(
        create_app(
            pipeline,
            metadata_provider=metadata_provider or BundleMetadataProvider(),
        )
    )


class StaleMetadataProvider(BundleMetadataProvider):
    def __init__(self, mutation):
        self.mutation = mutation

    def get_metadata(self, pipeline):
        value = deepcopy(super().get_metadata(pipeline))
        self.mutation(value)
        return value
