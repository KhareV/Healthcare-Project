"""Immutable prediction orchestration over validated Phase-6/7 dependencies."""

import math
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from evaluation.selection_validation import TASKS
from explainability.router import ExplanationAdapter, ExplanationContext, ExplanationRouter
from serving.artifacts import (
    CalibratorLoader,
    ModelLoader,
    PreprocessorLoader,
    ServingBundle,
    load_serving_bundle,
)
from serving.interfaces import CanonicalInputProvider, ServingPostprocessor
from serving.prediction_schema import validate_request, validate_response


class PredictionPipelineError(RuntimeError):
    """Raised for invalid runtime dependency behavior after bundle construction."""


class PredictionPipeline:
    """One immutable selected-manifest/artifact state for deterministic inference."""

    def __init__(
        self,
        bundle: ServingBundle,
        *,
        input_provider: CanonicalInputProvider,
        postprocessor: ServingPostprocessor,
        explanation_router: ExplanationRouter,
        explanation_targets: Mapping[str, str] = MappingProxyType({}),
    ) -> None:
        if explanation_router.manifest_version != bundle.manifest_version or explanation_router.manifest_sha256 != bundle.manifest_file_sha256:
            raise PredictionPipelineError("explanation router is bound to a different manifest")
        self._bundle = bundle
        self._input_provider = input_provider
        self._postprocessor = postprocessor
        self._explanation_router = explanation_router
        # Absent for every task by default, exactly reproducing the prior
        # unconditional ``explanation_target=None`` behavior relied on by the
        # existing synthetic-fixture adapters/tests.
        self._explanation_targets = dict(explanation_targets)

    @classmethod
    def build(
        cls,
        root: Path,
        *,
        manifest_ref: str,
        expected_manifest_sha256: str,
        scope: str,
        model_loaders: Mapping[str, ModelLoader],
        preprocessor_loader: PreprocessorLoader,
        calibrator_loader: CalibratorLoader,
        input_provider: CanonicalInputProvider,
        postprocessor: ServingPostprocessor,
        explanation_adapters: Mapping[str, ExplanationAdapter],
    ) -> "PredictionPipeline":
        bundle = load_serving_bundle(
            root,
            manifest_ref=manifest_ref,
            expected_manifest_sha256=expected_manifest_sha256,
            scope=scope,
            model_loaders=model_loaders,
            preprocessor_loader=preprocessor_loader,
            calibrator_loader=calibrator_loader,
        )
        router = ExplanationRouter(bundle, explanation_adapters)
        return cls(
            bundle,
            input_provider=input_provider,
            postprocessor=postprocessor,
            explanation_router=router,
        )

    @property
    def bundle(self) -> ServingBundle:
        return self._bundle

    def predict(self, request: object) -> Mapping[str, object]:
        request = validate_request(request)
        stay_id = request["stay_id"]
        prediction_time = str(request["prediction_time"])
        prepared = {}
        raw = {}
        explanations = {}
        for task in TASKS:
            selected = self._bundle.task(task)
            supplied = self._input_provider.get_canonical_input(
                stay_id=stay_id,
                prediction_time=prediction_time,
                task=task,
                family=selected.identity.family,
                feature_version=selected.identity.feature_version,
            )
            model_input = selected.preprocessor.transform(supplied)
            prepared[task] = model_input
            raw[task] = selected.predictor.predict(model_input)
            result = self._explanation_router.explain(ExplanationContext(
                task=task,
                input_task=task,
                family=selected.identity.family,
                input_family=selected.identity.family,
                model_sha256=selected.identity.artifact_sha256,
                prediction_time=prediction_time,
                input_prediction_time=prediction_time,
                manifest_version=self._bundle.manifest_version,
                manifest_sha256=self._bundle.manifest_file_sha256,
                feature_schema_version=selected.identity.feature_version,
                feature_schema_sha256=None,
                model=selected.predictor,
                prepared_input=model_input,
                raw_output=raw[task],
                explanation_target=self._explanation_targets.get(task),
                feature_metadata={
                    "feature_schema_version": selected.identity.feature_version,
                    **getattr(selected.preprocessor, "feature_metadata", {}),
                },
                synthetic=self._bundle.scope == "synthetic",
                support_calibrator_sha256=(self._bundle.support_calibrator_sha256 if task == "organ_support" else None),
                support_threshold_sha256=(self._bundle.support_threshold_sha256 if task == "organ_support" else None),
            ))
            explanations[task] = {
                "family": result.family,
                "explanation_method": result.method,
                "items": [dict(item) for item in result.items],
            }

        task_bundles = {item.identity.task: item for item in self._bundle.tasks}
        recovery = self._postprocessor.recovery(
            raw["recovery"],
            prepared["recovery"],
            stay_id=stay_id,
            prediction_time=prediction_time,
            model_identity=task_bundles["recovery"].identity,
        )
        icu_hours = self._postprocessor.icu_stay_time_hours(raw["icu_stay_time"])
        support_raw = raw["organ_support"]
        if isinstance(support_raw, bool) or not isinstance(support_raw, (int, float)):
            raise PredictionPipelineError("support adapter must return raw probability")
        support_raw = float(support_raw)
        if not math.isfinite(support_raw) or not 0 <= support_raw <= 1:
            raise PredictionPipelineError("support raw probability must be finite within [0,1]")
        support_probability = self._bundle.support_calibrator.transform(support_raw)
        if (
            isinstance(support_probability, bool)
            or not isinstance(support_probability, (int, float))
            or not math.isfinite(float(support_probability))
            or not 0 <= float(support_probability) <= 1
        ):
            raise PredictionPipelineError("support calibrator returned invalid probability")
        support_probability = float(support_probability)
        # Applying this immutable comparison proves the threshold dependency is
        # usable; Phase 5 intentionally omits alert/class from its public v1 response.
        _support_class = int(support_probability >= self._bundle.support_threshold)

        response = {
            "schema_version": "prediction_schema_v1",
            "prediction_time": prediction_time,
            "mode": "RETROSPECTIVE_SEQUENTIAL_REPLAY",
            "model_versions": {
                task: {
                    "family": task_bundles[task].identity.family,
                    "model_version": task_bundles[task].identity.model_version,
                    "artifact_sha256": task_bundles[task].identity.artifact_sha256,
                    "preprocessor_sha256": task_bundles[task].identity.preprocessor_sha256,
                    **(
                        {"calibrator_sha256": self._bundle.support_calibrator_sha256}
                        if task == "organ_support"
                        else {}
                    ),
                }
                for task in TASKS
            },
            "recovery": dict(recovery),
            "icu_stay_time_hours": icu_hours,
            "organ_support_probability_calibrated": support_probability,
            "explanation_features": explanations,
            "data_quality": dict(
                self._input_provider.data_quality(
                    stay_id=stay_id, prediction_time=prediction_time
                )
            ),
            "model_metadata": {
                "feature_version": {
                    task: task_bundles[task].identity.feature_version for task in TASKS
                },
                "label_version": {
                    task: task_bundles[task].identity.label_version for task in TASKS
                },
                "split_version": self._bundle.split_version,
                "manifest_version": self._bundle.manifest_version,
            },
        }
        return validate_response(response, synthetic=self._bundle.scope == "synthetic")
