"""Fail-closed per-task model/explanation routing without attribution math."""

import math
from dataclasses import dataclass
from datetime import datetime
from types import MappingProxyType
from typing import Mapping, Optional, Protocol, Sequence, Tuple

from evaluation.selection_validation import EXPLANATION_METHOD, SERVING_FAMILIES, TASKS
from serving.artifacts import ServingBundle
from serving.history import StoredStayTimeline, TruncatedStayHistory


ROUTING_POLICY = MappingProxyType(
    {"xgboost": "tree_shap", "gru": "integrated_gradients"}
)


class ExplanationRouterError(RuntimeError):
    """Base error for explanation routing and lineage failures."""


class UnsupportedExplanationFamilyError(ExplanationRouterError):
    pass


class InvalidExplanationRouteError(ExplanationRouterError):
    pass


class MissingExplanationAdapterError(ExplanationRouterError):
    pass


class ModelIdentityMismatchError(ExplanationRouterError):
    pass


class ExplanationCutoffMismatchError(ExplanationRouterError):
    pass


class ExplanationTaskMismatchError(ExplanationRouterError):
    pass


class StaleExplanationManifestError(ExplanationRouterError):
    pass


class ExplanationSchemaError(ExplanationRouterError):
    pass


@dataclass(frozen=True)
class ExplanationContext:
    """Prediction-bound input supplied to exactly one explanation adapter."""

    task: str
    input_task: str
    family: str
    input_family: str
    model_sha256: str
    prediction_time: str
    input_prediction_time: str
    manifest_version: str
    manifest_sha256: str
    feature_schema_version: str
    feature_schema_sha256: Optional[str]
    model: object
    prepared_input: object
    raw_output: object
    explanation_target: Optional[str]
    feature_metadata: Optional[Mapping[str, object]]
    synthetic: bool
    support_calibrator_sha256: Optional[str] = None
    support_threshold_sha256: Optional[str] = None


@dataclass(frozen=True)
class AdapterExplanation:
    """Adapter-returned metadata and minimal Phase-5-compatible payload."""

    task: str
    family: str
    method: str
    model_sha256: str
    prediction_time: str
    manifest_version: str
    manifest_sha256: str
    feature_schema_version: str
    synthetic: bool
    items: Tuple[Mapping[str, object], ...]
    details: Optional[Mapping[str, object]] = None


@dataclass(frozen=True)
class ExplanationResult:
    """Validated router output bound to the selected serving state."""

    task: str
    family: str
    method: str
    model_sha256: str
    prediction_time: str
    manifest_version: str
    manifest_sha256: str
    feature_schema_version: str
    feature_schema_sha256: Optional[str]
    explanation_target: Optional[str]
    synthetic: bool
    items: Tuple[Mapping[str, object], ...]
    details: Optional[Mapping[str, object]] = None


class ExplanationAdapter(Protocol):
    method: str
    supported_families: Tuple[str, ...]
    synthetic: bool

    def explain(self, context: ExplanationContext) -> AdapterExplanation:
        """Return attribution payload metadata; computation is deferred."""


def lint_descriptive_explanation_text(text: str) -> Tuple[str, ...]:
    """Reject causal or treatment-directive wording in explanation prose."""

    prohibited = (
        "caused deterioration",
        "caused recovery",
        "caused the prediction",
        "should trigger treatment",
        "recommend medication",
        "recommend ventilation",
        "order treatment",
        "clinical decision directive",
        "discovered why the patient",
    )
    lowered = text.lower()
    return tuple(term for term in prohibited if term in lowered)


def _validate_timestamp(value: str, field: str) -> None:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (AttributeError, ValueError) as error:
        raise ExplanationSchemaError(field + " must be ISO-8601") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ExplanationSchemaError(field + " must be timezone-aware")


class ExplanationRouter:
    """Resolve adapters solely from an already validated serving bundle."""

    def __init__(
        self,
        bundle: ServingBundle,
        adapters: Mapping[str, ExplanationAdapter],
    ) -> None:
        self._bundle = bundle
        self._adapters = MappingProxyType(dict(adapters))
        self._validate_selected_routes()
        self._validate_registry()

    @property
    def synthetic(self) -> bool:
        return self._bundle.scope == "synthetic"

    @property
    def manifest_version(self) -> str:
        return self._bundle.manifest_version

    @property
    def manifest_sha256(self) -> str:
        return self._bundle.manifest_file_sha256

    def _validate_selected_routes(self) -> None:
        for task in TASKS:
            selected = self._bundle.task(task)
            family = selected.identity.family
            if family not in SERVING_FAMILIES or family not in ROUTING_POLICY:
                raise UnsupportedExplanationFamilyError(
                    "serving explanation family is unsupported: " + str(family)
                )
            expected = ROUTING_POLICY[family]
            if selected.identity.explanation_method != expected:
                raise InvalidExplanationRouteError(
                    "selected manifest family/explanation route mismatch for " + task
                )

    def _validate_registry(self) -> None:
        required = {
            ROUTING_POLICY[self._bundle.task(task).identity.family] for task in TASKS
        }
        for method in required:
            adapter = self._adapters.get(method)
            if adapter is None:
                raise MissingExplanationAdapterError(
                    "required explanation adapter is absent: " + method
                )
            if adapter.method != method:
                raise InvalidExplanationRouteError(
                    "adapter registry key and declared method differ"
                )
            expected_family = next(
                family for family, routed in ROUTING_POLICY.items() if routed == method
            )
            if tuple(adapter.supported_families) != (expected_family,):
                raise InvalidExplanationRouteError(
                    "adapter declares incompatible supported family"
                )
            if adapter.synthetic is not self.synthetic:
                raise InvalidExplanationRouteError(
                    "adapter synthetic/real scope is incompatible with serving bundle"
                )

    def explain(self, context: ExplanationContext) -> ExplanationResult:
        """Validate all prediction bindings before invoking one adapter."""

        if context.task not in TASKS:
            raise ExplanationTaskMismatchError("unknown explanation task")
        selected = self._bundle.task(context.task)
        identity = selected.identity
        if context.input_task != context.task:
            raise ExplanationTaskMismatchError(
                "prepared-input task differs from explanation task"
            )
        if getattr(context.model, "task", None) != context.task:
            raise ExplanationTaskMismatchError("model task differs from explanation task")
        if (
            context.family != identity.family
            or context.input_family != identity.family
            or getattr(context.model, "family", None) != identity.family
        ):
            raise ModelIdentityMismatchError("selected model family mismatch")
        if (
            context.model_sha256 != identity.artifact_sha256
            or getattr(context.model, "artifact_sha256", None) != identity.artifact_sha256
            or context.model is not selected.predictor
        ):
            raise ModelIdentityMismatchError("selected model artifact/hash mismatch")
        if (
            context.manifest_version != self._bundle.manifest_version
            or context.manifest_sha256 != self._bundle.manifest_file_sha256
        ):
            raise StaleExplanationManifestError("explanation context uses a stale manifest")
        if context.feature_schema_version != identity.feature_version:
            raise ExplanationSchemaError("feature schema version mismatch")
        if context.feature_schema_sha256 is not None:
            raise ExplanationSchemaError(
                "feature schema hash is not bound by the current selected manifest"
            )
        if context.feature_metadata is not None and context.feature_metadata.get(
            "feature_schema_version"
        ) != context.feature_schema_version:
            raise ExplanationSchemaError("feature metadata version mismatch")
        if context.prediction_time != context.input_prediction_time:
            raise ExplanationCutoffMismatchError(
                "prediction and prepared-input cutoffs differ"
            )
        _validate_timestamp(context.prediction_time, "prediction_time")
        if context.synthetic is not self.synthetic:
            raise ExplanationSchemaError("explanation scope mismatch")
        if isinstance(context.prepared_input, (StoredStayTimeline, TruncatedStayHistory)):
            raise ExplanationSchemaError("raw history cannot be passed to an explanation adapter")
        if context.task == "organ_support":
            if (
                context.support_calibrator_sha256
                != self._bundle.support_calibrator_sha256
                or context.support_threshold_sha256
                != self._bundle.support_threshold_sha256
            ):
                raise ExplanationSchemaError("support calibration lineage mismatch")
        elif (
            context.support_calibrator_sha256 is not None
            or context.support_threshold_sha256 is not None
        ):
            raise ExplanationSchemaError("support calibration identity attached to wrong task")

        method = ROUTING_POLICY[identity.family]
        if identity.explanation_method != method:
            raise InvalidExplanationRouteError("selected explanation method is incompatible")
        adapter = self._adapters.get(method)
        if adapter is None:
            raise MissingExplanationAdapterError("required explanation adapter is absent")
        # All selected-state, model, task, cutoff, scope, and input checks occur
        # before this call. The router never recomputes features or predictions.
        output = adapter.explain(context)
        self._validate_adapter_output(output, context, method)
        return ExplanationResult(
            task=context.task,
            family=context.family,
            method=method,
            model_sha256=context.model_sha256,
            prediction_time=context.prediction_time,
            manifest_version=context.manifest_version,
            manifest_sha256=context.manifest_sha256,
            feature_schema_version=context.feature_schema_version,
            feature_schema_sha256=context.feature_schema_sha256,
            explanation_target=context.explanation_target,
            synthetic=context.synthetic,
            items=tuple(MappingProxyType(dict(item)) for item in output.items),
            details=output.details,
        )

    def _validate_adapter_output(
        self,
        output: AdapterExplanation,
        context: ExplanationContext,
        method: str,
    ) -> None:
        expected = (
            context.task,
            context.family,
            method,
            context.model_sha256,
            context.prediction_time,
            context.manifest_version,
            context.manifest_sha256,
            context.feature_schema_version,
            context.synthetic,
        )
        actual = (
            output.task,
            output.family,
            output.method,
            output.model_sha256,
            output.prediction_time,
            output.manifest_version,
            output.manifest_sha256,
            output.feature_schema_version,
            output.synthetic,
        )
        if actual != expected:
            raise ExplanationSchemaError("adapter result metadata mismatch")
        if not isinstance(output.items, tuple):
            raise ExplanationSchemaError("adapter items must be an immutable tuple")
        for item in output.items:
            if not isinstance(item, Mapping) or set(item) != {"feature_name", "attribution"}:
                raise ExplanationSchemaError("explanation item schema mismatch")
            name = item["feature_name"]
            value = item["attribution"]
            if not isinstance(name, str) or not name.strip():
                raise ExplanationSchemaError("feature name must be non-empty")
            if lint_descriptive_explanation_text(name):
                raise ExplanationSchemaError("causal or treatment wording is prohibited")
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise ExplanationSchemaError("attribution must be finite numeric")
