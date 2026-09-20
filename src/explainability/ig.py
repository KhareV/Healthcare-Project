"""Captum Integrated Gradients adapter for validated synthetic GRU contexts.

Real execution remains blocked until baseline and task-output domains are
reviewed and frozen.  This module never implements a fallback attribution
algorithm: if Captum is unavailable, execution fails explicitly.
"""

from dataclasses import asdict, dataclass, replace
from typing import Mapping, Optional, Protocol, Sequence, Tuple

import torch

from data.collate import CanonicalBatch
from explainability.router import AdapterExplanation, ExplanationContext
from experiments.search_governance import canonical_sha256


class IntegratedGradientsError(RuntimeError):
    pass


class CaptumDependencyError(IntegratedGradientsError):
    pass


class IGBaselineError(IntegratedGradientsError):
    pass


class IGTargetError(IntegratedGradientsError):
    pass


class IGInputError(IntegratedGradientsError):
    pass


@dataclass(frozen=True)
class AttributionTarget:
    task: str
    output_name: str
    output_index: int
    output_domain_version: str
    scientific_scope: str

    def validate(self, model: object, synthetic: bool) -> None:
        if self.task not in ("recovery", "icu_stay_time", "organ_support"):
            raise IGTargetError("unsupported attribution task")
        if self.output_index < 0:
            raise IGTargetError("output index must be nonnegative")
        if synthetic:
            if self.scientific_scope != "synthetic_development_only":
                raise IGTargetError("synthetic target must be explicitly development-only")
            if not self.output_domain_version.startswith("SYNTHETIC_"):
                raise IGTargetError("synthetic target domain must be unmistakably synthetic")
        else:
            raise IGTargetError(
                "real IG output domains are unresolved and require review"
            )
        if self.task == "recovery":
            order = getattr(model, "horizon_order", None)
            if not isinstance(order, tuple) or self.output_index >= len(order):
                raise IGTargetError("recovery horizon order is absent or incompatible")
            if order[self.output_index] != self.output_name:
                raise IGTargetError("recovery output name/index does not match model contract")
        elif self.output_index != 0:
            raise IGTargetError("scalar task output index must be zero")


@dataclass(frozen=True)
class IGConfig:
    version: str
    n_steps: int
    method: str
    internal_batch_size: Optional[int]
    return_convergence_delta: bool
    scientific_scope: str

    def validate(self) -> None:
        if self.scientific_scope != "synthetic_development_only":
            raise IntegratedGradientsError("real IG integration settings are not frozen")
        if not self.version.startswith("SYNTHETIC_") or self.n_steps < 2:
            raise IntegratedGradientsError("invalid synthetic IG configuration")
        if self.method not in (
            "gausslegendre",
            "riemann_left",
            "riemann_right",
            "riemann_middle",
            "riemann_trapezoid",
        ):
            raise IntegratedGradientsError("unsupported Captum integration method")
        if self.internal_batch_size is not None and self.internal_batch_size < 1:
            raise IntegratedGradientsError("internal_batch_size must be positive")

    @property
    def sha256(self) -> str:
        return canonical_sha256(asdict(self))


@dataclass(frozen=True)
class BaselineIdentity:
    policy_name: str
    version: str
    config_sha256: str
    feature_schema_version: str
    scientific_scope: str


class BaselineProvider(Protocol):
    identity: BaselineIdentity

    def baselines(self, batch: CanonicalBatch) -> Tuple[torch.Tensor, ...]:
        """Return one exact-shape baseline for every attributable tensor."""


class SyntheticZeroBaselineProvider:
    """Development-only zero reference for every attributable tensor.

    This is not an approved scientific baseline and is rejected in real scope.
    Padding is a fixed non-attributed argument. Values, observation indicators,
    optional TSLO, and optional statics each receive explicit zero references.
    """

    def __init__(self, feature_schema_version: str) -> None:
        config = {
            "policy": "all_attributable_tensors_zero",
            "padding": "fixed_additional_forward_argument",
            "scope": "synthetic_development_only",
            "feature_schema_version": feature_schema_version,
        }
        self.identity = BaselineIdentity(
            policy_name="SYNTHETIC_ZERO_ALL_ATTRIBUTABLE_INPUTS",
            version="SYNTHETIC_IG_BASELINE_V1",
            config_sha256=canonical_sha256(config),
            feature_schema_version=feature_schema_version,
            scientific_scope="synthetic_development_only",
        )

    def baselines(self, batch: CanonicalBatch) -> Tuple[torch.Tensor, ...]:
        return tuple(torch.zeros_like(value) for value in attributable_inputs(batch))


def attributable_inputs(batch: CanonicalBatch) -> Tuple[torch.Tensor, ...]:
    values = [batch.sequence, batch.observation_mask.to(batch.sequence.dtype)]
    if batch.tslo is not None:
        values.append(batch.tslo)
    if batch.static_features is not None:
        values.append(batch.static_features)
    return tuple(values)


class GRUForwardWrapper(torch.nn.Module):
    """Rebuild only the typed batch shell around the same prepared tensors."""

    def __init__(
        self, model: torch.nn.Module, template: CanonicalBatch, target: AttributionTarget
    ) -> None:
        super().__init__()
        self.model = model
        self.template = template
        self.target = target

    def full_output(self, *inputs: torch.Tensor) -> torch.Tensor:
        batch = self._batch(inputs)
        return self.model(batch)

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        output = self.full_output(*inputs)
        if output.ndim != 2 or self.target.output_index >= output.shape[1]:
            raise IGTargetError("model output shape is incompatible with target")
        return output[:, self.target.output_index]

    def _batch(self, inputs: Sequence[torch.Tensor]) -> CanonicalBatch:
        expected = 2 + int(self.template.tslo is not None) + int(
            self.template.static_features is not None
        )
        if len(inputs) != expected:
            raise IGInputError("attributable input count mismatch")
        index = 0
        sequence = inputs[index]
        index += 1
        observation = inputs[index]
        index += 1
        tslo = None
        static = None
        if self.template.tslo is not None:
            tslo = inputs[index]
            index += 1
        if self.template.static_features is not None:
            static = inputs[index]
        return replace(
            self.template,
            sequence=sequence,
            observation_mask=observation,
            tslo=tslo,
            static_features=static,
        )


def aggregate_absolute_attribution(
    attributions: Sequence[torch.Tensor],
    batch: CanonicalBatch,
    *,
    static_feature_names: Sequence[str] = (),
) -> Tuple[Mapping[str, object], ...]:
    """Aggregate absolute magnitude without truncation or normalization."""

    if len(attributions) != len(attributable_inputs(batch)):
        raise IGInputError("attribution component count mismatch")
    rows = []
    component_names = ("value", "observation_mask")
    for component_index, component_name in enumerate(component_names):
        tensor = attributions[component_index]
        if tensor.shape != batch.sequence.shape:
            raise IGInputError("temporal attribution shape mismatch")
        scores = tensor.abs().sum(dim=(0, 1))
        rows.extend(
            {
                "feature_name": component_name + ":" + name,
                "absolute_attribution": float(scores[index].item()),
            }
            for index, name in enumerate(batch.feature_names)
        )
    offset = 2
    if batch.tslo is not None:
        tensor = attributions[offset]
        if tensor.shape != batch.tslo.shape:
            raise IGInputError("TSLO attribution shape mismatch")
        scores = tensor.abs().sum(dim=(0, 1))
        rows.extend(
            {
                "feature_name": "tslo:" + name,
                "absolute_attribution": float(scores[index].item()),
            }
            for index, name in enumerate(batch.feature_names)
        )
        offset += 1
    if batch.static_features is not None:
        tensor = attributions[offset]
        if tensor.shape != batch.static_features.shape:
            raise IGInputError("static attribution shape mismatch")
        if len(static_feature_names) != tensor.shape[1]:
            raise IGInputError("static feature-name order mismatch")
        scores = tensor.abs().sum(dim=0)
        rows.extend(
            {
                "feature_name": "static:" + name,
                "absolute_attribution": float(scores[index].item()),
            }
            for index, name in enumerate(static_feature_names)
        )
    return tuple(rows)


def _captum_integrated_gradients_class():
    try:
        from captum.attr import IntegratedGradients
    except ImportError as error:
        raise CaptumDependencyError(
            "BLOCKED — CAPTUM DEPENDENCY REQUIRED; no fallback IG is permitted"
        ) from error
    return IntegratedGradients


class IntegratedGradientsAdapter:
    method = "integrated_gradients"
    supported_families = ("gru",)

    def __init__(
        self,
        *,
        baseline_provider: BaselineProvider,
        config: IGConfig,
        targets: Mapping[str, AttributionTarget],
        synthetic: bool,
    ) -> None:
        config.validate()
        self.baseline_provider = baseline_provider
        self.config = config
        self.targets = dict(targets)
        self.synthetic = synthetic
        if not synthetic:
            raise IGBaselineError(
                "BLOCKED — REAL IG BASELINE AND OUTPUT DOMAINS MUST BE FROZEN"
            )
        if baseline_provider.identity.scientific_scope != "synthetic_development_only":
            raise IGBaselineError("synthetic adapter requires synthetic baseline scope")

    def explain(self, context: ExplanationContext) -> AdapterExplanation:
        if context.family != "gru" or getattr(context.model, "family", None) != "gru":
            raise IGInputError("Integrated Gradients accepts selected GRU models only")
        if context.model_sha256 != getattr(context.model, "artifact_sha256", None):
            raise IGInputError("model hash mismatch")
        if context.prediction_time != context.input_prediction_time:
            raise IGInputError("prediction and input cutoff mismatch")
        if not context.synthetic or not self.synthetic:
            raise IGBaselineError("synthetic IG policy cannot authorize real explanation")
        if not isinstance(context.model, torch.nn.Module):
            raise IGInputError("IG model must be a differentiable torch module")
        if not isinstance(context.prepared_input, CanonicalBatch):
            raise IGInputError("IG requires the exact prepared CanonicalBatch")
        batch = context.prepared_input
        if len(batch.feature_names) != batch.sequence.shape[-1]:
            raise IGInputError("feature schema/order does not match sequence channels")
        if batch.versions.get("feature_schema_version") != context.feature_schema_version:
            raise IGInputError("prepared input feature schema version mismatch")
        times = batch.identifiers.get("prediction_time", ())
        if tuple(times) != (context.prediction_time,):
            raise IGInputError("prepared input cutoff mismatch")
        inputs = attributable_inputs(batch)
        if any(not torch.is_floating_point(value) or not torch.isfinite(value).all() for value in inputs):
            raise IGInputError("all attributable model inputs must be finite floating tensors")
        target = self.targets.get(context.task)
        if target is None or context.explanation_target != target.output_name:
            raise IGTargetError("explicit approved attribution target is required")
        target.validate(context.model, context.synthetic)
        identity = self.baseline_provider.identity
        if identity.feature_schema_version != context.feature_schema_version:
            raise IGBaselineError("baseline feature schema mismatch")
        baselines = self.baseline_provider.baselines(batch)
        if len(baselines) != len(inputs) or any(
            baseline.shape != value.shape for baseline, value in zip(baselines, inputs)
        ):
            raise IGBaselineError("baseline tensors must exactly match input shapes")
        if any(not torch.isfinite(value).all() for value in baselines):
            raise IGBaselineError("baseline tensors must be finite")

        model = context.model
        state_before = {name: value.detach().clone() for name, value in model.state_dict().items()}
        was_training = model.training
        wrapper = GRUForwardWrapper(model, batch, target)
        IntegratedGradients = _captum_integrated_gradients_class()
        try:
            model.eval()
            ig = IntegratedGradients(wrapper)
            result = ig.attribute(
                inputs=inputs,
                baselines=baselines,
                n_steps=self.config.n_steps,
                method=self.config.method,
                internal_batch_size=self.config.internal_batch_size,
                return_convergence_delta=self.config.return_convergence_delta,
            )
        finally:
            model.train(was_training)
        if self.config.return_convergence_delta:
            attributions, delta = result
        else:
            attributions, delta = result, None
        if isinstance(attributions, torch.Tensor):
            attributions = (attributions,)
        attributions = tuple(attributions)
        if len(attributions) != len(inputs) or any(
            attribution.shape != value.shape
            for attribution, value in zip(attributions, inputs)
        ):
            raise IGInputError("Captum attribution shape mismatch")
        if any(not torch.isfinite(value).all() for value in attributions):
            raise IGInputError("Captum returned non-finite attribution")
        if any(not torch.equal(state_before[name], value) for name, value in model.state_dict().items()):
            raise IGInputError("model parameters changed during explanation")
        static_names = ()
        if context.feature_metadata is not None:
            static_names = tuple(context.feature_metadata.get("static_feature_names", ()))
        aggregate = aggregate_absolute_attribution(
            attributions, batch, static_feature_names=static_names
        )
        items = tuple(
            {
                "feature_name": str(row["feature_name"]),
                "attribution": float(row["absolute_attribution"]),
            }
            for row in aggregate
        )
        details = {
            "scientific_scope": "synthetic_non_scientific",
            "baseline": asdict(identity),
            "ig_config": {**asdict(self.config), "sha256": self.config.sha256},
            "target": asdict(target),
            "raw_signed_attribution": [value.detach().cpu().tolist() for value in attributions],
            "absolute_aggregation": list(aggregate),
            "convergence_delta": None if delta is None else delta.detach().cpu().tolist(),
            "convergence_delta_semantics": "numerical_ig_completeness_diagnostic_not_confidence",
        }
        return AdapterExplanation(
            task=context.task,
            family="gru",
            method=self.method,
            model_sha256=context.model_sha256,
            prediction_time=context.prediction_time,
            manifest_version=context.manifest_version,
            manifest_sha256=context.manifest_sha256,
            feature_schema_version=context.feature_schema_version,
            synthetic=True,
            items=items,
            details=details,
        )
