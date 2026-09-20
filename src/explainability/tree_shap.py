"""SHAP TreeExplainer adapter for validated flattened XGBoost inputs.

Production execution is blocked until the Sanskruti XGBoost serving loader,
flattened-name contract, output domains, and governed dependencies are present.
There is deliberately no homemade SHAP fallback and no training code here.
"""

from dataclasses import asdict, dataclass
from typing import Mapping, Optional, Sequence, Tuple

import numpy as np

from explainability.router import AdapterExplanation, ExplanationContext
from experiments.search_governance import canonical_sha256


class TreeShapError(RuntimeError):
    pass


class ShapDependencyError(TreeShapError):
    pass


class TreeShapInputError(TreeShapError):
    pass


class TreeShapTargetError(TreeShapError):
    pass


class FlattenedFeatureContractError(TreeShapError):
    pass


CHANNEL_TYPES = ("value", "observation_mask", "tslo", "padding", "static")


@dataclass(frozen=True)
class FlattenedFeatureIdentity:
    """Explicit synthetic reverse mapping; names are never parsed heuristically."""

    flat_name: str
    base_feature: str
    channel_type: str
    bin_index: Optional[int]
    relative_start_hours: Optional[int]
    relative_end_hours: Optional[int]
    aggregation: Optional[str]

    def validate(self) -> None:
        if not self.flat_name or not self.base_feature or self.channel_type not in CHANNEL_TYPES:
            raise FlattenedFeatureContractError("invalid flattened feature identity")
        temporal = self.channel_type in ("value", "observation_mask", "tslo", "padding")
        if temporal:
            if self.bin_index is None or not 0 <= self.bin_index < 8:
                raise FlattenedFeatureContractError("temporal feature requires bin_index 0..7")
            expected_start = -48 + 6 * self.bin_index
            expected_end = expected_start + 6
            if (
                self.relative_start_hours != expected_start
                or self.relative_end_hours != expected_end
            ):
                raise FlattenedFeatureContractError("relative bin interval mismatch")
        elif any(
            value is not None
            for value in (
                self.bin_index,
                self.relative_start_hours,
                self.relative_end_hours,
            )
        ):
            raise FlattenedFeatureContractError("static feature cannot have a temporal bin")


@dataclass(frozen=True)
class PreparedTreeInput:
    values: np.ndarray
    feature_identities: Tuple[FlattenedFeatureIdentity, ...]
    feature_schema_version: str
    flattening_version: str
    flattening_sha256: str
    prediction_time: str
    task: str
    family: str = "xgboost"

    def validate(self) -> None:
        if not isinstance(self.values, np.ndarray) or self.values.ndim != 2 or self.values.shape[0] != 1:
            raise TreeShapInputError("prepared XGBoost input must have shape [1,D]")
        if not np.issubdtype(self.values.dtype, np.number) or not np.isfinite(self.values).all():
            raise TreeShapInputError("prepared XGBoost input must be finite numeric")
        if len(self.feature_identities) != self.values.shape[1]:
            raise FlattenedFeatureContractError("feature-name count does not match input dimension")
        names = tuple(item.flat_name for item in self.feature_identities)
        if len(set(names)) != len(names):
            raise FlattenedFeatureContractError("flattened feature names must be unique")
        for item in self.feature_identities:
            item.validate()
        if not self.flattening_version or len(self.flattening_sha256) != 64:
            raise FlattenedFeatureContractError("flattening identity is incomplete")


@dataclass(frozen=True)
class TreeShapTarget:
    task: str
    output_name: str
    output_index: int
    model_output_domain: str
    scientific_scope: str

    def validate(self, model: object, synthetic: bool) -> None:
        if self.task not in ("recovery", "icu_stay_time", "organ_support"):
            raise TreeShapTargetError("unsupported TreeSHAP task")
        if self.task != getattr(model, "task", None):
            raise TreeShapTargetError("TreeSHAP target/model task mismatch")
        if self.output_index < 0:
            raise TreeShapTargetError("output index must be nonnegative")
        if self.task != "recovery" and self.output_index != 0:
            raise TreeShapTargetError("scalar task output index must be zero")
        if not synthetic:
            raise TreeShapTargetError("real TreeSHAP output domains are unresolved")
        if (
            self.scientific_scope != "synthetic_development_only"
            or not self.model_output_domain.startswith("SYNTHETIC_")
        ):
            raise TreeShapTargetError("synthetic target must be development-only")
        outputs = getattr(model, "output_names", None)
        if not isinstance(outputs, tuple) or self.output_index >= len(outputs):
            raise TreeShapTargetError("model output-name contract is absent or incompatible")
        if outputs[self.output_index] != self.output_name:
            raise TreeShapTargetError("output name/index does not match model contract")


@dataclass(frozen=True)
class TreeShapConfig:
    version: str
    feature_perturbation: str
    model_output: str
    background_policy: str
    additivity_tolerance: float
    scientific_scope: str

    def validate(self) -> None:
        if self.scientific_scope != "synthetic_development_only":
            raise TreeShapError("real TreeSHAP configuration is not frozen")
        if not self.version.startswith("SYNTHETIC_"):
            raise TreeShapError("synthetic TreeSHAP config must be unmistakably synthetic")
        if self.background_policy != "none_tree_path_dependent_development_only":
            raise TreeShapError("unauthorized TreeSHAP background policy")
        if self.feature_perturbation != "tree_path_dependent" or self.model_output != "raw":
            raise TreeShapError("unsupported synthetic TreeSHAP configuration")
        if not np.isfinite(self.additivity_tolerance) or self.additivity_tolerance <= 0:
            raise TreeShapError("additivity tolerance must be finite and positive")

    @property
    def sha256(self) -> str:
        return canonical_sha256(asdict(self))


def aggregate_absolute_shap(
    signed_values: Sequence[float],
    identities: Sequence[FlattenedFeatureIdentity],
) -> Tuple[Mapping[str, object], ...]:
    """Sum absolute SHAP by explicit base feature without discarding raw values."""

    if len(signed_values) != len(identities):
        raise FlattenedFeatureContractError("SHAP/feature dimension mismatch")
    ordered = []
    totals = {}
    for value, identity in zip(signed_values, identities):
        if not np.isfinite(value):
            raise TreeShapInputError("SHAP values must be finite")
        key = (identity.base_feature, identity.channel_type, identity.aggregation)
        if key not in totals:
            ordered.append(key)
            totals[key] = 0.0
        totals[key] += abs(float(value))
    return tuple(
        {
            "base_feature": key[0],
            "channel_type": key[1],
            "aggregation": key[2],
            "absolute_attribution": totals[key],
        }
        for key in ordered
    )


def normalize_shap_output(
    values: object,
    base_values: object,
    *,
    output_index: int,
    feature_dim: int,
) -> Tuple[np.ndarray, float]:
    """Normalize supported one-sample SHAP shapes with explicit output selection."""

    if isinstance(values, list):
        if output_index >= len(values):
            raise TreeShapTargetError("SHAP output index is unavailable")
        selected = np.asarray(values[output_index])
    else:
        selected = np.asarray(values)
        if selected.ndim == 3:
            if output_index >= selected.shape[2]:
                raise TreeShapTargetError("SHAP output index is unavailable")
            selected = selected[:, :, output_index]
        elif selected.ndim == 2 and output_index != 0:
            raise TreeShapTargetError("scalar SHAP output cannot use nonzero index")
    if selected.shape != (1, feature_dim):
        raise FlattenedFeatureContractError("SHAP attribution dimension mismatch")
    base = np.asarray(base_values)
    if base.ndim == 0:
        if output_index != 0:
            raise TreeShapTargetError("scalar base value cannot use nonzero index")
        selected_base = float(base)
    elif base.ndim == 1:
        if base.size == 1:
            if output_index != 0:
                raise TreeShapTargetError("scalar base value cannot use nonzero index")
            selected_base = float(base[0])
        else:
            if output_index >= base.size:
                raise TreeShapTargetError("base-value output index is unavailable")
            selected_base = float(base[output_index])
    elif base.ndim == 2 and base.shape[0] == 1:
        if output_index >= base.shape[1]:
            raise TreeShapTargetError("base-value output index is unavailable")
        selected_base = float(base[0, output_index])
    else:
        raise TreeShapTargetError("unsupported SHAP base-value shape")
    if not np.isfinite(selected).all() or not np.isfinite(selected_base):
        raise TreeShapInputError("SHAP returned non-finite values")
    return selected[0].astype(float, copy=False), selected_base


def _tree_explainer_class():
    try:
        import shap
    except (ImportError, OSError, RuntimeError) as error:
        raise ShapDependencyError(
            "BLOCKED — SHAP DEPENDENCY REQUIRED; no fallback TreeSHAP is permitted"
        ) from error
    return shap.TreeExplainer, getattr(shap, "__version__", "UNKNOWN")


def _snapshot_model(model: object) -> bytes:
    explicit = getattr(model, "explanation_state_bytes", None)
    if callable(explicit):
        value = explicit()
    else:
        booster = model.get_booster() if callable(getattr(model, "get_booster", None)) else model
        save_raw = getattr(booster, "save_raw", None)
        if not callable(save_raw):
            raise TreeShapInputError("XGBoost model state snapshot interface is unavailable")
        value = save_raw()
    if not isinstance(value, (bytes, bytearray)):
        raise TreeShapInputError("model state snapshot must be bytes")
    return bytes(value)


class TreeShapAdapter:
    method = "tree_shap"
    supported_families = ("xgboost",)

    def __init__(
        self,
        *,
        config: TreeShapConfig,
        targets: Mapping[str, TreeShapTarget],
        synthetic: bool,
    ) -> None:
        config.validate()
        self.config = config
        self.targets = dict(targets)
        self.synthetic = synthetic
        if not synthetic:
            raise TreeShapError(
                "BLOCKED — REAL XGBOOST CONTRACT AND TREESHAP POLICIES REQUIRED"
            )

    def explain(self, context: ExplanationContext) -> AdapterExplanation:
        if context.family != "xgboost" or getattr(context.model, "family", None) != "xgboost":
            raise TreeShapInputError("TreeSHAP accepts selected XGBoost models only")
        if context.task != getattr(context.model, "task", None):
            raise TreeShapInputError("TreeSHAP task/model mismatch")
        if context.model_sha256 != getattr(context.model, "artifact_sha256", None):
            raise TreeShapInputError("model hash mismatch")
        if context.prediction_time != context.input_prediction_time:
            raise TreeShapInputError("prediction and prepared-input cutoff mismatch")
        if not context.synthetic or not self.synthetic:
            raise TreeShapError("synthetic TreeSHAP cannot authorize real explanation")
        if not isinstance(context.prepared_input, PreparedTreeInput):
            raise TreeShapInputError("TreeSHAP requires exact prepared flattened input")
        prepared = context.prepared_input
        prepared.validate()
        if prepared.task != context.task or prepared.family != "xgboost":
            raise TreeShapInputError("prepared input task/family mismatch")
        if prepared.prediction_time != context.prediction_time:
            raise TreeShapInputError("prepared input cutoff mismatch")
        if prepared.feature_schema_version != context.feature_schema_version:
            raise FlattenedFeatureContractError("feature schema version mismatch")
        model_names = tuple(getattr(context.model, "flattened_feature_names", ()))
        actual_names = tuple(item.flat_name for item in prepared.feature_identities)
        if model_names != actual_names:
            raise FlattenedFeatureContractError("flattened feature order mismatch")
        if getattr(context.model, "input_dim", None) != prepared.values.shape[1]:
            raise FlattenedFeatureContractError("model input dimension mismatch")
        if getattr(context.model, "feature_schema_version", None) != prepared.feature_schema_version:
            raise FlattenedFeatureContractError("model feature schema mismatch")
        if getattr(context.model, "flattening_version", None) != prepared.flattening_version:
            raise FlattenedFeatureContractError("flattening version mismatch")
        if getattr(context.model, "flattening_sha256", None) != prepared.flattening_sha256:
            raise FlattenedFeatureContractError("flattening hash mismatch")
        target = self.targets.get(context.task)
        if target is None or context.explanation_target != target.output_name:
            raise TreeShapTargetError("explicit approved TreeSHAP target is required")
        target.validate(context.model, context.synthetic)

        before = _snapshot_model(context.model)
        before_contract = (
            tuple(getattr(context.model, "flattened_feature_names", ())),
            getattr(context.model, "input_dim", None),
            getattr(context.model, "feature_schema_version", None),
            getattr(context.model, "flattening_version", None),
            getattr(context.model, "flattening_sha256", None),
        )
        TreeExplainer, shap_version = _tree_explainer_class()
        explainer = TreeExplainer(
            context.model,
            data=None,
            feature_perturbation=self.config.feature_perturbation,
            model_output=self.config.model_output,
        )
        explanation = explainer(prepared.values)
        signed, base_value = normalize_shap_output(
            explanation.values,
            explanation.base_values,
            output_index=target.output_index,
            feature_dim=prepared.values.shape[1],
        )
        after_contract = (
            tuple(getattr(context.model, "flattened_feature_names", ())),
            getattr(context.model, "input_dim", None),
            getattr(context.model, "feature_schema_version", None),
            getattr(context.model, "flattening_version", None),
            getattr(context.model, "flattening_sha256", None),
        )
        if _snapshot_model(context.model) != before or after_contract != before_contract:
            raise TreeShapInputError("model artifact changed during TreeSHAP")
        predicted = np.asarray(context.raw_output)
        if predicted.ndim == 0:
            predicted_value = float(predicted)
        else:
            flat = predicted.reshape(-1)
            if target.output_index >= flat.size:
                raise TreeShapTargetError("prediction output index is unavailable")
            predicted_value = float(flat[target.output_index])
        reconstructed = base_value + float(signed.sum())
        if not np.isfinite(predicted_value) or not np.isclose(
            reconstructed,
            predicted_value,
            rtol=0.0,
            atol=self.config.additivity_tolerance,
        ):
            raise TreeShapInputError("TreeSHAP additivity/model-output coherence failed")
        aggregate = aggregate_absolute_shap(signed, prepared.feature_identities)
        items = tuple(
            {
                "feature_name": identity.flat_name,
                "attribution": float(value),
            }
            for identity, value in zip(prepared.feature_identities, signed)
        )
        details = {
            "scientific_scope": "synthetic_non_scientific",
            "tree_shap_config": {**asdict(self.config), "sha256": self.config.sha256},
            "target": asdict(target),
            "shap_version": shap_version,
            "xgboost_version": getattr(context.model, "xgboost_version", "UNAVAILABLE"),
            "flattening_version": prepared.flattening_version,
            "flattening_sha256": prepared.flattening_sha256,
            "feature_identities": [asdict(item) for item in prepared.feature_identities],
            "raw_signed_shap_values": signed.tolist(),
            "base_value": base_value,
            "model_output": predicted_value,
            "additivity_residual": predicted_value - reconstructed,
            "absolute_aggregation": list(aggregate),
        }
        return AdapterExplanation(
            task=context.task,
            family="xgboost",
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
