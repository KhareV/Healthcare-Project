"""Real (Stage-4) explanation adapters bound to the frozen selected models.

Two adapters are built here:

- ``RealTreeShapAdapter`` (method ``tree_shap``, family ``xgboost``) explains
  both xgboost tasks: recovery (two independent horizons, one TreeSHAP pass
  per frozen sub-model, affine-rescaled into original SOFA-delta units) and
  organ support (one TreeSHAP pass on the selected classifier's raw margin,
  explicitly documented as pre-calibration).
- the real Integrated Gradients adapter for ICU time is assembled with
  ``explainability.ig.IntegratedGradientsAdapter`` directly (no new class
  needed — that module was extended in Stage 4 to support this scope).

Neither adapter reloads a model, rebuilds features, runs a second
preprocessor, or touches labels/future history. Every attribution is
computed from exactly the predictor object and prepared input the pipeline
already used for prediction.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Mapping, Tuple

import numpy as np

from explainability.ig import (
    IGConfig,
    IntegratedGradientsAdapter,
    RealICUBaselineProvider,
    REAL_SCOPE as IG_REAL_SCOPE,
    AttributionTarget,
)
from explainability.router import AdapterExplanation, ExplanationContext
from explainability.tree_shap import (
    PreparedTreeInput,
    TreeShapConfig,
    TreeShapError,
    TreeShapInputError,
    TreeShapTarget,
    TreeShapTargetError,
    _tree_explainer_class,
    aggregate_absolute_shap,
    normalize_shap_output,
)
from models.gru_recovery import RECOVERY_HORIZON_ORDER
from serving.recovery import ORIGINAL_DELTA_DOMAIN


TREE_SHAP_REAL_SCOPE = "real_frozen_v1"
SUPPORT_OUTPUT_NAME = "organ_support_probability"
RECOVERY_EXPLANATION_TARGET = "delta_sofa_24h+delta_sofa_48h"


def real_tree_shap_config() -> TreeShapConfig:
    return TreeShapConfig(
        version="STAGE4_TREE_SHAP_CONFIG_V1",
        feature_perturbation="tree_path_dependent",
        model_output="raw",
        background_policy="none_tree_path_dependent_frozen_real_v1",
        additivity_tolerance=1e-3,
        scientific_scope=TREE_SHAP_REAL_SCOPE,
    )


def real_tree_shap_targets() -> Mapping[str, TreeShapTarget]:
    return {
        "recovery": TreeShapTarget(
            task="recovery",
            output_name=RECOVERY_HORIZON_ORDER[0],
            output_index=0,
            model_output_domain=ORIGINAL_DELTA_DOMAIN,
            scientific_scope=TREE_SHAP_REAL_SCOPE,
        ),
        "organ_support": TreeShapTarget(
            task="organ_support",
            output_name=SUPPORT_OUTPUT_NAME,
            output_index=0,
            model_output_domain="raw_margin_pre_calibration",
            scientific_scope=TREE_SHAP_REAL_SCOPE,
        ),
    }


def _validate_common(context: ExplanationContext, prepared: PreparedTreeInput) -> None:
    if context.family != "xgboost" or getattr(context.model, "family", None) != "xgboost":
        raise TreeShapInputError("TreeSHAP accepts selected XGBoost models only")
    if context.task != getattr(context.model, "task", None):
        raise TreeShapInputError("TreeSHAP task/model mismatch")
    if context.model_sha256 != getattr(context.model, "artifact_sha256", None):
        raise TreeShapInputError("model hash mismatch")
    if context.prediction_time != context.input_prediction_time:
        raise TreeShapInputError("prediction and prepared-input cutoff mismatch")
    if context.synthetic:
        raise TreeShapError("real TreeSHAP adapter cannot explain a synthetic context")
    if not isinstance(prepared, PreparedTreeInput):
        raise TreeShapInputError("TreeSHAP requires exact prepared flattened input")
    prepared.validate()
    if prepared.task != context.task or prepared.family != "xgboost":
        raise TreeShapInputError("prepared input task/family mismatch")
    if prepared.prediction_time != context.prediction_time:
        raise TreeShapInputError("prepared input cutoff mismatch")
    if prepared.feature_schema_version != context.feature_schema_version:
        raise TreeShapInputError("feature schema version mismatch")
    model_names = tuple(getattr(context.model, "flattened_feature_names", ()))
    actual_names = tuple(item.flat_name for item in prepared.feature_identities)
    if model_names != actual_names:
        raise TreeShapInputError("flattened feature order mismatch")
    if getattr(context.model, "input_dim", None) != prepared.values.shape[1]:
        raise TreeShapInputError("model input dimension mismatch")
    if getattr(context.model, "feature_schema_version", None) != prepared.feature_schema_version:
        raise TreeShapInputError("model feature schema mismatch")
    if getattr(context.model, "flattening_version", None) != prepared.flattening_version:
        raise TreeShapInputError("flattening version mismatch")
    if getattr(context.model, "flattening_sha256", None) != prepared.flattening_sha256:
        raise TreeShapInputError("flattening hash mismatch")


class RealTreeShapAdapter:
    """Composite adapter: routes recovery (dual-horizon) and organ_support
    (single, margin-space) through explicit, task-specific TreeSHAP passes.
    """

    method = "tree_shap"
    supported_families = ("xgboost",)
    synthetic = False

    def __init__(self, *, config: TreeShapConfig, targets: Mapping[str, TreeShapTarget]) -> None:
        config.validate()
        if config.scientific_scope != TREE_SHAP_REAL_SCOPE:
            raise TreeShapError("real TreeSHAP adapter requires the frozen real scope")
        for name in ("recovery", "organ_support"):
            target = targets.get(name)
            if target is None or target.scientific_scope != TREE_SHAP_REAL_SCOPE:
                raise TreeShapTargetError("real TreeSHAP adapter requires both xgboost task targets")
        self.config = config
        self.targets = dict(targets)

    def explain(self, context: ExplanationContext) -> AdapterExplanation:
        prepared = context.prepared_input
        _validate_common(context, prepared)
        expected_target = (
            RECOVERY_EXPLANATION_TARGET if context.task == "recovery" else SUPPORT_OUTPUT_NAME
        )
        if context.explanation_target != expected_target:
            raise TreeShapTargetError("explicit approved TreeSHAP target is required")
        tree_models = getattr(context.model, "tree_shap_models", None)
        if not isinstance(tree_models, Mapping) or not tree_models:
            raise TreeShapInputError("selected model exposes no TreeSHAP-explainable sub-models")
        TreeExplainer, shap_version = _tree_explainer_class()
        if context.task == "recovery":
            return self._explain_recovery(context, prepared, tree_models, TreeExplainer, shap_version)
        return self._explain_support(context, prepared, tree_models, TreeExplainer, shap_version)

    def _one_pass(self, TreeExplainer, shap_version, model, prepared: PreparedTreeInput):
        before = model.get_booster().save_raw()
        explainer = TreeExplainer(
            model,
            data=None,
            feature_perturbation=self.config.feature_perturbation,
            model_output=self.config.model_output,
        )
        explanation = explainer(prepared.values)
        signed, base_value = normalize_shap_output(
            explanation.values, explanation.base_values, output_index=0, feature_dim=prepared.values.shape[1]
        )
        after = model.get_booster().save_raw()
        if after != before:
            raise TreeShapInputError("model artifact changed during TreeSHAP")
        margin_prediction = float(np.asarray(model.predict(prepared.values, output_margin=True)).reshape(-1)[0])
        return signed, base_value, margin_prediction

    def _explain_recovery(self, context, prepared, tree_models, TreeExplainer, shap_version):
        scaler = getattr(context.model, "scaler", None)
        if scaler is None:
            raise TreeShapInputError("recovery model exposes no target scaler for SHAP rescaling")
        raw_output = context.raw_output
        combined_items = []
        per_horizon = {}
        for horizon_index, output_name in enumerate(RECOVERY_HORIZON_ORDER):
            model = tree_models.get(output_name)
            if model is None:
                raise TreeShapInputError("recovery model is missing the " + output_name + " sub-model")
            signed, base_value, margin_prediction = self._one_pass(TreeExplainer, shap_version, model, prepared)
            scale = scaler.scale_24 if horizon_index == 0 else scaler.scale_48
            mean = scaler.mean_24 if horizon_index == 0 else scaler.mean_48
            signed_original = signed * scale
            base_original = base_value * scale + mean
            predicted_value = raw_output.delta_24h if horizon_index == 0 else raw_output.delta_48h
            reconstructed = base_original + float(signed_original.sum())
            if not np.isfinite(predicted_value) or not np.isclose(
                reconstructed, predicted_value, rtol=0.0, atol=self.config.additivity_tolerance
            ):
                raise TreeShapInputError("TreeSHAP additivity/model-output coherence failed for " + output_name)
            aggregate = aggregate_absolute_shap(signed_original, prepared.feature_identities)
            combined_items.extend(
                {"feature_name": output_name + ":" + identity.flat_name, "attribution": float(value)}
                for identity, value in zip(prepared.feature_identities, signed_original)
            )
            per_horizon[output_name] = {
                "base_value_original_units": base_original,
                "model_output_margin_standardized": margin_prediction,
                "reconstructed_original_units": reconstructed,
                "predicted_value_original_units": predicted_value,
                "absolute_aggregation": list(aggregate),
                "output_units": ORIGINAL_DELTA_DOMAIN,
                "rescale": "affine_shap_times_scale_plus_mean_applied_to_standardized_margin_shap",
            }
        details = {
            "scientific_scope": TREE_SHAP_REAL_SCOPE,
            "tree_shap_config": {**asdict(self.config), "sha256": self.config.sha256},
            "shap_version": shap_version,
            "xgboost_version": getattr(context.model, "xgboost_version", "UNAVAILABLE"),
            "flattening_version": prepared.flattening_version,
            "flattening_sha256": prepared.flattening_sha256,
            "feature_identities": [asdict(item) for item in prepared.feature_identities],
            "per_horizon": per_horizon,
            "note": (
                "Each horizon's XGBoost model is explained independently in its "
                "own frozen, standardized margin space, then SHAP values and the "
                "base value are mapped through the exact frozen affine recovery "
                "target scaler into original DeltaSOFA units. Predictions are "
                "never altered by this mapping."
            ),
        }
        return AdapterExplanation(
            task="recovery",
            family="xgboost",
            method=self.method,
            model_sha256=context.model_sha256,
            prediction_time=context.prediction_time,
            manifest_version=context.manifest_version,
            manifest_sha256=context.manifest_sha256,
            feature_schema_version=context.feature_schema_version,
            synthetic=False,
            items=tuple(combined_items),
            details=details,
        )

    def _explain_support(self, context, prepared, tree_models, TreeExplainer, shap_version):
        model = tree_models.get(SUPPORT_OUTPUT_NAME)
        if model is None:
            raise TreeShapInputError("support model is missing its TreeSHAP-explainable sub-model")
        signed, base_value, margin_prediction = self._one_pass(TreeExplainer, shap_version, model, prepared)
        reconstructed = base_value + float(signed.sum())
        if not np.isfinite(margin_prediction) or not np.isclose(
            reconstructed, margin_prediction, rtol=0.0, atol=self.config.additivity_tolerance
        ):
            raise TreeShapInputError("TreeSHAP additivity/model-output coherence failed")
        aggregate = aggregate_absolute_shap(signed, prepared.feature_identities)
        items = tuple(
            {"feature_name": identity.flat_name, "attribution": float(value)}
            for identity, value in zip(prepared.feature_identities, signed)
        )
        details = {
            "scientific_scope": TREE_SHAP_REAL_SCOPE,
            "tree_shap_config": {**asdict(self.config), "sha256": self.config.sha256},
            "shap_version": shap_version,
            "xgboost_version": getattr(context.model, "xgboost_version", "UNAVAILABLE"),
            "flattening_version": prepared.flattening_version,
            "flattening_sha256": prepared.flattening_sha256,
            "feature_identities": [asdict(item) for item in prepared.feature_identities],
            "raw_signed_shap_values": signed.tolist(),
            "base_value": base_value,
            "model_output_margin": margin_prediction,
            "additivity_residual": margin_prediction - reconstructed,
            "absolute_aggregation": list(aggregate),
            "attribution_domain": "raw_margin_pre_calibration",
            "note": (
                "This TreeSHAP explanation decomposes the selected XGBoost "
                "support model's raw margin output before isotonic "
                "calibration. It does not decompose the calibrated "
                "probability reported by the API, since isotonic regression "
                "is a non-linear, non-additive transform."
            ),
        }
        return AdapterExplanation(
            task="organ_support",
            family="xgboost",
            method=self.method,
            model_sha256=context.model_sha256,
            prediction_time=context.prediction_time,
            manifest_version=context.manifest_version,
            manifest_sha256=context.manifest_sha256,
            feature_schema_version=context.feature_schema_version,
            synthetic=False,
            items=items,
            details=details,
        )


def real_ig_config() -> IGConfig:
    return IGConfig(
        version="REAL_IG_CONFIG_STAGE4_V1",
        n_steps=64,
        method="gausslegendre",
        internal_batch_size=None,
        return_convergence_delta=True,
        scientific_scope=IG_REAL_SCOPE,
    )


def real_ig_target() -> AttributionTarget:
    return AttributionTarget(
        task="icu_stay_time",
        output_name="icu_stay_time_hours",
        output_index=0,
        output_domain_version="remaining_icu_hours_postprocessed_v1",
        scientific_scope=IG_REAL_SCOPE,
    )


def build_real_ig_adapter(*, feature_schema_version: str, tslo_no_observation_value: float) -> IntegratedGradientsAdapter:
    baseline_provider = RealICUBaselineProvider(
        feature_schema_version=feature_schema_version,
        tslo_no_observation_value=tslo_no_observation_value,
    )
    return IntegratedGradientsAdapter(
        baseline_provider=baseline_provider,
        config=real_ig_config(),
        targets={"icu_stay_time": real_ig_target()},
        synthetic=False,
    )


def build_real_tree_shap_adapter() -> RealTreeShapAdapter:
    return RealTreeShapAdapter(config=real_tree_shap_config(), targets=real_tree_shap_targets())
