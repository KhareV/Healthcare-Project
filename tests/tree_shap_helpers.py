from dataclasses import replace

import numpy as np

from explainability.router import ExplanationContext
from explainability.tree_shap import (
    FlattenedFeatureIdentity,
    PreparedTreeInput,
    TreeShapConfig,
    TreeShapTarget,
)


CUTOFF = "2026-01-04T12:00:00+00:00"
FEATURE_VERSION = "SYNTHETIC_FEATURES_PHASE10_NOT_REAL"
FLATTENING_VERSION = "SYNTHETIC_FLATTENING_PHASE10_V1"
FLATTENING_HASH = "7" * 64
MODEL_HASH = "6" * 64


def identities():
    return (
        FlattenedFeatureIdentity("A_value_oldest", "A", "value", 0, -48, -42, "latest"),
        FlattenedFeatureIdentity("A_value_newest", "A", "value", 7, -6, 0, "latest"),
        FlattenedFeatureIdentity("B_value_bin3", "B", "value", 3, -30, -24, "mean"),
        FlattenedFeatureIdentity("A_observed_newest", "A", "observation_mask", 7, -6, 0, None),
        FlattenedFeatureIdentity("A_tslo_newest", "A", "tslo", 7, -6, 0, None),
        FlattenedFeatureIdentity("padding_oldest", "padding", "padding", 0, -48, -42, None),
        FlattenedFeatureIdentity("static_X", "X", "static", None, None, None, None),
    )


def prepared(task="icu_stay_time"):
    return PreparedTreeInput(
        values=np.asarray([[1.0, 2.0, -1.0, 1.0, 6.0, 0.0, 44.0]], dtype=np.float32),
        feature_identities=identities(),
        feature_schema_version=FEATURE_VERSION,
        flattening_version=FLATTENING_VERSION,
        flattening_sha256=FLATTENING_HASH,
        prediction_time=CUTOFF,
        task=task,
    )


class SyntheticXGBContract:
    family = "xgboost"
    artifact_sha256 = MODEL_HASH
    feature_schema_version = FEATURE_VERSION
    flattening_version = FLATTENING_VERSION
    flattening_sha256 = FLATTENING_HASH
    xgboost_version = "SYNTHETIC_TEST_DOUBLE_NOT_XGBOOST"

    def __init__(self, task="icu_stay_time", output_names=("remaining_icu_log1p_hours",)):
        self.task = task
        self.output_names = tuple(output_names)
        self.flattened_feature_names = tuple(item.flat_name for item in identities())
        self.input_dim = len(self.flattened_feature_names)
        self._state = b"synthetic-immutable-model-state"

    def explanation_state_bytes(self):
        return self._state


def config():
    return TreeShapConfig(
        version="SYNTHETIC_TREESHAP_CONFIG_V1",
        feature_perturbation="tree_path_dependent",
        model_output="raw",
        background_policy="none_tree_path_dependent_development_only",
        additivity_tolerance=1e-6,
        scientific_scope="synthetic_development_only",
    )


def target(task="icu_stay_time", name="remaining_icu_log1p_hours", index=0):
    return TreeShapTarget(
        task=task,
        output_name=name,
        output_index=index,
        model_output_domain="SYNTHETIC_RAW_MODEL_OUTPUT_V1",
        scientific_scope="synthetic_development_only",
    )


def context(model, *, target_name=None, prepared_input=None, raw_output=2.0):
    prepared_input = prepared_input or prepared(model.task)
    return ExplanationContext(
        task=model.task,
        input_task=model.task,
        family="xgboost",
        input_family="xgboost",
        model_sha256=MODEL_HASH,
        prediction_time=CUTOFF,
        input_prediction_time=CUTOFF,
        manifest_version="selected_models_synthetic_phase10_v1",
        manifest_sha256="8" * 64,
        feature_schema_version=FEATURE_VERSION,
        feature_schema_sha256=None,
        model=model,
        prepared_input=prepared_input,
        raw_output=raw_output,
        explanation_target=target_name or model.output_names[0],
        feature_metadata={"feature_schema_version": FEATURE_VERSION},
        synthetic=True,
    )


def changed_prepared(**changes):
    return replace(prepared(), **changes)
