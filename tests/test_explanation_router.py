from dataclasses import replace

import pytest

from explainability.router import (
    AdapterExplanation,
    ExplanationContext,
    ExplanationCutoffMismatchError,
    ExplanationRouter,
    ExplanationSchemaError,
    ExplanationTaskMismatchError,
    InvalidExplanationRouteError,
    MissingExplanationAdapterError,
    ModelIdentityMismatchError,
    StaleExplanationManifestError,
    UnsupportedExplanationFamilyError,
    lint_descriptive_explanation_text,
)
from serving.artifacts import load_serving_bundle
from serving.history import TruncatedStayHistory
from serving_helpers import FixtureRuntime, build_kwargs, copy_serving_fixture


CUTOFF = "2026-01-02T12:00:00+00:00"


class SpyAdapter:
    synthetic = True

    def __init__(self, method, family):
        self.method = method
        self.supported_families = (family,)
        self.calls = []
        self.mutate_output = None

    def explain(self, context):
        self.calls.append(context)
        output = AdapterExplanation(
            task=context.task,
            family=context.family,
            method=self.method,
            model_sha256=context.model_sha256,
            prediction_time=context.prediction_time,
            manifest_version=context.manifest_version,
            manifest_sha256=context.manifest_sha256,
            feature_schema_version=context.feature_schema_version,
            synthetic=True,
            items=({"feature_name": "SYNTHETIC_SIGNAL", "attribution": 0.25},),
        )
        return self.mutate_output(output) if self.mutate_output else output


def adapters():
    return {
        "integrated_gradients": SpyAdapter("integrated_gradients", "gru"),
        "tree_shap": SpyAdapter("tree_shap", "xgboost"),
    }


def bundle(tmp_path):
    root = copy_serving_fixture(tmp_path)
    runtime = FixtureRuntime()
    return load_serving_bundle(**build_kwargs(root, runtime)), runtime


def context_for(bundle, task, **changes):
    selected = bundle.task(task)
    values = dict(
        task=task,
        input_task=task,
        family=selected.identity.family,
        input_family=selected.identity.family,
        model_sha256=selected.identity.artifact_sha256,
        prediction_time=CUTOFF,
        input_prediction_time=CUTOFF,
        manifest_version=bundle.manifest_version,
        manifest_sha256=bundle.manifest_file_sha256,
        feature_schema_version=selected.identity.feature_version,
        feature_schema_sha256=None,
        model=selected.predictor,
        prepared_input={"cutoff_safe": True},
        raw_output=0.5,
        explanation_target=None,
        feature_metadata={"feature_schema_version": selected.identity.feature_version},
        synthetic=True,
        support_calibrator_sha256=(bundle.support_calibrator_sha256 if task == "organ_support" else None),
        support_threshold_sha256=(bundle.support_threshold_sha256 if task == "organ_support" else None),
    )
    values.update(changes)
    return ExplanationContext(**values)


def test_xgboost_routes_only_to_tree_shap_and_gru_only_to_ig(tmp_path):
    selected, _ = bundle(tmp_path)
    registered = adapters()
    router = ExplanationRouter(selected, registered)
    icu = router.explain(context_for(selected, "icu_stay_time"))
    recovery = router.explain(context_for(selected, "recovery"))
    assert icu.method == "tree_shap"
    assert recovery.method == "integrated_gradients"
    assert len(registered["tree_shap"].calls) == 1
    assert len(registered["integrated_gradients"].calls) == 1


@pytest.mark.parametrize(
    "family,method",
    (("xgboost", "integrated_gradients"), ("gru", "tree_shap")),
)
def test_wrong_family_method_manifest_fails_before_adapter(family, method, tmp_path):
    selected, _ = bundle(tmp_path)
    task_bundle = selected.task("recovery")
    attacked_identity = replace(task_bundle.identity, family=family, explanation_method=method)
    attacked_task = replace(task_bundle, identity=attacked_identity)
    attacked = replace(
        selected,
        tasks=tuple(attacked_task if item.identity.task == "recovery" else item for item in selected.tasks),
    )
    registered = adapters()
    with pytest.raises(InvalidExplanationRouteError, match="route mismatch"):
        ExplanationRouter(attacked, registered)
    assert all(not adapter.calls for adapter in registered.values())


@pytest.mark.parametrize("family", ("naive", "lstm"))
def test_nonserving_family_is_rejected(family, tmp_path):
    selected, _ = bundle(tmp_path)
    item = selected.task("recovery")
    attacked = replace(
        selected,
        tasks=tuple(
            replace(item, identity=replace(item.identity, family=family))
            if candidate.identity.task == "recovery"
            else candidate
            for candidate in selected.tasks
        ),
    )
    with pytest.raises(UnsupportedExplanationFamilyError):
        ExplanationRouter(attacked, adapters())


def test_model_hash_mismatch_fails_before_adapter(tmp_path):
    selected, _ = bundle(tmp_path)
    registered = adapters()
    router = ExplanationRouter(selected, registered)
    with pytest.raises(ModelIdentityMismatchError, match="artifact/hash"):
        router.explain(context_for(selected, "recovery", model_sha256="b" * 64))
    assert all(not adapter.calls for adapter in registered.values())


def test_wrong_model_object_family_and_task_fail_before_adapter(tmp_path):
    selected, _ = bundle(tmp_path)
    registered = adapters()
    router = ExplanationRouter(selected, registered)
    wrong_model = selected.task("icu_stay_time").predictor
    with pytest.raises(ExplanationTaskMismatchError):
        router.explain(context_for(selected, "recovery", model=wrong_model))
    assert all(not adapter.calls for adapter in registered.values())


def test_cutoff_mismatch_fails_before_adapter(tmp_path):
    selected, _ = bundle(tmp_path)
    registered = adapters()
    router = ExplanationRouter(selected, registered)
    with pytest.raises(ExplanationCutoffMismatchError):
        router.explain(
            context_for(
                selected,
                "recovery",
                input_prediction_time="2026-01-02T18:00:00+00:00",
            )
        )
    assert all(not adapter.calls for adapter in registered.values())


@pytest.mark.parametrize("field", ("manifest_version", "manifest_sha256"))
def test_stale_manifest_fails_before_adapter(field, tmp_path):
    selected, _ = bundle(tmp_path)
    registered = adapters()
    router = ExplanationRouter(selected, registered)
    value = "STALE_MANIFEST" if field == "manifest_version" else "c" * 64
    with pytest.raises(StaleExplanationManifestError):
        router.explain(context_for(selected, "recovery", **{field: value}))
    assert all(not adapter.calls for adapter in registered.values())


def test_unknown_task_fails_before_adapter(tmp_path):
    selected, _ = bundle(tmp_path)
    registered = adapters()
    router = ExplanationRouter(selected, registered)
    with pytest.raises(ExplanationTaskMismatchError):
        router.explain(replace(context_for(selected, "recovery"), task="misspelled"))
    assert all(not adapter.calls for adapter in registered.values())


def test_prepared_input_task_and_family_mismatch_fail_before_adapter(tmp_path):
    selected, _ = bundle(tmp_path)
    registered = adapters()
    router = ExplanationRouter(selected, registered)
    with pytest.raises(ExplanationTaskMismatchError, match="prepared-input task"):
        router.explain(context_for(selected, "recovery", input_task="icu_stay_time"))
    with pytest.raises(ModelIdentityMismatchError, match="family"):
        router.explain(context_for(selected, "recovery", input_family="xgboost"))
    assert all(not adapter.calls for adapter in registered.values())


def test_adapter_declared_wrong_method_or_family_fails_at_startup(tmp_path):
    selected, _ = bundle(tmp_path)
    wrong_method = adapters()
    wrong_method["tree_shap"].method = "integrated_gradients"
    with pytest.raises(InvalidExplanationRouteError, match="declared method"):
        ExplanationRouter(selected, wrong_method)
    wrong_family = adapters()
    wrong_family["tree_shap"].supported_families = ("gru",)
    with pytest.raises(InvalidExplanationRouteError, match="supported family"):
        ExplanationRouter(selected, wrong_family)


def test_missing_required_adapter_has_no_empty_fallback(tmp_path):
    selected, _ = bundle(tmp_path)
    with pytest.raises(MissingExplanationAdapterError):
        ExplanationRouter(selected, {"integrated_gradients": adapters()["integrated_gradients"]})


def test_missing_manifest_method_and_synthetic_adapter_in_real_scope_fail_closed(tmp_path):
    selected, _ = bundle(tmp_path)
    item = selected.task("recovery")
    missing_method_bundle = replace(
        selected,
        tasks=tuple(
            replace(item, identity=replace(item.identity, explanation_method=None))
            if candidate.identity.task == "recovery"
            else candidate
            for candidate in selected.tasks
        ),
    )
    with pytest.raises(InvalidExplanationRouteError):
        ExplanationRouter(missing_method_bundle, adapters())
    with pytest.raises(InvalidExplanationRouteError, match="scope"):
        ExplanationRouter(replace(selected, scope="real"), adapters())


def test_feature_schema_scope_and_support_lineage_are_bound(tmp_path):
    selected, _ = bundle(tmp_path)
    router = ExplanationRouter(selected, adapters())
    with pytest.raises(ExplanationSchemaError, match="feature schema"):
        router.explain(context_for(selected, "recovery", feature_schema_version="DRIFT"))
    with pytest.raises(ExplanationSchemaError, match="hash is not bound"):
        router.explain(context_for(selected, "recovery", feature_schema_sha256="a" * 64))
    with pytest.raises(ExplanationSchemaError, match="metadata version"):
        router.explain(
            context_for(
                selected,
                "recovery",
                feature_metadata={"feature_schema_version": "DRIFT"},
            )
        )
    with pytest.raises(ExplanationSchemaError, match="scope"):
        router.explain(context_for(selected, "recovery", synthetic=False))
    with pytest.raises(ExplanationSchemaError, match="calibration lineage"):
        router.explain(context_for(selected, "organ_support", support_calibrator_sha256="d" * 64))


def test_adapter_result_metadata_is_validated(tmp_path):
    selected, _ = bundle(tmp_path)
    registered = adapters()
    registered["integrated_gradients"].mutate_output = lambda output: replace(
        output, model_sha256="e" * 64
    )
    with pytest.raises(ExplanationSchemaError, match="metadata"):
        ExplanationRouter(selected, registered).explain(context_for(selected, "recovery"))


def test_same_context_routes_deterministically(tmp_path):
    selected, _ = bundle(tmp_path)
    router = ExplanationRouter(selected, adapters())
    context = context_for(selected, "recovery")
    assert router.explain(context) == router.explain(context)


def test_noncausal_and_treatment_wording_lint():
    assert lint_descriptive_explanation_text("Feature X was influential for this prediction") == ()
    assert lint_descriptive_explanation_text("Feature X caused deterioration")
    assert lint_descriptive_explanation_text("Recommend medication now")


def test_router_remains_free_of_algorithm_code_after_phase11():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    source = (root / "src/explainability/router.py").read_text().lower()
    assert "captum" not in source
    assert "treeexplainer" not in source
    assert "integratedgradients" not in source
    assert "def build_features" not in source
    assert ".predict(" not in source
    assert (root / "src/explainability/ig.py").is_file()
    assert (root / "src/explainability/tree_shap.py").is_file()
    assert (root / "api/main.py").is_file()
    assert (root / "dashboard/app.py").is_file()
