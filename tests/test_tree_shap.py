from dataclasses import replace

import numpy as np
import pytest

import explainability.tree_shap as tree_module
from explainability.tree_shap import (
    FlattenedFeatureContractError,
    ShapDependencyError,
    TreeShapAdapter,
    TreeShapInputError,
    TreeShapTargetError,
    normalize_shap_output,
)
from tree_shap_helpers import (
    FEATURE_VERSION,
    FLATTENING_HASH,
    SyntheticXGBContract,
    config,
    context,
    prepared,
    target,
)


class FakeExplanation:
    def __init__(self, values, base_values):
        self.values = values
        self.base_values = base_values


def fake_explainer_factory(calls, signed=None, base=1.0):
    signed = np.asarray(signed if signed is not None else [1.0, -2.0, 0.5, 0.0, 0.25, -0.25, 1.5])

    class FakeTreeExplainer:
        def __init__(self, model, **kwargs):
            calls.append(("construct", model, kwargs))

        def __call__(self, values):
            calls.append(("explain", values.copy()))
            return FakeExplanation(signed.reshape(1, -1), np.asarray([base]))

    return FakeTreeExplainer


def adapter(task_name="icu_stay_time", chosen=None):
    chosen = chosen or target(task_name)
    return TreeShapAdapter(config=config(), targets={task_name: chosen}, synthetic=True)


def test_treeexplainer_path_exact_input_additivity_signed_output_and_immutability(monkeypatch):
    calls = []
    monkeypatch.setattr(tree_module, "_tree_explainer_class", lambda: (fake_explainer_factory(calls), "SYNTHETIC_SHAP_TEST_DOUBLE"))
    model = SyntheticXGBContract()
    before = model.explanation_state_bytes()
    result = adapter().explain(context(model, raw_output=2.0))
    assert calls[0][0] == "construct" and calls[0][1] is model
    assert np.array_equal(calls[1][1], prepared().values)
    assert model.explanation_state_bytes() == before
    assert len(result.items) == model.input_dim
    assert result.items[1]["attribution"] == -2.0
    assert result.details["base_value"] + sum(result.details["raw_signed_shap_values"]) == pytest.approx(2.0)
    assert result.details["scientific_scope"] == "synthetic_non_scientific"
    assert len(result.details["feature_identities"]) == model.input_dim


@pytest.mark.parametrize("family", ("gru", "lstm", "naive", "unknown"))
def test_wrong_family_fails_before_shap(monkeypatch, family):
    monkeypatch.setattr(tree_module, "_tree_explainer_class", lambda: pytest.fail("SHAP invoked"))
    model = SyntheticXGBContract()
    with pytest.raises(TreeShapInputError, match="XGBoost"):
        adapter().explain(replace(context(model), family=family))


def test_wrong_hash_cutoff_task_and_target_fail_before_shap(monkeypatch):
    monkeypatch.setattr(tree_module, "_tree_explainer_class", lambda: pytest.fail("SHAP invoked"))
    model = SyntheticXGBContract()
    with pytest.raises(TreeShapInputError, match="hash"):
        adapter().explain(replace(context(model), model_sha256="1" * 64))
    with pytest.raises(TreeShapInputError, match="cutoff"):
        adapter().explain(replace(context(model), input_prediction_time="2026-01-04T18:00:00+00:00"))
    with pytest.raises(TreeShapInputError, match="task/model"):
        adapter().explain(replace(context(model), task="recovery"))
    with pytest.raises(TreeShapTargetError, match="target"):
        adapter().explain(replace(context(model), explanation_target="threshold_state"))


def test_schema_feature_order_flattening_and_dimension_attacks_fail_before_shap(monkeypatch):
    monkeypatch.setattr(tree_module, "_tree_explainer_class", lambda: pytest.fail("SHAP invoked"))
    model = SyntheticXGBContract()
    with pytest.raises(FlattenedFeatureContractError, match="schema"):
        adapter().explain(context(model, prepared_input=replace(prepared(), feature_schema_version="OTHER")))
    swapped = list(prepared().feature_identities)
    swapped[0], swapped[1] = swapped[1], swapped[0]
    with pytest.raises(FlattenedFeatureContractError, match="order"):
        adapter().explain(context(model, prepared_input=replace(prepared(), feature_identities=tuple(swapped))))
    with pytest.raises(FlattenedFeatureContractError, match="version"):
        adapter().explain(context(model, prepared_input=replace(prepared(), flattening_version="OTHER")))
    with pytest.raises(FlattenedFeatureContractError, match="hash"):
        adapter().explain(context(model, prepared_input=replace(prepared(), flattening_sha256="4" * 64)))
    attacked = SyntheticXGBContract()
    attacked.input_dim += 1
    with pytest.raises(FlattenedFeatureContractError, match="dimension"):
        adapter().explain(context(attacked))


def test_shap_dimension_nonfinite_and_additivity_fail_closed(monkeypatch):
    model = SyntheticXGBContract()
    calls = []
    monkeypatch.setattr(tree_module, "_tree_explainer_class", lambda: (fake_explainer_factory(calls, [1.0, 2.0]), "fake"))
    with pytest.raises(FlattenedFeatureContractError, match="dimension"):
        adapter().explain(context(model))
    monkeypatch.setattr(tree_module, "_tree_explainer_class", lambda: (fake_explainer_factory([], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, np.nan]), "fake"))
    with pytest.raises(TreeShapInputError, match="non-finite"):
        adapter().explain(context(model))
    monkeypatch.setattr(tree_module, "_tree_explainer_class", lambda: (fake_explainer_factory([]), "fake"))
    with pytest.raises(TreeShapInputError, match="additivity"):
        adapter().explain(context(model, raw_output=99.0))


def test_explainer_cannot_mutate_model_contract(monkeypatch):
    model = SyntheticXGBContract()

    class MutatingTreeExplainer:
        def __init__(self, supplied_model, **kwargs):
            self.model = supplied_model

        def __call__(self, values):
            self.model.flattened_feature_names = tuple(reversed(self.model.flattened_feature_names))
            signed = np.asarray([[1.0, -2.0, 0.5, 0.0, 0.25, -0.25, 1.5]])
            return FakeExplanation(signed, np.asarray([1.0]))

    monkeypatch.setattr(tree_module, "_tree_explainer_class", lambda: (MutatingTreeExplainer, "fake"))
    with pytest.raises(TreeShapInputError, match="changed"):
        adapter().explain(context(model, raw_output=2.0))


def test_multioutput_shape_is_normalized_by_explicit_output_index():
    values = np.zeros((1, 7, 2))
    values[0, :, 1] = np.arange(7)
    selected, base = normalize_shap_output(values, np.asarray([[10.0, 20.0]]), output_index=1, feature_dim=7)
    assert selected.tolist() == list(range(7))
    assert base == 20.0


@pytest.mark.parametrize(
    "name,index,signed,base,raw",
    (
        ("delta_sofa_24h", 0, [1.0, -2.0, 0.5, 0.0, 0.25, -0.25, 1.5], 1.0, [2.0, 9.0]),
        ("delta_sofa_48h", 1, [-1.0, 2.0, -0.5, 0.0, -0.25, 0.25, -1.5], 5.0, [9.0, 4.0]),
    ),
)
def test_recovery_horizons_use_explicit_output_without_chaining(monkeypatch, name, index, signed, base, raw):
    model = SyntheticXGBContract("recovery", ("delta_sofa_24h", "delta_sofa_48h"))
    values = np.zeros((1, model.input_dim, 2), dtype=float)
    values[0, :, index] = signed

    class MultiOutputTreeExplainer:
        def __init__(self, supplied_model, **kwargs):
            assert supplied_model is model

        def __call__(self, supplied_values):
            return FakeExplanation(values, np.asarray([[1.0, 5.0]]))

    monkeypatch.setattr(tree_module, "_tree_explainer_class", lambda: (MultiOutputTreeExplainer, "fake"))
    chosen = target("recovery", name, index)
    result = adapter("recovery", chosen).explain(
        context(
            model,
            target_name=name,
            prepared_input=replace(prepared(), task="recovery"),
            raw_output=raw,
        )
    )
    assert result.details["target"]["output_name"] == name
    assert result.details["raw_signed_shap_values"] == signed
    assert result.details["base_value"] == base


def test_missing_shap_dependency_is_explicit_and_has_no_fallback():
    try:
        TreeExplainer, _ = tree_module._tree_explainer_class()
    except ShapDependencyError as error:
        assert "SHAP DEPENDENCY REQUIRED" in str(error)
    else:
        assert TreeExplainer.__name__ == "TreeExplainer"


def test_real_scope_cannot_use_synthetic_configuration():
    with pytest.raises(Exception, match="REAL XGBOOST CONTRACT"):
        TreeShapAdapter(config=config(), targets={"icu_stay_time": target()}, synthetic=False)
