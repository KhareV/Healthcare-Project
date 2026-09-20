from dataclasses import replace

import pytest
import torch

import explainability.ig as ig_module
from explainability.ig import (
    CaptumDependencyError,
    GRUForwardWrapper,
    IGInputError,
    IntegratedGradientsAdapter,
    SyntheticZeroBaselineProvider,
    aggregate_absolute_attribution,
    attributable_inputs,
)
from ig_helpers import SyntheticTwoOutputGRU, batch, config, context, target


def test_forward_wrapper_exactly_matches_normal_model_output():
    model = SyntheticTwoOutputGRU()
    prepared = batch()
    wrapper = GRUForwardWrapper(model, prepared, target())
    assert torch.equal(wrapper.full_output(*attributable_inputs(prepared)), model(prepared))


def test_feature_timestep_shape_sign_and_absolute_aggregation():
    prepared = batch()
    signed = torch.tensor(
        [[[-2.0, 1.0]] * 8]
    )
    mask = torch.zeros_like(signed)
    rows = aggregate_absolute_attribution((signed, mask), prepared)
    assert signed.shape == (1, 8, 2)
    assert signed[0, 0, 0] < 0
    assert rows == (
        {"feature_name": "value:SYNTHETIC_A", "absolute_attribution": 16.0},
        {"feature_name": "value:SYNTHETIC_B", "absolute_attribution": 8.0},
        {"feature_name": "observation_mask:SYNTHETIC_A", "absolute_attribution": 0.0},
        {"feature_name": "observation_mask:SYNTHETIC_B", "absolute_attribution": 0.0},
    )


def test_static_attribution_is_preserved_separately_not_repeated_over_time():
    prepared = replace(batch(), static_features=torch.tensor([[2.0, -3.0]]))
    temporal = torch.zeros_like(prepared.sequence)
    observation = torch.zeros_like(prepared.sequence)
    static = torch.tensor([[0.5, -1.5]])
    rows = aggregate_absolute_attribution(
        (temporal, observation, static),
        prepared,
        static_feature_names=("SYNTHETIC_STATIC_A", "SYNTHETIC_STATIC_B"),
    )
    assert rows[-2:] == (
        {"feature_name": "static:SYNTHETIC_STATIC_A", "absolute_attribution": 0.5},
        {"feature_name": "static:SYNTHETIC_STATIC_B", "absolute_attribution": 1.5},
    )


def test_wrong_family_hash_and_cutoff_fail_before_captum(monkeypatch):
    calls = []
    monkeypatch.setattr(ig_module, "_captum_integrated_gradients_class", lambda: calls.append(1))
    adapter = IntegratedGradientsAdapter(
        baseline_provider=SyntheticZeroBaselineProvider("SYNTHETIC_FEATURES_PHASE6_NOT_REAL"),
        config=config(),
        targets={"recovery": target()},
        synthetic=True,
    )
    model = SyntheticTwoOutputGRU()
    with pytest.raises(IGInputError, match="GRU"):
        adapter.explain(replace(context(model), family="xgboost"))
    with pytest.raises(IGInputError, match="hash"):
        adapter.explain(replace(context(model), model_sha256="1" * 64))
    with pytest.raises(IGInputError, match="cutoff"):
        adapter.explain(replace(context(model), input_prediction_time="2026-01-04T18:00:00+00:00"))
    assert calls == []


def test_missing_captum_fails_without_fallback_when_dependency_absent():
    try:
        import captum  # noqa: F401
    except ImportError:
        with pytest.raises(CaptumDependencyError, match="CAPTUM DEPENDENCY REQUIRED"):
            ig_module._captum_integrated_gradients_class()
    else:
        assert ig_module._captum_integrated_gradients_class().__name__ == "IntegratedGradients"


def test_nonfinite_prepared_input_fails_before_captum(monkeypatch):
    monkeypatch.setattr(ig_module, "_captum_integrated_gradients_class", lambda: pytest.fail("Captum called"))
    prepared = batch()
    prepared.sequence[0, 2, 0] = float("nan")
    adapter = IntegratedGradientsAdapter(
        baseline_provider=SyntheticZeroBaselineProvider(prepared.versions["feature_schema_version"]),
        config=config(), targets={"recovery": target()}, synthetic=True
    )
    with pytest.raises(IGInputError, match="finite"):
        adapter.explain(context(SyntheticTwoOutputGRU(), prepared=prepared))
