import importlib.util

import pytest
import torch

from explainability.ig import IntegratedGradientsAdapter, SyntheticZeroBaselineProvider
from ig_helpers import FEATURE_VERSION, SyntheticTwoOutputGRU, config, context, target


CAPTUM_AVAILABLE = importlib.util.find_spec("captum") is not None


@pytest.mark.skipif(not CAPTUM_AVAILABLE, reason="BLOCKED — CAPTUM DEPENDENCY REQUIRED")
def test_captum_linear_sanity_completeness_reproducibility_and_immutability():
    model = SyntheticTwoOutputGRU()
    model.train()
    adapter = IntegratedGradientsAdapter(
        baseline_provider=SyntheticZeroBaselineProvider(FEATURE_VERSION),
        config=config(), targets={"recovery": target()}, synthetic=True
    )
    explanation_context = context(model)
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    first = adapter.explain(explanation_context)
    second = adapter.explain(explanation_context)
    assert first == second
    assert model.training is True
    assert all(torch.equal(before[name], value) for name, value in model.state_dict().items())
    raw = torch.tensor(first.details["raw_signed_attribution"][0])
    assert raw.shape == (1, 8, 2)
    assert torch.count_nonzero(raw[:, :2]) == 0
    assert raw[0, 2:, 1].abs().sum() == 0
    assert raw.sum().item() == pytest.approx(21.0, abs=1e-5)
    delta = first.details["convergence_delta"][0]
    assert abs(delta) < 1e-5
    assert len(first.items) == 4


def test_phase9_ig_remains_isolated_after_treeshap_is_added():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    source = (root / "src/explainability/ig.py").read_text().lower()
    assert "captum.attr" in source
    assert "optimizer" not in source
    assert "calibrator" not in source
    assert "threshold" not in source
    assert "final_test" not in source
    assert "build_features" not in source
    assert (root / "src/explainability/tree_shap.py").is_file()
