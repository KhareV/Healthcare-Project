from dataclasses import replace

import pytest
import torch

from explainability.ig import (
    BaselineIdentity,
    IGBaselineError,
    IGConfig,
    IntegratedGradientsAdapter,
    SyntheticZeroBaselineProvider,
    attributable_inputs,
)
from ig_helpers import FEATURE_VERSION, batch, config, target


def test_synthetic_baseline_has_version_hash_scope_and_exact_shapes():
    prepared = batch()
    provider = SyntheticZeroBaselineProvider(FEATURE_VERSION)
    identity = provider.identity
    assert identity.version == "SYNTHETIC_IG_BASELINE_V1"
    assert len(identity.config_sha256) == 64
    assert identity.scientific_scope == "synthetic_development_only"
    baselines = provider.baselines(prepared)
    assert all(base.shape == value.shape for base, value in zip(baselines, attributable_inputs(prepared)))
    assert all(torch.count_nonzero(base) == 0 for base in baselines)


def test_synthetic_baseline_cannot_authorize_real_adapter():
    with pytest.raises(IGBaselineError, match="scope and baseline scope must match"):
        IntegratedGradientsAdapter(
            baseline_provider=SyntheticZeroBaselineProvider(FEATURE_VERSION),
            config=config(), targets={"recovery": target()}, synthetic=False
        )


def test_invalid_unfrozen_integration_config_is_rejected():
    with pytest.raises(Exception, match="unrecognized IG configuration scope"):
        replace(config(), scientific_scope="real_unfrozen").validate()


def test_config_identity_is_deterministic_and_settings_are_explicit():
    first = config()
    second = config()
    assert first == second
    assert first.sha256 == second.sha256
    assert first.n_steps == 32
    assert first.method == "gausslegendre"
