import pytest
import torch

from explainability.ig import GRUForwardWrapper, IGTargetError, attributable_inputs
from ig_helpers import SyntheticScalarGRU, SyntheticTwoOutputGRU, batch, target


def test_recovery_24h_and_48h_targets_are_independent_and_explicit():
    model = SyntheticTwoOutputGRU()
    prepared = batch()
    target24 = target(name="delta_sofa_24h", index=0)
    target48 = target(name="delta_sofa_48h", index=1)
    target24.validate(model, True)
    target48.validate(model, True)
    inputs = attributable_inputs(prepared)
    output24 = GRUForwardWrapper(model, prepared, target24)(*inputs)
    output48 = GRUForwardWrapper(model, prepared, target48)(*inputs)
    assert output24.item() == pytest.approx(21.0)
    assert output48.item() == pytest.approx(27.0)


def test_wrong_recovery_name_index_and_cross_task_targets_fail():
    model = SyntheticTwoOutputGRU()
    with pytest.raises(IGTargetError, match="name/index"):
        target(name="delta_sofa_48h", index=0).validate(model, True)
    with pytest.raises(IGTargetError, match="scalar"):
        target(task="icu_stay_time", name="log_time", index=1).validate(
            SyntheticScalarGRU("icu_stay_time"), True
        )


def test_synthetic_scalar_icu_and_support_target_descriptors():
    for task_name, output_name in (
        ("icu_stay_time", "synthetic_model_native_log_time"),
        ("organ_support", "synthetic_model_native_logit"),
    ):
        descriptor = target(task=task_name, name=output_name, index=0)
        descriptor.validate(SyntheticScalarGRU(task_name), True)


def test_synthetic_target_cannot_authorize_real_explanation():
    with pytest.raises(IGTargetError, match="real target must use the frozen Stage-4 scope"):
        target().validate(SyntheticTwoOutputGRU(), False)
