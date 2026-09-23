import pytest

from explainability.tree_shap import TreeShapTargetError
from tree_shap_helpers import SyntheticXGBContract, target


def test_recovery_24h_and_48h_outputs_are_selected_independently():
    model = SyntheticXGBContract("recovery", ("delta_sofa_24h", "delta_sofa_48h"))
    target("recovery", "delta_sofa_24h", 0).validate(model, synthetic=True)
    target("recovery", "delta_sofa_48h", 1).validate(model, synthetic=True)


def test_wrong_horizon_index_name_and_cross_task_target_fail():
    model = SyntheticXGBContract("recovery", ("delta_sofa_24h", "delta_sofa_48h"))
    with pytest.raises(TreeShapTargetError, match="name/index"):
        target("recovery", "delta_sofa_48h", 0).validate(model, synthetic=True)
    with pytest.raises(TreeShapTargetError, match="unavailable|incompatible"):
        target("recovery", "delta_sofa_48h", 2).validate(model, synthetic=True)
    with pytest.raises(TreeShapTargetError, match="task mismatch"):
        target("icu_stay_time", "delta_sofa_24h", 0).validate(model, synthetic=True)


def test_scalar_task_cannot_select_a_nonzero_output():
    model = SyntheticXGBContract("organ_support", ("support_raw_margin", "not_authorized"))
    with pytest.raises(TreeShapTargetError, match="must be zero"):
        target("organ_support", "not_authorized", 1).validate(model, synthetic=True)


@pytest.mark.parametrize("task,name", (("icu_stay_time", "remaining_icu_log1p_hours"), ("organ_support", "support_raw_margin")))
def test_scalar_synthetic_targets_are_explicit_and_cannot_authorize_real_explanation(task, name):
    model = SyntheticXGBContract(task, (name,))
    chosen = target(task, name, 0)
    chosen.validate(model, synthetic=True)
    with pytest.raises(TreeShapTargetError, match="explicitly frozen non-synthetic domain"):
        chosen.validate(model, synthetic=False)
