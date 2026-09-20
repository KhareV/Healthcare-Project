import pytest

from evaluation.error_analysis import evaluate_error_analysis
from error_analysis_helpers import binding, record, spec


def test_stay_weights_are_recomputed_inside_each_group():
    bound = binding()
    rows = (
        record(0, "A", target=0, prediction=1, selected_binding=bound),
        record(1, "A", target=0, prediction=3, selected_binding=bound),
        record(2, "A", target=0, prediction=100, metadata={"baseline_sofa_at_t": 8}, selected_binding=bound),
        record(3, "A", target=0, prediction=100, metadata={"baseline_sofa_at_t": 8}, selected_binding=bound),
        record(4, "B", target=0, prediction=2, selected_binding=bound),
    )
    table = evaluate_error_analysis(rows, spec(), bound)
    mae = next(row for row in table if row.slice_group == "low" and row.metric_name == "mae")
    assert mae.metric_value == pytest.approx(2.0)
    assert mae.N_examples == 3 and mae.N_ICU_stays == 2


def test_ineligible_attack_does_not_change_metrics():
    bound = binding()
    base = (record(0, target=1, prediction=2, selected_binding=bound),)
    attacked = base + (record(1, "B", target=-1e9, prediction=1e9, eligible=False, selected_binding=bound),)
    get = lambda rows: next(r.metric_value for r in evaluate_error_analysis(rows, spec(), bound) if r.slice_group == "low" and r.metric_name == "mae")
    assert get(base) == get(attacked)
