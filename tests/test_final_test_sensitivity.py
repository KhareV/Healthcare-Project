from evaluation.final_test import evaluate_test_complete_component
from evaluation.sensitivity import CompleteComponentRecord


def _row(stay, row, future=(True,) * 6):
    return CompleteComponentRecord(
        stay_id=stay,
        prediction_time="t" + str(row),
        partition="test",
        target=2.0,
        prediction=1.0,
        eligible=True,
        baseline_component_observed=(True,) * 6,
        future_component_observed=future,
    )


def test_test_sensitivity_reuses_phase14_membership_and_reweights_subset():
    rows = (_row("A", 1), _row("A", 2), _row("B", 3, (False,) + (True,) * 5))
    result = evaluate_test_complete_component(rows, rows)
    assert result.recovery24.complete_N_examples == 2
    assert result.recovery24.complete_N_ICU_stays == 1
