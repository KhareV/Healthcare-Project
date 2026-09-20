from evaluation.error_analysis import evaluate_error_analysis
from error_analysis_helpers import binding, record, spec


def test_all_groups_and_denominators_are_emitted_including_empty_and_missing():
    bound = binding()
    rows = (
        record(0, "A", selected_binding=bound),
        record(1, "A", selected_binding=bound),
        record(2, "B", metadata={}, selected_binding=bound),
    )
    table = evaluate_error_analysis(rows, spec(), bound)
    groups = {row.slice_group for row in table}
    assert groups == {"low", "high", "unavailable"}
    low = next(row for row in table if row.slice_group == "low")
    assert low.N_examples == 2 and low.N_ICU_stays == 1
    high = next(row for row in table if row.slice_group == "high")
    assert high.status == "EMPTY_ELIGIBLE_GROUP"
