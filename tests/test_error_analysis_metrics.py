from evaluation.error_analysis import evaluate_error_analysis
from error_analysis_helpers import binding, definition, record, spec


def test_recovery_horizons_are_reported_separately_and_raw():
    bound = binding()
    rows = (
        record(0, horizon="24h", target=-2, prediction=-1, selected_binding=bound),
        record(1, horizon="48h", target=4, prediction=2, selected_binding=bound),
    )
    table = evaluate_error_analysis(rows, spec(), bound)
    values = {(r.horizon, r.metric_name): r.metric_value for r in table if r.slice_group == "low"}
    assert values[("24h", "mae")] == 1
    assert values[("48h", "mae")] == 2


def test_icu_time_uses_hours_domain_phase9_metrics():
    bound = binding("icu_stay_time")
    rows = (record(0, task="icu_stay_time", target=24, prediction=12, selected_binding=bound),)
    table = evaluate_error_analysis(rows, spec(definition(tasks=("icu_stay_time",))), bound)
    median = next(r for r in table if r.metric_name == "median_absolute_error" and r.slice_group == "low")
    assert median.metric_value == 12


def test_one_class_support_keeps_slice_and_marks_auroc_undefined():
    bound = binding("organ_support")
    rows = (record(0, task="organ_support", target=1, prediction=.8, selected_binding=bound),)
    table = evaluate_error_analysis(rows, spec(definition(tasks=("organ_support",))), bound)
    auroc = next(r for r in table if r.metric_name == "auroc" and r.slice_group == "low")
    assert auroc.metric_value is None and auroc.status == "METRIC_UNDEFINED"
    assert auroc.N_positive_examples == 1 and auroc.N_positive_stays == 1
