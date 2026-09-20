from dataclasses import replace

from evaluation.final_test import evaluate_prespecified_test_error_analysis
from error_analysis_helpers import binding, record, spec


def test_test_error_analysis_reuses_prespecified_phase15_slices():
    frozen_binding = binding()
    test_record = replace(record(0), split="test")
    rows = evaluate_prespecified_test_error_analysis(
        (test_record,), spec(), frozen_binding
    )
    assert rows
    assert {row.split for row in rows} == {"test"}
