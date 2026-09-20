from dataclasses import replace

import pytest

from evaluation.sensitivity import (
    CompleteComponentRecord,
    SensitivityValidationError,
    complete_component_included,
    evaluate_complete_component_sensitivity,
)


ALL = (True,) * 6
ONE_MISSING = (True, True, True, True, True, False)


def record(row, stay, *, eligible=True, baseline=ALL, future=ALL, target=1.0, prediction=0.5):
    return CompleteComponentRecord(
        stay_id=stay,
        prediction_time="2020-01-{:02d}T00:00:00".format(row),
        partition="validation",
        target=target,
        prediction=prediction,
        eligible=eligible,
        baseline_component_observed=baseline,
        future_component_observed=future,
    )


def test_requires_all_six_at_baseline_and_relevant_future():
    assert complete_component_included(record(1, "A"))
    assert not complete_component_included(record(2, "A", baseline=ONE_MISSING))
    assert not complete_component_included(record(3, "A", future=ONE_MISSING))
    assert not complete_component_included(record(4, "A", eligible=False))


def test_horizons_are_filtered_independently():
    row24 = record(1, "A", future=ONE_MISSING)
    row48 = record(1, "A", future=ALL)
    result = evaluate_complete_component_sensitivity((row24,), (row48,))
    assert result.recovery24.status == "EMPTY_COMPLETE_COMPONENT_SUBSET"
    assert result.recovery48.status == "COMPLETE"
    assert result.recovery48.complete_N_examples == 1


def test_subset_counts_and_weights_are_recomputed_after_filtering():
    rows = (
        record(1, "A", target=2.0, prediction=1.0),
        record(2, "A", target=4.0, prediction=2.0),
        record(3, "B", target=2.0, prediction=1.0),
        record(4, "B", future=ONE_MISSING, target=100.0, prediction=1.0),
    )
    result = evaluate_complete_component_sensitivity(rows, rows)
    complete = result.recovery24.complete_component
    assert complete.counts == {"N_examples": 3, "N_ICU_stays": 2}
    assert complete.metrics["mae"] == pytest.approx(1.25)
    assert result.recovery24.full.counts["N_examples"] == 4


def test_missing_flags_duplicates_and_test_partition_fail_explicitly():
    with pytest.raises(SensitivityValidationError, match="six"):
        evaluate_complete_component_sensitivity(
            (record(1, "A", baseline=(True,) * 5),), (record(2, "A"),)
        )
    duplicate = record(1, "A")
    with pytest.raises(SensitivityValidationError, match="duplicate"):
        evaluate_complete_component_sensitivity((duplicate, duplicate), (record(2, "A"),))
    with pytest.raises(SensitivityValidationError, match="TEST ACCESS"):
        evaluate_complete_component_sensitivity(
            (replace(record(1, "A"), partition="test"),), (record(2, "A"),)
        )

