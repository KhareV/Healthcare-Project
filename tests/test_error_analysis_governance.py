from dataclasses import replace

import pytest

from evaluation.error_analysis import ErrorAnalysisValidationError, evaluate_error_analysis
from evaluation.slices import SliceValidationError, validate_slice_specification
from error_analysis_helpers import binding, definition, record, spec


def test_error_based_slice_is_rejected():
    with pytest.raises(SliceValidationError, match="forbidden"):
        validate_slice_specification(spec(definition(source="absolute_error")), require_frozen=True)


def test_unselected_or_mismatched_model_is_rejected():
    bound = binding()
    with pytest.raises(ErrorAnalysisValidationError, match="selected model"):
        evaluate_error_analysis((record(0, selected_binding=bound),), spec(), replace(bound, selected=False))
    with pytest.raises(ErrorAnalysisValidationError, match="model_hash"):
        evaluate_error_analysis((record(0, selected_binding=bound, model_hash="wrong"),), spec(), bound)


def test_support_uses_one_global_calibrator_and_threshold_for_every_group():
    bound = binding("organ_support")
    rows = (
        record(0, task="organ_support", target=1, prediction=.8, selected_binding=bound),
        record(1, task="organ_support", target=0, prediction=.2, metadata={"baseline_sofa_at_t": 8}, selected_binding=bound),
    )
    table = evaluate_error_analysis(rows, spec(definition(tasks=("organ_support",))), bound)
    assert {row.threshold_id for row in table} == {"global-threshold-v1"}
    assert {row.probability_type for row in table} == {"calibrated"}
    assert {row.calibrator_id for row in table} == {"global-calibrator-v1"}
    with pytest.raises(ErrorAnalysisValidationError, match="calibrator lineage"):
        evaluate_error_analysis(
            (replace(rows[0], calibrator_id="subgroup-calibrator"),),
            spec(definition(tasks=("organ_support",))),
            bound,
        )


def test_unfrozen_slice_definition_blocks_real_scientific_execution():
    blocked = replace(definition(), status="BLOCKED — PRESPECIFIED SLICE BOUNDARIES REQUIRED")
    with pytest.raises(SliceValidationError, match="PRESPECIFIED"):
        evaluate_error_analysis((record(0),), spec(blocked), binding())


def test_remaining_duration_is_analysis_metadata_not_model_input():
    row = record(0, metadata={"remaining_current_icu_hours_at_t": 36})
    assert not hasattr(row, "model_inputs")
    assert row.slice_metadata["remaining_current_icu_hours_at_t"] == 36
