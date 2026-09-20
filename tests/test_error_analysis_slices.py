from dataclasses import replace

from evaluation.slices import SliceDefinition, SliceGroup, assign_slice_group
from evaluation.sensitivity import CompleteComponentRecord
from error_analysis_helpers import definition


def test_assignment_is_deterministic_and_baseline_sofa_ignores_future_values():
    item = definition()
    first = {"baseline_sofa_at_t": 2, "sofa_t_plus_24": 20, "prediction": 100}
    second = dict(first, sofa_t_plus_24=0, prediction=-100)
    assert assign_slice_group(item, first) == "low"
    assert assign_slice_group(item, second) == "low"


def test_cutoff_and_baseline_support_use_only_frozen_at_t_metadata():
    cutoff = replace(definition("timing", "canonical_cutoff_index"), groups=(SliceGroup("early", upper=3), SliceGroup("late", lower=3), SliceGroup("unavailable")))
    assert assign_slice_group(cutoff, {"canonical_cutoff_index": 2, "target": 99}) == "early"
    support = SliceDefinition(
        "baseline_support", "support_state_at_t", ("recovery",),
        (SliceGroup("off", categories=("OFF",)), SliceGroup("on", categories=("ON",)), SliceGroup("unavailable")),
        "group", "unavailable", "synthetic-v1", "synthetic", assignment_kind="categorical"
    )
    assert assign_slice_group(support, {"support_state_at_t": "OFF", "support_state_after_t": "ON"}) == "off"


def test_padding_is_not_used_as_observation_density():
    density = replace(definition("density", "canonical_observation_density"), groups=(SliceGroup("sparse", upper=.5), SliceGroup("dense", lower=.5), SliceGroup("unavailable")))
    a = {"canonical_observation_density": .75, "structural_padding_bins": 6}
    b = {"canonical_observation_density": .25, "structural_padding_bins": 0}
    assert assign_slice_group(density, a) == "dense"
    assert assign_slice_group(density, b) == "sparse"


def test_medical_surgical_unknown_uses_declared_unavailable_group():
    item = SliceDefinition(
        "medical_surgical", "frozen_admission_group", ("recovery",),
        (SliceGroup("medical", categories=("MED",)), SliceGroup("surgical", categories=("SURG",)), SliceGroup("unavailable")),
        "group", "unavailable", "reviewed-map-v1", "synthetic", assignment_kind="categorical"
    )
    assert assign_slice_group(item, {"frozen_admission_group": "UNKNOWN"}) == "unavailable"


def test_complete_component_assignment_calls_phase14_membership_logic():
    item = SliceDefinition(
        "complete_component_sofa", "phase14_complete_component_included", ("recovery",),
        (SliceGroup("complete"), SliceGroup("incomplete")), "group", "incomplete",
        "phase14-v1", "evaluation.sensitivity.complete_component_included",
        assignment_kind="complete_component",
    )
    complete = CompleteComponentRecord(
        "A", "2020-01-01", "validation", 1.0, 2.0, True,
        (True,) * 6, (True,) * 6,
    )
    assert assign_slice_group(item, {}, complete_component_record=complete) == "complete"
    assert assign_slice_group(item, {}, complete_component_record=replace(complete, future_component_observed=(False,) + (True,) * 5)) == "incomplete"
