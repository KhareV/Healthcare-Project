from dataclasses import replace

from evaluation.error_analysis import ErrorAnalysisRecord, SelectedModelBinding
from evaluation.slices import SliceDefinition, SliceGroup, SliceSpecification


def binding(task="recovery", **updates):
    values = {
        "task": task,
        "model_family": "gru",
        "model_run_id": "selected-run",
        "model_hash": "model-hash",
        "feature_version": "features-v1",
        "label_version": "labels-v1",
        "split_hash": "split-hash",
        "selected": True,
        "probability_type": "calibrated" if task == "organ_support" else None,
        "threshold": 0.4 if task == "organ_support" else None,
        "threshold_id": "global-threshold-v1" if task == "organ_support" else None,
        "calibrator_id": "global-calibrator-v1" if task == "organ_support" else None,
        "calibrator_hash": "calibrator-hash" if task == "organ_support" else None,
    }
    values.update(updates)
    return SelectedModelBinding(**values)


def definition(name="severity", source="baseline_sofa_at_t", tasks=("recovery",)):
    return SliceDefinition(
        slice_name=name,
        source_variable=source,
        allowed_tasks=tasks,
        groups=(
            SliceGroup("low", upper=5.0),
            SliceGroup("high", lower=5.0),
            SliceGroup("unavailable"),
        ),
        missing_value_policy="group",
        missing_group="unavailable",
        mapping_version="synthetic-v1",
        provenance="synthetic test fixture",
    )


def spec(*definitions):
    return SliceSpecification(
        spec_version="synthetic-slices-v1",
        slices=tuple(definitions or (definition(),)),
        mode="synthetic",
    )


def record(
    index,
    stay="A",
    task="recovery",
    horizon="24h",
    target=1.0,
    prediction=2.0,
    eligible=True,
    metadata=None,
    selected_binding=None,
    **updates,
):
    bound = selected_binding or binding(task)
    value = ErrorAnalysisRecord(
        row_id=f"row-{index}",
        stay_id=stay,
        prediction_time=f"2020-01-{index + 1:02d}T00:00:00",
        split="validation",
        task=task,
        horizon=horizon if task == "recovery" else None,
        target=target,
        prediction=prediction,
        eligible=eligible,
        model_run_id=bound.model_run_id,
        model_hash=bound.model_hash,
        feature_version=bound.feature_version,
        label_version=bound.label_version,
        split_hash=bound.split_hash,
        slice_metadata=(
            {"baseline_sofa_at_t": 2.0} if metadata is None else metadata
        ),
        probability_type=bound.probability_type,
        calibrator_id=bound.calibrator_id,
        calibrator_hash=bound.calibrator_hash,
    )
    return replace(value, **updates)
