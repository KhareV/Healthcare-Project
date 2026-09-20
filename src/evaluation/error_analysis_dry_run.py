"""Generate a synthetic-only Phase-15 table without loading project data."""

import argparse
from pathlib import Path

from evaluation.error_analysis import (
    ErrorAnalysisRecord,
    SelectedModelBinding,
    evaluate_error_analysis,
    write_error_analysis_table,
)
from evaluation.slices import SliceDefinition, SliceGroup, SliceSpecification


def synthetic_spec() -> SliceSpecification:
    return SliceSpecification(
        spec_version="synthetic_error_analysis_slice_spec_v1",
        mode="synthetic",
        slices=(
            SliceDefinition(
                slice_name="baseline_sofa",
                source_variable="baseline_sofa_at_t",
                allowed_tasks=("recovery",),
                groups=(
                    SliceGroup("lower", upper=5.0),
                    SliceGroup("higher", lower=5.0),
                    SliceGroup("unavailable"),
                ),
                missing_value_policy="group",
                missing_group="unavailable",
                mapping_version="synthetic_only_not_scientific",
                provenance="synthetic fixture",
            ),
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/error_analysis/synthetic/error_analysis_table.csv"),
    )
    args = parser.parse_args()
    binding = SelectedModelBinding(
        task="recovery",
        model_family="gru",
        model_run_id="synthetic_selected_recovery",
        model_hash="synthetic-model-hash",
        feature_version="synthetic-features-v1",
        label_version="synthetic-labels-v1",
        split_hash="synthetic-validation-split-hash",
        selected=True,
    )
    records = tuple(
        ErrorAnalysisRecord(
            row_id=f"row-{index}",
            stay_id=stay,
            prediction_time=f"2020-01-0{index + 1}T00:00:00",
            split="validation",
            task="recovery",
            horizon=horizon,
            target=target,
            prediction=prediction,
            eligible=True,
            model_run_id=binding.model_run_id,
            model_hash=binding.model_hash,
            feature_version=binding.feature_version,
            label_version=binding.label_version,
            split_hash=binding.split_hash,
            slice_metadata={"baseline_sofa_at_t": sofa},
        )
        for index, (stay, horizon, target, prediction, sofa) in enumerate(
            (
                ("A", "24h", -1.0, -0.5, 3.0),
                ("B", "24h", 2.0, 1.0, 8.0),
                ("A", "48h", -2.0, -1.0, 3.0),
                ("B", "48h", 1.0, 2.0, 8.0),
            )
        )
    )
    write_error_analysis_table(
        evaluate_error_analysis(records, synthetic_spec(), binding), args.output
    )
    print(args.output)


if __name__ == "__main__":
    main()
