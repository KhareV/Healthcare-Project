"""Freeze the Stage-5 prespecified error-analysis slice spec from TRAIN only.

Writes concrete groups/boundaries into configs/error_analysis/error_analysis_v1.json,
deriving every numeric boundary from TRAIN-partition rows (never validation or
test). Categorical slices use the accepted schema's own fixed vocabulary.
Two slices (medical_surgical, baseline_support) are left NOT_REPORTED because
no authoritative frozen mapping exists in the accepted schema, matching the
project's own prior precedent for that field.
"""

import csv
import json
import math
from pathlib import Path

from evaluation.bootstrap import percentile


ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "artifacts/data/synthetic/phase10/final/synthetic_phase10_v1/train.jsonl"
PRE_SPLIT_PATH = ROOT / "artifacts/data/synthetic/pre_split/final/phase9_final_v1/pre_split_scientific_package.jsonl"
SPLIT_PATH = ROOT / "artifacts/splits/synthetic_split_v2.csv"
STATICS_PATH = ROOT / "artifacts/data/synthetic/timelines/final/phase9_final_v1/canonical_statics.jsonl"
CONFIG_PATH = ROOT / "configs/error_analysis/error_analysis_v1.json"


def _load_jsonl(path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _train_subjects():
    with SPLIT_PATH.open(newline="") as handle:
        return {row["subject_id"] for row in csv.DictReader(handle) if row["split"] == "train"}


def _tertile_bounds(values):
    return percentile(values, 1.0 / 3.0), percentile(values, 2.0 / 3.0)


def main() -> None:
    train_subjects = _train_subjects()
    train_rows = [row for row in _load_jsonl(TRAIN_PATH) if row["split"] == "train"]

    # baseline_sofa: joined from pre-split rows (train subjects only).
    pre_split = {
        (row["subject_id"], row["stay_id"], row["prediction_time"], row["grid_index"]): row["baseline_sofa"]
        for row in _load_jsonl(PRE_SPLIT_PATH)
        if row["subject_id"] in train_subjects
    }
    baseline_sofa_values = [
        float(pre_split[(row["subject_id"], row["stay_id"], row["prediction_time"], row["grid_index"])])
        for row in train_rows
        if (row["subject_id"], row["stay_id"], row["prediction_time"], row["grid_index"]) in pre_split
        and pre_split[(row["subject_id"], row["stay_id"], row["prediction_time"], row["grid_index"])] is not None
    ]
    sofa_low, sofa_high = _tertile_bounds(baseline_sofa_values)

    # remaining_icu_duration: expm1(icu_time_log1p) for icu-eligible train rows.
    remaining_hours = [
        math.expm1(float(row["icu_time_log1p"])) for row in train_rows if row["icu_time_eligible"]
    ]
    icu_low, icu_high = _tertile_bounds(remaining_hours)

    # observation_density: genuine-observation fraction over non-padded cells.
    densities = []
    for row in train_rows:
        mask = row["observation_mask"]
        padding = row["padding_mask"]
        total = 0
        observed = 0
        for bin_index, bin_values in enumerate(mask):
            if padding[bin_index]:
                continue
            total += len(bin_values)
            observed += sum(1 for value in bin_values if value)
        if total > 0:
            densities.append(observed / total)
    density_low, density_high = _tertile_bounds(densities)

    # cutoff_timing: TRAIN median grid_index (early/late split).
    grid_indices = sorted(float(row["grid_index"]) for row in train_rows)
    cutoff_median = percentile(grid_indices, 0.5)

    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    config["status"] = "FROZEN_FOR_FINAL_TEST"
    config["final_test_authorized"] = True
    config["final_test_status"] = "FINAL TEST ERROR ANALYSIS — AUTHORIZED FOR STAGE 5"
    config["minimum_sample_size"] = 10
    config["minimum_sample_size_status"] = "FROZEN_PRE_TEST_REPORTING_THRESHOLD_NOT_A_SUPPRESSION_GATE"

    by_name = {item["slice_name"]: item for item in config["slices"]}

    by_name["baseline_sofa"].update(
        groups=[
            {"name": "low", "upper": sofa_low},
            {"name": "mid", "lower": sofa_low, "upper": sofa_high},
            {"name": "high", "lower": sofa_high},
            {"name": "unavailable"},
        ],
        missing_value_policy="group",
        missing_group="unavailable",
        mapping_version="train_tertile_v1",
        status="FROZEN",
        train_derived_bounds={"low_upper": sofa_low, "mid_upper": sofa_high},
    )
    by_name["remaining_icu_duration"].update(
        groups=[
            {"name": "short", "upper": icu_low},
            {"name": "medium", "lower": icu_low, "upper": icu_high},
            {"name": "long", "lower": icu_high},
        ],
        missing_value_policy="exclude",
        mapping_version="train_tertile_v1",
        status="FROZEN",
        train_derived_bounds={"short_upper": icu_low, "medium_upper": icu_high},
    )
    by_name["observation_density"].update(
        groups=[
            {"name": "low", "upper": density_low},
            {"name": "mid", "lower": density_low, "upper": density_high},
            {"name": "high", "lower": density_high},
        ],
        missing_value_policy="exclude",
        mapping_version="train_tertile_v1",
        status="FROZEN",
        train_derived_bounds={"low_upper": density_low, "mid_upper": density_high},
    )
    by_name["cutoff_timing"].update(
        groups=[
            {"name": "earlier", "upper": cutoff_median},
            {"name": "later", "lower": cutoff_median},
        ],
        missing_value_policy="exclude",
        mapping_version="train_median_v1",
        status="FROZEN",
        train_derived_bounds={"earlier_upper": cutoff_median},
    )
    by_name["icu_type"].update(
        slice_name="cardiac_subtype",
        source_variable="frozen_cardiac_condition_group",
        groups=[
            {"name": "SYNTHETIC_HF", "categories": ["SYNTHETIC_HF"]},
            {"name": "SYNTHETIC_ISCHEMIC", "categories": ["SYNTHETIC_ISCHEMIC"]},
            {"name": "SYNTHETIC_ARRHYTHMIA", "categories": ["SYNTHETIC_ARRHYTHMIA"]},
            {"name": "unavailable"},
        ],
        missing_value_policy="group",
        missing_group="unavailable",
        mapping_version="accepted_cardiac_condition_group_v1",
        status="FROZEN",
        provenance="canonical_statics.cardiac_condition_group (accepted raw static)",
        assignment_kind="categorical",
    )
    config["task_slice_applicability"]["cardiac_subtype"] = config["task_slice_applicability"].pop("icu_type")
    by_name["medical_surgical"].update(status="NOT_REPORTED_NO_AUTHORITATIVE_MAPPING")
    by_name["baseline_support"].update(status="NOT_REPORTED_NO_AUTHORITATIVE_MAPPING")
    by_name["complete_component_sofa"].update(status="FROZEN")

    config["slices"] = list(by_name.values())
    CONFIG_PATH.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("wrote", CONFIG_PATH)
    print(
        "bounds:",
        json.dumps(
            {
                "baseline_sofa": [sofa_low, sofa_high],
                "remaining_icu_duration": [icu_low, icu_high],
                "observation_density": [density_low, density_high],
                "cutoff_timing": [cutoff_median],
            }
        ),
    )


if __name__ == "__main__":
    main()
