#!/usr/bin/env python3
"""Fail-closed audit of Stage-2 selection and fixed LSTM sensitivity evidence."""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.synthetic.provenance import sha256_file
from experiments.lineage import read_artifact_index, read_run_registry, validate_artifact_lineage
from experiments.stage2_finalization import TASKS, choose_family, validate_exact_run_set
from vedant_infra.registry import validate_registry


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    selection_path = ROOT / "artifacts/selection/validation_family_selection_v1.json"
    sensitivity_path = ROOT / "artifacts/models/lstm_sensitivity_v1.json"
    selection, sensitivity = load(selection_path), load(sensitivity_path)
    if (selection.get("status") != "VALIDATION_FAMILY_SELECTION_FROZEN"
            or selection.get("frozen_before_lstm_interpretation") is not True
            or selection.get("test_accessed") is not False):
        raise RuntimeError("family-selection freeze is invalid")
    if sensitivity.get("family_selection_sha256") != sha256_file(selection_path):
        raise RuntimeError("LSTM manifest is not bound to the frozen selection")
    if sensitivity.get("statuses") != ["SENSITIVITY_ONLY", "NO_LSTM_SEARCH", "NOT_SERVING_SELECTION", "TEST_NOT_ACCESSED"]:
        raise RuntimeError("LSTM sensitivity statuses are invalid")
    validate_exact_run_set(sensitivity["runs"])
    for task in TASKS:
        item = selection["tasks"][task]; sources = item["sources"]
        chosen = choose_family(task, sources["xgboost"]["metrics"], sources["gru"]["metrics"])
        if chosen != item["selected_family"]:
            raise RuntimeError("family selection is not reproducible: " + task)
        run = next(row for row in sensitivity["runs"] if row["task"] == task)
        if (run["parent_best_gru_candidate_id"] != sources["gru"]["candidate_id"]
                or sha256_file(ROOT / run["checkpoint_path"]) != run["checkpoint_sha256"]
                or sha256_file(ROOT / run["prediction_path"]) != run["prediction_sha256"]
                or sha256_file(ROOT / run["metrics_path"]) != run["metrics_sha256"]):
            raise RuntimeError("LSTM parent/artifact lineage mismatch: " + task)
        if task == "organ_support" and (run["probability_type"] != "raw_uncalibrated"
                                         or run["support_calibrated"] or run["support_threshold"] is not None):
            raise RuntimeError("support LSTM crossed calibration/threshold boundary")
    runs = read_run_registry(ROOT / "experiments/registry.csv")
    stage2 = [row for row in runs if row.get("run_type") == "scientific_sensitivity"]
    if len(stage2) != 3 or {row["model_family"] for row in stage2} != {"lstm"}:
        raise RuntimeError("registry does not contain exactly three scientific LSTM sensitivity runs")
    if any(row.get("search_version") != "vedant_finalization_stage2_v1" for row in stage2):
        raise RuntimeError("unexpected LSTM search/version declaration")
    validate_registry(ROOT / "experiments/registry.csv")
    validate_artifact_lineage(read_artifact_index(ROOT / "experiments/artifacts.csv"), runs, repository_root=ROOT)
    forbidden = (ROOT / "artifacts/models/selected_models_v1.json", ROOT / "artifacts/governance/g3_freeze.json",
                 ROOT / "artifacts/acceptance/g3_model_selection_freeze_v1.json")
    if any(path.exists() for path in forbidden):
        raise RuntimeError("forbidden final selection/G3 artifact exists")
    if sensitivity.get("support_calibrated") or sensitivity.get("support_threshold") is not None:
        raise RuntimeError("Stage 2 must leave support uncalibrated and threshold unset")
    output = {"status": "PASS", "selection": {task: selection["tasks"][task]["selected_family"] for task in TASKS},
              "lstm_run_count": len(stage2), "lstm_search": False, "support_calibrated": False,
              "support_threshold": None, "selected_models_v1_created": False, "g3_created": False,
              "test_accessed": False}
    print(json.dumps(output, sort_keys=True))


if __name__ == "__main__":
    main()
