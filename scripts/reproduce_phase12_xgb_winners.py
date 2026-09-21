#!/usr/bin/env python3
"""Deterministically refit the three Phase-12 winners as non-slot audit attempts."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch

from data.synthetic.config import canonical_json_bytes
from data.xgb_canonical import Phase10XGBData
from experiments.xgb_phase12 import validate_candidate_manifest
from models.icu_time_postprocess import remaining_icu_hours_from_log_prediction
from models.xgb_canonical import RUN_PHASE12_SCIENTIFIC, train_xgb_candidate
from run_phase12_xgb_search import (
    _icu_metrics, _lineage, _load, _recovery_metrics, _support_metrics,
    _task_matrices, _xgb_config,
)


SEARCH_ROOT = ROOT / "artifacts/search/xgb/phase12"


def _saved_predictions(path, field):
    with Path(path).open(encoding="utf-8") as handle:
        return np.asarray([json.loads(line)[field] for line in handle if line.strip()], dtype=np.float32)


def _max_metric_difference(left, right):
    values = []
    for key, value in left.items():
        other = right[key]
        if value is None and other is None:
            continue
        values.append(abs(float(value) - float(other)))
    return max(values, default=0.0)


def main() -> None:
    master = _load(SEARCH_ROOT / "manifests/phase12_search_manifest_v1.json")
    if master["status"] != "COMPLETE":
        raise RuntimeError("Phase-12 search must complete before reproducibility audit")
    phase11 = _load(ROOT / master["phase11_manifest_path"])
    space = _load(ROOT / master["search_space_path"])
    best = _load(ROOT / master["best_xgb_manifest_path"])
    gateway = Phase10XGBData(ROOT, ROOT / phase11["flat_feature_map_path"])
    train, validation = gateway.load_partition("train"), gateway.load_partition("validation")
    lineage = _lineage(master, phase11, master["execution_commit"])
    reports = {}
    with tempfile.TemporaryDirectory(prefix="phase12-xgb-reproduction-") as temporary:
        temporary = Path(temporary)
        for task in ("recovery", "icu_time", "organ_support"):
            manifest = _load(ROOT / master["candidate_manifests"][task]["path"])
            candidates = validate_candidate_manifest(manifest, space)
            candidate_id = best["tasks"][task]["candidate_id"]
            candidate = next(item for item in candidates if item["candidate_id"] == candidate_id)
            train_matrices, validation_matrices = _task_matrices(gateway, train, validation, task)
            kwargs = {}
            model_task = task
            if task == "recovery":
                kwargs.update(recovery_scaler=gateway.recovery_scaler, recovery_scaler_path=gateway.recovery_scaler_path)
            elif task == "icu_time":
                model_task = "icu_stay_time"
            else:
                kwargs.update(support_class_weight=gateway.support_class_weight, support_class_weight_path=gateway.class_weight_path)
            bundle = train_xgb_candidate(
                task=model_task, candidate_id=candidate_id, config=_xgb_config(candidate),
                train_matrices=train_matrices, validation_matrices=validation_matrices,
                output_directory=temporary / task, lineage=lineage,
                feature_names=[entry["flat_name"] for entry in gateway.flat_map["entries"]],
                seed=candidate["model_seed"], run_type=RUN_PHASE12_SCIENTIFIC,
                phase12_authorization="PHASE12_FROZEN_CANDIDATE_MANIFEST", **kwargs,
            )
            winning_metrics = best["tasks"][task]["metrics"]
            attempt_dir = ROOT / winning_metrics["metrics_path"]
            attempt_dir = attempt_dir.parent
            if task == "recovery":
                actual = bundle.predict(validation.X)
                expected = np.column_stack((
                    _saved_predictions(attempt_dir / "validation_predictions_24h.jsonl", "prediction"),
                    _saved_predictions(attempt_dir / "validation_predictions_48h.jsonl", "prediction"),
                ))
                metrics, _, _, _ = _recovery_metrics(validation, actual)
            elif task == "icu_time":
                raw = bundle.predict_log(validation.X)
                actual = raw
                expected = _saved_predictions(attempt_dir / "validation_predictions.jsonl", "prediction_log1p")
                hours = remaining_icu_hours_from_log_prediction(torch.from_numpy(raw)).numpy()
                metrics, _, _, _ = _icu_metrics(validation, raw, hours)
            else:
                actual = bundle.predict(validation.X)
                expected = _saved_predictions(attempt_dir / "validation_predictions.jsonl", "prediction")
                metrics, _, _ = _support_metrics(validation, actual)
            prediction_difference = float(np.max(np.abs(actual - expected)))
            metric_difference = _max_metric_difference(metrics, winning_metrics)
            if prediction_difference > 1e-7 or metric_difference > 1e-12:
                raise RuntimeError("winner reproducibility tolerance exceeded: " + task)
            reports[task] = {
                "candidate_id": candidate_id, "same_scientific_slot": True,
                "new_candidate_slot_consumed": False, "prediction_max_abs_difference": prediction_difference,
                "metric_max_abs_difference": metric_difference, "prediction_tolerance": 1e-7,
                "metric_tolerance": 1e-12, "status": "PASS",
            }
    output = {
        "audit_version": "phase12_xgb_winner_reproducibility_v1",
        "status": "PASS", "search_version": master["search_version"],
        "execution_commit": master["execution_commit"], "tasks": reports,
        "scientific_candidate_slots_added": 0, "test_accessed": False,
    }
    path = SEARCH_ROOT / "winner_reproducibility_audit_v1.json"
    path.write_bytes(canonical_json_bytes(output))
    print(json.dumps(output, sort_keys=True))


if __name__ == "__main__":
    main()
