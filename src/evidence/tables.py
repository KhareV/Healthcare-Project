"""Deterministic Phase-22 tables with no manually entered result values."""

import csv
import io
import json
from pathlib import Path

from experiments.lineage import read_artifact_index
from vedant_infra.hashing import sha256_file


TASKS = ("recovery", "icu_stay_time", "organ_support")


def csv_text(fieldnames, rows):
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def tuning_budget_rows(xgb, gru):
    rows = []
    for task in TASKS:
        for family, source in (("xgboost", xgb), ("gru", gru)):
            rows.append(
                {
                    "task": task,
                    "family": family,
                    "role": "serving_candidate",
                    "planned_scientific_candidates": source["budget_per_task"],
                    "registered_real_candidates": "",
                    "execution_attempts": "",
                    "failed_attempts": "",
                    "retry_attempts": "",
                    "candidate_list_hash": "",
                    "search_space_hash": "",
                    "status": "BLOCKED",
                    "note": "real candidate manifest and execution registry absent",
                }
            )
        rows.extend(
            (
                {
                    "task": task,
                    "family": "lstm",
                    "role": "fixed_sensitivity_only",
                    "planned_scientific_candidates": 1,
                    "registered_real_candidates": "",
                    "execution_attempts": "",
                    "failed_attempts": "",
                    "retry_attempts": "",
                    "candidate_list_hash": "",
                    "search_space_hash": "",
                    "status": "BLOCKED",
                    "note": "one fixed run required; no independent tuning",
                },
                {
                    "task": task,
                    "family": "naive",
                    "role": "reference_only",
                    "planned_scientific_candidates": 0,
                    "registered_real_candidates": "",
                    "execution_attempts": "",
                    "failed_attempts": "",
                    "retry_attempts": "",
                    "candidate_list_hash": "",
                    "search_space_hash": "",
                    "status": "BLOCKED",
                    "note": "reference baseline is not a search candidate",
                },
            )
        )
    return rows


def artifact_hash_rows(root: Path, *, index_path=None):
    rows = []
    for record in read_artifact_index(index_path or root / "experiments/artifacts.csv"):
        if record.artifact_id.startswith("phase22:"):
            continue
        path = root / record.artifact_path
        rows.append(
            {
                "artifact_id": record.artifact_id,
                "artifact_type": record.artifact_type,
                "artifact_version": record.artifact_version,
                "artifact_path": record.artifact_path,
                "registered_sha256": record.artifact_sha256,
                "actual_sha256": sha256_file(path) if path.is_file() else "",
                "hash_status": "MATCH" if path.is_file() and sha256_file(path) == record.artifact_sha256 else "MISMATCH_OR_MISSING",
                "run_id": record.producing_run_id,
                "partition": record.partition,
                "status": record.status,
            }
        )
    required = (
        ("real_split", "artifacts/splits/split_v1.csv"),
        ("selected_models", "artifacts/models/selected_models_v1.json"),
        ("g3", "artifacts/governance/g3_freeze.json"),
        ("g4", "artifacts/governance/g4_evaluation_freeze.json"),
    )
    known_paths = {row["artifact_path"] for row in rows}
    for identifier, reference in required:
        if reference not in known_paths:
            rows.append(
                {
                    "artifact_id": identifier,
                    "artifact_type": "required_real_artifact",
                    "artifact_version": "",
                    "artifact_path": reference,
                    "registered_sha256": "",
                    "actual_sha256": "",
                    "hash_status": "BLOCKED_SOURCE_ARTIFACT_REQUIRED",
                    "run_id": "",
                    "partition": "",
                    "status": "BLOCKED",
                }
            )
    return rows


def reproducibility_rows(report):
    return [
        {
            "layer": name,
            "status": value["status"],
            "detail": value.get("detail", value.get("scope", "")),
        }
        for name, value in sorted(report["layers"].items())
    ]


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))
