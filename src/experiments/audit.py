"""Candidate-versus-attempt accounting for governed searches."""

import csv
import json
from collections import Counter
from pathlib import Path

from experiments.search_governance import SEARCH_BUDGET, canonical_sha256
from experiments.lineage import (
    RESULT_TYPES,
    LineageValidationError,
    artifact_descendants,
    read_artifact_index,
    read_run_registry,
    validate_artifact_lineage,
)
from vedant_infra.hashing import sha256_file


def read_registry_rows(path):
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return tuple(csv.DictReader(handle))


def audit_search(manifest, candidates, registry_rows=()):
    candidate_ids = {candidate["candidate_id"] for candidate in candidates}
    rows = [
        row
        for row in registry_rows
        if row.get("search_version") == manifest["search_version"]
    ]
    attempted_candidates = {row.get("candidate_id") for row in rows}
    completed = {
        row.get("candidate_id") for row in rows if row.get("status") == "completed"
    }
    failed_attempts = sum(row.get("status") == "failed" for row in rows)
    retries = sum(bool(row.get("retry_of_run_id")) for row in rows)
    unauthorized_attempts = sum(
        row.get("candidate_id") not in candidate_ids for row in rows
    )
    return {
        "task": manifest["task"],
        "family": manifest["family"],
        "search_version": manifest["search_version"],
        "search_space_hash": manifest["search_space_hash"],
        "candidate_list_hash": manifest["candidate_list_hash"],
        "planned_candidates": len(candidates),
        "unique_candidate_configs": len(
            {candidate["config_hash"] for candidate in candidates}
        ),
        "candidate_list_hash_verified": canonical_sha256(candidates)
        == manifest["candidate_list_hash"],
        "attempted_candidates": len(attempted_candidates & candidate_ids),
        "execution_attempts": len(rows),
        "completed_candidates": len(completed & candidate_ids),
        "failed_attempts": failed_attempts,
        "retry_attempts": retries,
        "unauthorized_attempts": unauthorized_attempts,
        "exact_budget_valid": len(candidates) == SEARCH_BUDGET,
        "validation_objective": manifest["validation_objective"],
        "test_accessed": False,
        "run_type": manifest["run_type"],
    }


def format_audit(summary):
    return json.dumps(summary, indent=2, sort_keys=True)


def audit_registry_lineage(root):
    """Return a non-mutating, non-PHI summary of run and artifact integrity."""

    root = Path(root)
    runs = read_run_registry(root / "experiments/registry.csv")
    artifacts = read_artifact_index(root / "experiments/artifacts.csv")
    run_ids = [row.get("run_id", "") for row in runs]
    known_runs = set(run_ids)
    known_artifacts = {record.artifact_id for record in artifacts}
    known_paths = {record.artifact_path for record in artifacts}
    broken_parents = []
    for row in runs:
        for field in ("parent_run_id", "retry_of_run_id"):
            parent = row.get(field, "")
            if parent and parent not in known_runs:
                broken_parents.append(row.get("run_id", "") + ":" + parent)
    missing = []
    mismatches = []
    broken_artifact_parents = []
    unknown_producers = []
    run_reference_mismatches = []
    for record in artifacts:
        path = root / record.artifact_path
        if not path.is_file():
            missing.append(record.artifact_id)
        elif sha256_file(path) != record.artifact_sha256:
            mismatches.append(record.artifact_id)
        if record.producing_run_id and record.producing_run_id not in known_runs:
            unknown_producers.append(record.artifact_id)
        broken_artifact_parents.extend(
            record.artifact_id + ":" + parent
            for parent in record.parents
            if parent not in known_artifacts
        )
    for row in runs:
        for ref_field, hash_field in (
            ("config_ref", "config_hash"),
            ("model_artifact_ref", "model_sha256"),
            ("metrics_ref", "metrics_sha256"),
            ("preprocessor_ref", "preprocessor_sha256"),
        ):
            reference = row.get(ref_field, "")
            expected = row.get(hash_field, "")
            if not reference or not expected:
                continue
            path = root / reference
            if not path.is_file():
                run_reference_mismatches.append(row.get("run_id", "") + ":MISSING:" + ref_field)
            elif sha256_file(path) != expected:
                run_reference_mismatches.append(row.get("run_id", "") + ":HASH:" + ref_field)
    incompatible_dependencies = []
    try:
        validate_artifact_lineage(artifacts, runs, repository_root=root)
    except LineageValidationError as error:
        incompatible_dependencies.append(str(error))
    artifact_root = root / "artifacts"
    disk_paths = {
        str(path.relative_to(root))
        for path in artifact_root.rglob("*")
        if path.is_file()
    }
    orphans = tuple(sorted(disk_paths - known_paths))
    untraceable_results = tuple(
        record.artifact_id
        for record in artifacts
        if record.artifact_type in RESULT_TYPES and record.status == "legacy_incomplete"
    )
    final_prefixes = (
        "artifacts/models/",
        "artifacts/predictions/",
        "artifacts/evaluation/",
        "artifacts/error_analysis/final/",
        "artifacts/ablations/final/",
    )
    scientific_orphans = tuple(
        path for path in orphans if path.startswith(final_prefixes)
    )
    return {
        "run_count": len(runs),
        "runs_by_task": dict(Counter(row.get("task", "") for row in runs)),
        "runs_by_family": dict(Counter(row.get("model_family", "") for row in runs)),
        "runs_by_status": dict(Counter(row.get("status", "") for row in runs)),
        "runs_by_type": dict(Counter(row.get("run_type", "") for row in runs)),
        "duplicate_run_ids": len(run_ids) - len(set(run_ids)),
        "artifact_count": len(artifacts),
        "orphan_artifacts": orphans,
        "scientific_orphan_artifacts": scientific_orphans,
        "missing_artifacts": tuple(missing),
        "hash_mismatches": tuple(mismatches),
        "broken_parent_runs": tuple(broken_parents),
        "broken_artifact_parents": tuple(broken_artifact_parents),
        "unknown_artifact_producers": tuple(unknown_producers),
        "run_reference_mismatches": tuple(run_reference_mismatches),
        "incompatible_dependencies": tuple(incompatible_dependencies),
        "untraceable_result_artifacts": untraceable_results,
        "incomplete_historical_provenance": sum(
            record.status == "legacy_incomplete" for record in artifacts
        ),
        "final_test_accessed": False,
        "g3_marker_created": False,
    }
