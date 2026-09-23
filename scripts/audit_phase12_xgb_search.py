#!/usr/bin/env python3
"""Fail-closed post-search audit for Phase-12 scientific XGBoost results."""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.synthetic.provenance import sha256_file
from experiments.audit import audit_registry_lineage, audit_search, read_registry_rows
from experiments.lineage import read_artifact_index, read_run_registry, validate_artifact_lineage
from experiments.xgb_phase12 import RankedCandidate, canonical_sha256, choose_best, validate_candidate_manifest, validate_master_manifest, validate_search_space


SEARCH_ROOT = ROOT / "artifacts/search/xgb/phase12"
MASTER_PATH = SEARCH_ROOT / "manifests/phase12_search_manifest_v1.json"


def _load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> None:
    master = _load(MASTER_PATH)
    validate_master_manifest(ROOT, master)
    if master["status"] != "COMPLETE" or master["test_accessed"] is not False:
        raise RuntimeError("Phase-12 search is not complete or test isolation failed")
    if master["calibration_performed"] or master["threshold_selected"] or master["family_selection_performed"]:
        raise RuntimeError("Phase 13+ activity leaked into Phase 12")
    selected = master["selected_models_v1_presearch"]
    selected_path = ROOT / selected["path"]
    if selected["present"] and (
        not selected_path.is_file()
        or sha256_file(selected_path) != selected["sha256"]
    ):
        raise RuntimeError("the selected_models_v1 artifact recorded by Phase 12 changed")
    # When the snapshot records absence, it proves the manifest did not exist
    # at Phase-12 execution time. A later governed stage may create it without
    # retroactively invalidating that immutable non-leakage declaration.
    space = _load(ROOT / master["search_space_path"]); validate_search_space(space)
    registry = read_registry_rows(ROOT / "experiments/registry.csv")
    budget = {}
    winners = {}
    for task, item in master["candidate_manifests"].items():
        manifest = _load(ROOT / item["path"])
        candidates = validate_candidate_manifest(manifest, space)
        summary = audit_search(
            {**manifest, "validation_objective": master["validation_objectives"][task]},
            candidates,
            registry,
        )
        # Phase 12 hashes the exact canonical serialized candidate payload,
        # including NumPy-generated float precision, with the synthetic-data
        # canonical serializer. Historical dry-run audit uses its older JSON
        # helper; keep both contracts intact and report the Phase-12 verifier.
        summary["candidate_list_hash_verified"] = (
            canonical_sha256(candidates) == manifest["candidate_list_hash"]
        )
        if not (
            summary["planned_candidates"] == 30
            and summary["unique_candidate_configs"] == 30
            and summary["attempted_candidates"] == 30
            and summary["completed_candidates"] == 30
            and summary["unauthorized_attempts"] == 0
            and summary["candidate_list_hash_verified"]
            and summary["test_accessed"] is False
        ):
            raise RuntimeError("Phase-12 exact-budget audit failed: " + task)
        result = _load(SEARCH_ROOT / f"{task}_validation_summary.json")
        rows = result["candidates"]
        if len(rows) != 30 or any(row["status"] != "COMPLETE" for row in rows):
            raise RuntimeError("Phase-12 candidate summary is incomplete: " + task)
        ranked = [RankedCandidate(row["candidate_id"], row["status"], row, "raw_uncalibrated" if task == "organ_support" else "") for row in rows]
        winners[task] = choose_best(ranked, task).candidate_id
        budget[task] = summary
    best = _load(ROOT / master["best_xgb_manifest_path"])
    if best["status"] != "WITHIN_FAMILY_VALIDATION_SELECTION" or best["test_accessed"] or best["serving_selection"] or best["support_calibrated"] or best["support_threshold"] is not None:
        raise RuntimeError("best-XGB manifest overstates Phase-12 authority")
    if {task: value["candidate_id"] for task, value in best["tasks"].items()} != winners:
        raise RuntimeError("best-XGB manifest disagrees with frozen ranking")
    artifacts = read_artifact_index(ROOT / "experiments/artifacts.csv")
    runs = read_run_registry(ROOT / "experiments/registry.csv")
    validate_artifact_lineage(artifacts, runs, repository_root=ROOT)
    lineage = audit_registry_lineage(ROOT)
    for field in ("missing_artifacts", "hash_mismatches", "broken_parent_runs", "broken_artifact_parents", "unknown_artifact_producers", "run_reference_mismatches", "incompatible_dependencies", "scientific_orphan_artifacts"):
        if lineage[field]:
            raise RuntimeError("Phase-12 lineage audit failed: " + field)
    output = {
        "status": "PASS", "search_version": master["search_version"],
        "budget": budget, "winners": winners, "test_accessed": False,
        "calibration_performed": False, "threshold_selected": False,
        "family_selection_performed": False,
    }
    print(json.dumps(output, sort_keys=True))


if __name__ == "__main__":
    main()
