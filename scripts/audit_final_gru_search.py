#!/usr/bin/env python3
"""Fail-closed Stage-1 final GRU search audit."""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src"))

from data.synthetic.provenance import sha256_file
from experiments.gru_final import SEARCH_VERSION, TASKS, choose_best, validate_candidates, validate_checkpoint_lineage
from experiments.lineage import read_artifact_index, read_run_registry, validate_artifact_lineage
from vedant_infra.registry import validate_registry

SEARCH=ROOT/"artifacts/search/gru/final_v2"


def load(path): return json.loads(path.read_text(encoding="utf-8"))


def main():
    master=load(SEARCH/"manifests/search_manifest_v1.json")
    if master.get("status")!="COMPLETE" or master.get("search_version")!=SEARCH_VERSION or master.get("test_accessed") is not False:
        raise RuntimeError("final GRU master manifest is not complete/test-locked")
    if sha256_file(ROOT/master["g1_path"])!=master["g1_sha256"] or sha256_file(ROOT/master["phase14_handoff_path"])!=master["phase14_handoff_sha256"]:
        raise RuntimeError("G1/Phase-14 GRU parent mismatch")
    best=load(SEARCH/"best_gru_candidates_v1.json")
    if (best.get("status")!="GRU_WITHIN_FAMILY_VALIDATION_WINNER" or best.get("serving_selection") is not False
            or best.get("test_accessed") is not False or best.get("support_calibrated") is not False or best.get("support_threshold") is not None):
        raise RuntimeError("best-GRU governance boundary violation")
    counts={}
    for task in TASKS:
        item=master["candidate_manifests"][task]; manifest=load(ROOT/item["path"])
        if sha256_file(ROOT/item["path"])!=item["sha256"]: raise RuntimeError("candidate manifest hash mismatch")
        validate_candidates(manifest["candidates"],task)
        summary=load(SEARCH/f"{task}_validation_summary.json")["candidates"]
        if len(summary)!=30: raise RuntimeError("summary does not contain exact-30 candidates")
        selected=choose_best(task,summary)
        if selected["candidate_id"]!=best["tasks"][task]["candidate_id"]: raise RuntimeError("best-GRU ranking mismatch")
        lineage={"g1_sha256":master["g1_sha256"],"phase14_handoff_sha256":master["phase14_handoff_sha256"],
                 "preprocessor_sha256":master["preprocessor_sha256"],"config_hash":selected["config_hash"]}
        validate_checkpoint_lineage(ROOT/selected["checkpoint_path"],lineage)
        counts[task]={"complete":sum(row["status"]=="COMPLETE" for row in summary),"failed":sum(row["status"]=="FAILED" for row in summary)}
        if counts[task]["complete"]<1: raise RuntimeError("task has no completed GRU candidate")
    if any(path.name.endswith("031") for path in SEARCH.rglob("gru-*-031")): raise RuntimeError("candidate 31 exists")
    audit=load(SEARCH/"winner_reproducibility_audit_v1.json")
    if audit.get("status")!="PASS" or any(row["status"]!="PASS" for row in audit["tasks"].values()): raise RuntimeError("winner reproducibility audit failed")
    validate_registry(ROOT/"experiments/registry.csv")
    validate_artifact_lineage(read_artifact_index(ROOT/"experiments/artifacts.csv"),read_run_registry(ROOT/"experiments/registry.csv"),repository_root=ROOT)
    if (ROOT/"artifacts/models/selected_models_v1.json").exists() or (ROOT/"artifacts/governance/g3_freeze.json").exists(): raise RuntimeError("forbidden downstream artifact exists")
    print(json.dumps({"status":"PASS","terminal_candidate_counts":counts,"best":{task:best["tasks"][task]["candidate_id"] for task in TASKS},"support_calibrated":False,"support_threshold":None,"test_accessed":False},sort_keys=True))


if __name__=="__main__": main()
