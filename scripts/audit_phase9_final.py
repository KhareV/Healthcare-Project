#!/usr/bin/env python3
"""Fail-closed audit of the local Phase-9 final pre-split build."""
import hashlib, json, math, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src"))
from data.synthetic.config import load_runtime_config
from data.synthetic.manifest import validate_manifest
from data.synthetic.cohort_manifest import validate_cohort_manifest
from data.synthetic.processed_manifest import validate_processed_manifest
from data.synthetic.provenance import sha256_file
from data.synthetic.validation import load_jsonl
from features.manifest import validate_feature_manifest

RAW=ROOT/"artifacts/data/synthetic/final/phase9_final_v1/synthetic_dataset_manifest_v1.json"
COHORT=ROOT/"artifacts/data/synthetic/cohorts/final/phase9_final_v1/synthetic_cohort_manifest_v2.json"
TIMELINE=ROOT/"artifacts/data/synthetic/timelines/final/phase9_final_v1/synthetic_processed_manifest_v1.json"
FEATURE=ROOT/"artifacts/data/synthetic/features/final/phase9_final_v2/synthetic_feature_input_manifest_v1.json"
PACKAGE=ROOT/"artifacts/data/synthetic/pre_split/final/phase9_final_v1/phase9_pre_split_manifest_v1.json"

def main():
    runtime=load_runtime_config(ROOT/"configs/synthetic/final_benchmark_v1.json",ROOT)
    raw=validate_manifest(RAW,ROOT,runtime,ROOT/"configs/synthetic/synthetic_schema_v1.json")
    cohort=validate_cohort_manifest(COHORT,ROOT,ROOT/"configs/synthetic/cohort_spec_v2.yaml")
    processed=validate_processed_manifest(TIMELINE,ROOT); feature=validate_feature_manifest(FEATURE,ROOT)
    package=json.loads(PACKAGE.read_text()); artifacts={x["logical_name"]:x for x in package["artifacts"]}
    for item in artifacts.values():
        if sha256_file(ROOT/item["path"])!=item["sha256"]: raise RuntimeError("package artifact hash mismatch: "+item["logical_name"])
    rows=load_jsonl(ROOT/artifacts["pre_split_scientific_package"]["path"]); qa=json.loads((PACKAGE.parent/"phase9_qa.json").read_text())
    if raw["generated_subject_count"]!=2000 or cohort["counts"]["retained_subjects"]!=2000 or package["subject_count"]!=2000: raise RuntimeError("final accepted subject count is not exactly 2000")
    if package["row_count"]!=len(rows) or feature["example_count"]!=len(rows) or package["feature_dimension"]!=21: raise RuntimeError("row or feature dimension mismatch")
    if any(r.get("split") is not None or r.get("preprocessor") is not None for r in rows): raise RuntimeError("split/preprocessor leakage in pre-split package")
    if any(len(r["history_values"])!=8 or any(len(x)!=21 for x in r["history_values"]) for r in rows): raise RuntimeError("feature tensor shape mismatch")
    names=rows[0]["temporal_feature_names"]; support_names=("vasopressor_on","invasive_ventilation_on","norepinephrine_rate","epinephrine_rate","dopamine_rate","dobutamine_rate")
    if tuple(names[-6:])!=support_names: raise RuntimeError("support feature order mismatch")
    for r in rows:
        for b,padded in enumerate(r["padding_mask"]):
            for index in range(15,21):
                if padded:
                    if r["observation_mask"][b][index]: raise RuntimeError("support padding mask leakage")
                elif not r["observation_mask"][b][index] or r["tslo_hours"][b][index]!=0.0: raise RuntimeError("support known-state mask/TSLO mismatch")
    if not (qa["support_positive"]>0 and qa["support_negative"]>0 and qa["recovery24"]["variance"]>0 and qa["recovery48"]["variance"]>0 and qa["icu_time_log1p"]["variance"]>0): raise RuntimeError("task feasibility failure")
    print(json.dumps({"status":"PASS","subjects":2000,"rows":len(rows),"F":21,"support_positive":qa["support_positive"],"support_negative":qa["support_negative"],"split":"NOT_CREATED_PHASE10","model_training":"NOT_PERFORMED"},sort_keys=True))

if __name__=="__main__": main()
