"""Phase-5 canonical timeline build, provenance manifest, and verification."""
from __future__ import annotations
from collections import Counter
from datetime import datetime, timezone
import csv, json, os, shutil, tempfile
from pathlib import Path
from typing import Optional
from .cohort import SyntheticCohortError
from .cohort_manifest import validate_cohort_manifest
from .config import canonical_json_bytes, load_runtime_config
from .manifest import validate_manifest as validate_raw_manifest
from .normalization import SyntheticTimelineError, normalize_events, validate_timeline
from .provenance import environment_identity, git_identity, semantic_sha256, sha256_file, source_bundle_sha256
from .validation import load_jsonl

PROCESSED_MANIFEST_VERSION="synthetic_processed_manifest_v1"

def _write_jsonl(path,rows):
    with path.open("wb") as handle:
        for row in rows: handle.write(canonical_json_bytes(row))

def _artifact(root,temp,out,name,filename,rows,schema_version):
    source=temp/filename; published=out/filename
    return {"logical_name":name,"repository_relative_path":str(published.relative_to(root)),"format":"JSONL_CANONICAL_UTF8","schema_version":schema_version,"record_count":len(rows),"sha256":sha256_file(source),"semantic_sha256":semantic_sha256(name,rows)}

def _load_upstream(cohort_manifest_path,root):
    cohort=json.loads(cohort_manifest_path.read_text()); spec=root/cohort["cohort_spec_path"]; validate_cohort_manifest(cohort_manifest_path,root,spec)
    phase3_path=root/cohort["source_manifest_path"]; phase3=json.loads(phase3_path.read_text()); runtime=load_runtime_config(root/phase3["generator_config_source_path"],root); validate_raw_manifest(phase3_path,root,runtime,root/"configs/synthetic/synthetic_schema_v1.json")
    cohort_artifacts={a["logical_name"]:root/a["repository_relative_path"] for a in cohort["artifacts"]}; raw_artifacts={a["logical_name"]:root/a["repository_relative_path"] for a in phase3["artifacts"]}
    return cohort,phase3,{k:load_jsonl(v) for k,v in cohort_artifacts.items() if k in {"retained_cohort","structural_index"}},{k:load_jsonl(v) for k,v in raw_artifacts.items()}

def build_processed_bundle(cohort_manifest_path: Path,concept_map_path: Path,processed_schema_path: Path,output: Path,root: Path,*,mode: str,generation_command: str,generation_timestamp: Optional[str]=None):
    if output.exists(): raise FileExistsError(f"output already exists; refusing overwrite: {output}")
    cohort,phase3,cohort_tables,raw_tables=_load_upstream(cohort_manifest_path,root)
    if mode=="final" and not (cohort["manifest_status"]=="RETAINED_SYNTHETIC_COHORT" and phase3["manifest_status"]=="AUTHORIZED_FINAL_SYNTHETIC_DATA"): raise SyntheticTimelineError("final timeline requires authorized final raw data and retained final cohort")
    if mode=="engineering" and not cohort["manifest_status"].startswith("ENGINEERING_COHORT_"): raise SyntheticTimelineError("engineering timeline requires engineering cohort")
    if mode not in {"engineering","final"}: raise SyntheticTimelineError("mode expected engineering or final")
    concept_map=json.loads(concept_map_path.read_text()); schema=json.loads(processed_schema_path.read_text())
    rows,filtered=normalize_events(raw_tables["raw_events"],cohort_tables["retained_cohort"],concept_map,phase3["raw_schema_version"]); validate_timeline(rows,schema,concept_map)
    subjects={r["subject_id"]:r for r in raw_tables["subjects"]}; statics=[]
    for stay in cohort_tables["retained_cohort"]:
        subject=subjects[stay["subject_id"]]
        statics.append({"subject_id":stay["subject_id"],"stay_id":stay["stay_id"],"age_years":subject["age_years"],"sex_category":subject["sex_category"],"cardiac_condition_group":subject["cardiac_condition_group"],"intime":stay["intime"],"outtime":stay["outtime"],"provenance_id":"PROV-STATIC-SUBJECT-EPISODE-V1"})
    statics.sort(key=lambda r:(r["subject_id"],r["stay_id"]))
    output.parent.mkdir(parents=True,exist_ok=True); temp=Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-",dir=output.parent))
    try:
        _write_jsonl(temp/"canonical_timeline.jsonl",rows); _write_jsonl(temp/"canonical_statics.jsonl",statics)
        by_concept=Counter(r["canonical_concept"] for r in rows); by_kind=Counter(r["event_kind"] for r in rows); by_stay=Counter(r["stay_id"] for r in rows)
        summary={"summary_version":"synthetic_timeline_summary_v1","retained_stays":len(statics),"canonical_event_rows":len(rows),"source_events_filtered_nonretained":filtered,"event_counts_by_concept":dict(sorted(by_concept.items())),"event_kind_counts":dict(sorted(by_kind.items())),"events_per_stay":{"minimum":min(by_stay.values()),"maximum":max(by_stay.values())},"invalid_or_rejected_events":0,"unknown_concepts":0,"out_of_bounds_events":0,"provenance_complete":True}
        (temp/"timeline_summary.json").write_bytes(canonical_json_bytes(summary))
        provenance_path=root/"docs/sanskruti/feature_provenance_v1.csv"; dictionary_path=root/"docs/sanskruti/FEATURE_DICTIONARY.md"; static_schema_path=root/"configs/synthetic/canonical_static_schema_v1.json"
        code={**git_identity(root),"source_bundle_sha256":source_bundle_sha256(root)}
        artifacts=[_artifact(root,temp,output,"canonical_timeline","canonical_timeline.jsonl",rows,schema["schema_version"]),_artifact(root,temp,output,"canonical_statics","canonical_statics.jsonl",statics,"synthetic_canonical_statics_v1")]
        summary_file=temp/"timeline_summary.json"; artifacts.append({"logical_name":"timeline_summary","repository_relative_path":str((output/"timeline_summary.json").relative_to(root)),"format":"JSON_CANONICAL_UTF8","schema_version":summary["summary_version"],"record_count":1,"sha256":sha256_file(summary_file),"semantic_sha256":sha256_file(summary_file)})
        source_status=cohort["manifest_status"].removeprefix("ENGINEERING_COHORT_") if hasattr(str,"removeprefix") else cohort["manifest_status"].replace("ENGINEERING_COHORT_","")
        manifest={"manifest_version":PROCESSED_MANIFEST_VERSION,"manifest_status":"ENGINEERING_CANONICAL_TIMELINE_"+source_status if mode=="engineering" else "CANONICAL_SYNTHETIC_TIMELINE","source_stage":"RAW_PHASE3_SYNTHETIC_EVENTS_PLUS_PHASE4_RETAINED_COHORT","project_scope_version":cohort["project_scope_version"],"project_scope_sha256":cohort["project_scope_sha256"],"phase3_manifest_path":cohort["source_manifest_path"],"phase3_manifest_sha256":sha256_file(root/cohort["source_manifest_path"]),"phase4_manifest_path":str(cohort_manifest_path.relative_to(root)),"phase4_manifest_sha256":sha256_file(cohort_manifest_path),"processed_schema_version":schema["schema_version"],"processed_schema_path":str(processed_schema_path.relative_to(root)),"processed_schema_sha256":sha256_file(processed_schema_path),"static_schema_version":"synthetic_canonical_statics_v1","static_schema_path":str(static_schema_path.relative_to(root)),"static_schema_sha256":sha256_file(static_schema_path),"concept_map_version":concept_map["concept_map_version"],"concept_map_path":str(concept_map_path.relative_to(root)),"concept_map_sha256":sha256_file(concept_map_path),"provenance_path":str(provenance_path.relative_to(root)),"provenance_sha256":sha256_file(provenance_path),"feature_dictionary_path":str(dictionary_path.relative_to(root)),"feature_dictionary_sha256":sha256_file(dictionary_path),"code_identity":code,"environment_identity":environment_identity(),"generation_timestamp_utc":generation_timestamp or datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00","Z"),"generation_command":generation_command,"counts":{"retained_stays":len(statics),"canonical_events":len(rows),"filtered_nonretained_events":filtered},"support_state_provenance":"DEFERRED_TO_PHASE_9","preprocessing_status":"NOT_FIT_PHASE_10","split_status":"NOT_ASSIGNED_PHASE_10","feature_schema_status":"NOT_FINALIZED_PHASE_7","labels_status":"NOT_CREATED_PHASE_8_OR_9","artifacts":artifacts}
        (temp/"synthetic_processed_manifest_v1.json").write_bytes(canonical_json_bytes(manifest)); os.replace(temp,output); validate_processed_manifest(output/"synthetic_processed_manifest_v1.json",root)
    except Exception:
        shutil.rmtree(temp,ignore_errors=True)
        if output.exists(): shutil.rmtree(output,ignore_errors=True)
        raise
    return output/"synthetic_processed_manifest_v1.json"

def validate_processed_manifest(path: Path,root: Path):
    m=json.loads(path.read_text())
    if m.get("manifest_version")!=PROCESSED_MANIFEST_VERSION: raise SyntheticTimelineError("processed manifest version mismatch")
    checks=(("phase3_manifest_sha256","phase3_manifest_path"),("phase4_manifest_sha256","phase4_manifest_path"),("processed_schema_sha256","processed_schema_path"),("static_schema_sha256","static_schema_path"),("concept_map_sha256","concept_map_path"),("provenance_sha256","provenance_path"),("feature_dictionary_sha256","feature_dictionary_path"))
    for hash_field,path_field in checks:
        if sha256_file(root/m[path_field])!=m[hash_field]: raise SyntheticTimelineError(f"{hash_field} mismatch")
    schema=json.loads((root/m["processed_schema_path"]).read_text()); concept_map=json.loads((root/m["concept_map_path"]).read_text()); records={}
    for a in m.get("artifacts",[]):
        target=root/a["repository_relative_path"]
        if not target.is_file() or sha256_file(target)!=a["sha256"]: raise SyntheticTimelineError(f"artifact missing or hash mismatch: {a['logical_name']}")
        value=load_jsonl(target) if a["format"].startswith("JSONL") else json.loads(target.read_text()); records[a["logical_name"]]=value
        semantic=semantic_sha256(a["logical_name"],value) if isinstance(value,list) else sha256_file(target)
        if semantic!=a["semantic_sha256"]: raise SyntheticTimelineError(f"semantic hash mismatch: {a['logical_name']}")
    if set(records)!={"canonical_timeline","canonical_statics","timeline_summary"}: raise SyntheticTimelineError("processed artifact inventory incomplete")
    validate_timeline(records["canonical_timeline"],schema,concept_map)
    if m["counts"]["canonical_events"]!=len(records["canonical_timeline"]) or m["counts"]["retained_stays"]!=len(records["canonical_statics"]): raise SyntheticTimelineError("processed manifest count mismatch")
    return m
