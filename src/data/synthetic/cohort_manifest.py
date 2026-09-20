"""Phase-4 cohort artifact build, provenance chain, and verification."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json, os, shutil, tempfile
from pathlib import Path
from typing import Mapping, Optional

import yaml

from .cohort import COHORT_VERSION, INDEX_SCHEMA_VERSION, SyntheticCohortError, build_cohort, validate_output_schemas
from .config import canonical_json_bytes, load_runtime_config
from .manifest import validate_manifest as validate_source_manifest
from .provenance import environment_identity, git_identity, semantic_sha256, sha256_file, source_bundle_sha256
from .validation import load_jsonl

COHORT_MANIFEST_VERSION = "synthetic_cohort_manifest_v2"


def _write_jsonl(path: Path, rows) -> None:
    with path.open("wb") as handle:
        for row in rows: handle.write(canonical_json_bytes(row))


def _canonical_hash(value: object) -> str:
    import hashlib
    # Round-trip normalizes JSON object keys (for example integer count bins).
    normalized = json.loads(canonical_json_bytes(value))
    return hashlib.sha256(canonical_json_bytes(normalized)).hexdigest()


def _load_source(source_manifest_path: Path, root: Path):
    source = json.loads(source_manifest_path.read_text())
    config_path = root / source["generator_config_source_path"]
    runtime = load_runtime_config(config_path, root)
    schema_path = root / "configs/synthetic/synthetic_schema_v1.json"
    validate_source_manifest(source_manifest_path, root, runtime, schema_path)
    paths = {item["logical_name"]: root / item["repository_relative_path"] for item in source["artifacts"]}
    return source, runtime, {name:load_jsonl(path) for name,path in paths.items()}


def _validate_authorities(root: Path, spec_path: Path):
    spec = yaml.safe_load(spec_path.read_text())
    if spec.get("cohort_spec_version") != "synthetic_cohort_v2": raise SyntheticCohortError("cohort spec version mismatch")
    scope_path = root / spec["scope"]["path"]
    if sha256_file(scope_path) != spec["scope"]["sha256"]: raise SyntheticCohortError("project scope hash mismatch")
    timestamp_path = root / spec["timestamp_contract"]["implementation"]
    if sha256_file(timestamp_path) != spec["timestamp_contract"]["sha256"]: raise SyntheticCohortError("timestamp spec hash mismatch")
    cohort_schema_path = root / "configs/synthetic/retained_cohort_schema_v2.json"
    index_schema_path = root / "configs/synthetic/structural_index_schema_v2.json"
    cohort_schema=json.loads(cohort_schema_path.read_text()); index_schema=json.loads(index_schema_path.read_text())
    validate_output_schemas(cohort_schema,index_schema)
    return spec, timestamp_path, cohort_schema_path, index_schema_path


def _summary(build, source_status, source_hash, spec_hash, code_identity):
    counts=Counter(row["stay_id"] for row in build.structural_index); rows_per_stay=list(counts.values())
    flags=("recovery24_followup_available","recovery48_followup_available","support24_full_followup_available","icu_time_temporally_eligible")
    return {
        "summary_version":"synthetic_cohort_summary_v2","source_dataset_status":source_status,
        "source_manifest_sha256":source_hash,"cohort_spec_sha256":spec_hash,"cohort_code_identity":code_identity,
        "generated_subject_count":build.generated_subject_count,"generated_episode_count":build.generated_episode_count,
        "retained_subject_count":len(build.retained_cohort),"retained_stay_count":len(build.retained_cohort),
        "exclusion_counts":dict(build.exclusion_counts),"total_legal_prediction_rows":len(build.structural_index),
        "rows_per_stay":{"minimum":min(rows_per_stay) if rows_per_stay else 0,"maximum":max(rows_per_stay) if rows_per_stay else 0,"distribution":dict(sorted(Counter(rows_per_stay).items()))},
        "structural_horizon_availability_counts":{flag:sum(bool(row[flag]) for row in build.structural_index) for flag in flags},
        "outcome_free":True,"split_status":"NOT_ASSIGNED_PHASE_10","feature_status":"NOT_CREATED_PHASE_5_OR_7",
        "labels_status":"NOT_COMPUTED_PHASE_8_OR_9","clinical_validation_claim":False,
    }


def build_to_directory(source_manifest_path: Path, spec_path: Path, output: Path, root: Path, *, mode: str, generation_command: str, generation_timestamp: Optional[str]=None):
    if mode not in {"engineering","final"}: raise SyntheticCohortError("mode expected engineering or final")
    if output.exists(): raise FileExistsError(f"output already exists; refusing overwrite: {output}")
    source,runtime,tables=_load_source(source_manifest_path,root); source_status=source["manifest_status"]
    if mode=="final" and source_status!="AUTHORIZED_FINAL_SYNTHETIC_DATA": raise SyntheticCohortError(f"final cohort requires AUTHORIZED_FINAL_SYNTHETIC_DATA; observed {source_status}")
    if mode=="engineering" and source_status not in {"FIXTURE","SMOKE"}: raise SyntheticCohortError(f"engineering cohort requires FIXTURE or SMOKE source; observed {source_status}")
    spec,timestamp_path,cohort_schema_path,index_schema_path=_validate_authorities(root,spec_path)
    build=build_cohort(tables["subjects"],tables["episodes"],tables["raw_events"],minimum_age=int(spec["population"]["minimum_age_years"]),allowed_cardiac_groups=runtime.values["population"]["condition_groups"])
    output.parent.mkdir(parents=True,exist_ok=True); temporary=Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-",dir=output.parent))
    try:
        _write_jsonl(temporary/"retained_cohort.jsonl",build.retained_cohort); _write_jsonl(temporary/"structural_index.jsonl",build.structural_index)
        exclusions={"exclusion_report_version":"synthetic_cohort_exclusions_v2","counts":build.exclusion_counts,"outcome_based_reasons_present":False}
        (temporary/"exclusion_summary.json").write_bytes(canonical_json_bytes(exclusions))
        code_identity={**git_identity(root),"source_bundle_sha256":source_bundle_sha256(root)}; spec_hash=sha256_file(spec_path); source_hash=sha256_file(source_manifest_path)
        summary=_summary(build,source_status,source_hash,spec_hash,code_identity); (temporary/"cohort_summary.json").write_bytes(canonical_json_bytes(summary))
        table_artifacts=[]
        for name,rows,schema_version in (("retained_cohort",build.retained_cohort,COHORT_VERSION),("structural_index",build.structural_index,INDEX_SCHEMA_VERSION)):
            file=temporary/f"{name}.jsonl"; published=output/f"{name}.jsonl"
            table_artifacts.append({"logical_name":name,"repository_relative_path":str(published.relative_to(root)),"format":"JSONL_CANONICAL_UTF8","schema_version":schema_version,"record_count":len(rows),"sha256":sha256_file(file),"semantic_sha256":semantic_sha256(name,rows)})
        for name,value in (("exclusion_summary",exclusions),("cohort_summary",summary)):
            file=temporary/f"{name}.json"; published=output/f"{name}.json"
            table_artifacts.append({"logical_name":name,"repository_relative_path":str(published.relative_to(root)),"format":"JSON_CANONICAL_UTF8","schema_version":value[next(k for k in value if k.endswith("version"))],"record_count":1,"sha256":sha256_file(file),"semantic_sha256":_canonical_hash(value)})
        status_prefix="ENGINEERING_COHORT_"+source_status if mode=="engineering" else "RETAINED_SYNTHETIC_COHORT"
        manifest={
            "manifest_version":COHORT_MANIFEST_VERSION,"manifest_status":status_prefix,"structural_index_status":"ENGINEERING_OUTCOME_FREE_STRUCTURAL_INDEX_"+source_status if mode=="engineering" else "OUTCOME_FREE_STRUCTURAL_INDEX",
            "project_scope_version":spec["scope"]["version"],"project_scope_sha256":spec["scope"]["sha256"],
            "source_dataset_status":source_status,"source_manifest_path":str(source_manifest_path.relative_to(root)),"source_manifest_sha256":source_hash,
            "generator_version":source["generator_version"],"raw_schema_version":source["raw_schema_version"],
            "cohort_spec_version":spec["cohort_spec_version"],"cohort_spec_path":str(spec_path.relative_to(root)),"cohort_spec_sha256":spec_hash,
            "timestamp_spec_version":spec["timestamp_contract"]["version"],"timestamp_implementation_path":str(timestamp_path.relative_to(root)),"timestamp_spec_sha256":sha256_file(timestamp_path),
            "retained_cohort_schema_path":str(cohort_schema_path.relative_to(root)),"retained_cohort_schema_sha256":sha256_file(cohort_schema_path),
            "structural_index_schema_path":str(index_schema_path.relative_to(root)),"structural_index_schema_version":INDEX_SCHEMA_VERSION,"structural_index_schema_sha256":sha256_file(index_schema_path),
            "code_identity":code_identity,"environment_identity":environment_identity(),"generation_timestamp_utc":generation_timestamp or datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00","Z"),"generation_command":generation_command,
            "counts":{"generated_subjects":build.generated_subject_count,"generated_episodes":build.generated_episode_count,"retained_subjects":len(build.retained_cohort),"retained_stays":len(build.retained_cohort),"structural_rows":len(build.structural_index)},
            "exclusion_counts":build.exclusion_counts,"split_status":"NOT_ASSIGNED_PHASE_10","feature_status":"NOT_CREATED_PHASE_5_OR_7","label_status":"NOT_COMPUTED_PHASE_8_OR_9","artifacts":table_artifacts,
        }
        (temporary/"synthetic_cohort_manifest_v2.json").write_bytes(canonical_json_bytes(manifest)); os.replace(temporary,output)
        validate_cohort_manifest(output/"synthetic_cohort_manifest_v2.json",root,spec_path)
    except Exception:
        shutil.rmtree(temporary,ignore_errors=True)
        if output.exists(): shutil.rmtree(output,ignore_errors=True)
        raise
    return output/"synthetic_cohort_manifest_v2.json"


def validate_cohort_manifest(path: Path, root: Path, spec_path: Path):
    manifest=json.loads(path.read_text()); spec,timestamp_path,cohort_schema_path,index_schema_path=_validate_authorities(root,spec_path)
    checks=(("source_manifest_sha256",root/manifest["source_manifest_path"]),("cohort_spec_sha256",spec_path),("timestamp_spec_sha256",timestamp_path),("retained_cohort_schema_sha256",cohort_schema_path),("structural_index_schema_sha256",index_schema_path))
    for field,target in checks:
        if manifest.get(field)!=sha256_file(target): raise SyntheticCohortError(f"{field} mismatch")
    if manifest.get("manifest_version")!=COHORT_MANIFEST_VERSION: raise SyntheticCohortError("cohort manifest version mismatch")
    if manifest.get("split_status")!="NOT_ASSIGNED_PHASE_10": raise SyntheticCohortError("unexpected split status")
    records={}
    for item in manifest.get("artifacts",[]):
        target=root/item["repository_relative_path"]
        if not target.is_file() or sha256_file(target)!=item["sha256"]: raise SyntheticCohortError(f"artifact missing or hash mismatch: {item['logical_name']}")
        if item["format"].startswith("JSONL"):
            value=load_jsonl(target)
            if semantic_sha256(item["logical_name"],value)!=item["semantic_sha256"]: raise SyntheticCohortError(f"semantic hash mismatch: {item['logical_name']}")
            if len(value)!=item["record_count"]: raise SyntheticCohortError(f"record count mismatch: {item['logical_name']}")
        else:
            value=json.loads(target.read_text())
            if _canonical_hash(value)!=item["semantic_sha256"]: raise SyntheticCohortError(f"semantic hash mismatch: {item['logical_name']}")
        records[item["logical_name"]]=value
    if set(records)!={"retained_cohort","structural_index","exclusion_summary","cohort_summary"}: raise SyntheticCohortError("cohort manifest artifact inventory incomplete")
    _validate_output_only(records["retained_cohort"],records["structural_index"])
    expected=manifest["counts"]
    if expected["retained_subjects"]!=len(records["retained_cohort"]) or expected["structural_rows"]!=len(records["structural_index"]): raise SyntheticCohortError("cohort manifest counts mismatch")
    return manifest


def _validate_output_only(cohort,index):
    from .cohort import _validate_outputs
    _validate_outputs(cohort,index)
