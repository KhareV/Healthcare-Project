"""Deterministic Phase-7 feature-input bundle generation and validation."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json, os, shutil, tempfile
from pathlib import Path
from typing import Optional

from data.schema import CanonicalFeatureInput, validate_canonical_feature_input
from data.synthetic.config import canonical_json_bytes
from data.synthetic.processed_manifest import validate_processed_manifest
from data.synthetic.provenance import semantic_sha256, sha256_file
from data.synthetic.validation import load_jsonl
from features.synthetic import FeatureBuildContext, SyntheticCanonicalFeatureBuilder, _dt, load_feature_schema


MANIFEST_VERSION = "synthetic_feature_input_manifest_v1"


def _mapping(item):
    return {
        "subject_id":item.subject_id,"stay_id":item.stay_id,"prediction_time":item.prediction_time,
        "grid_index":item.grid_index,"icu_elapsed_hours":item.icu_elapsed_hours,
        "tensor_contract_version":item.tensor_contract_version,"timestamp_spec_version":item.timestamp_spec_version,
        "feature_schema_version":item.feature_schema_version,"feature_schema_sha256":item.feature_schema_sha256,
        "temporal_feature_names":list(item.temporal_feature_names),
        "history_dtype":item.history_dtype,"history_values":[list(x) for x in item.history_values],
        "padding_mask_dtype":item.padding_mask_dtype,"padding_mask":list(item.padding_mask),
        "observation_mask_dtype":item.observation_mask_dtype,"observation_mask":[list(x) for x in item.observation_mask],
        "tslo_hours":[list(x) for x in item.tslo_hours],"tslo_status":item.tslo_status,
        "static_features":list(item.static_features),"static_feature_names":list(item.static_feature_names),
        "static_status":item.static_status,"quality_metadata":dict(item.quality_metadata),"case_tags":list(item.case_tags),
    }


def build_feature_bundle(*, processed_manifest_path:Path, structural_index_path:Path, feature_schema_path:Path, output:Path, root:Path, mode:str, generation_timestamp:Optional[str]=None):
    processed_manifest_path=processed_manifest_path if processed_manifest_path.is_absolute() else root/processed_manifest_path
    structural_index_path=structural_index_path if structural_index_path.is_absolute() else root/structural_index_path
    feature_schema_path=feature_schema_path if feature_schema_path.is_absolute() else root/feature_schema_path
    output=output if output.is_absolute() else root/output
    processed=validate_processed_manifest(processed_manifest_path,root)
    if mode=="final":
        raise ValueError("final feature generation requires an authorized final canonical timeline")
    if mode!="engineering" or not processed["manifest_status"].startswith("ENGINEERING_"):
        raise ValueError("engineering feature build requires engineering canonical timeline")
    artifacts={a["logical_name"]:root/a["repository_relative_path"] for a in processed["artifacts"]}
    timeline=load_jsonl(artifacts["canonical_timeline"]); statics=load_jsonl(artifacts["canonical_statics"]); structural=load_jsonl(structural_index_path)
    by_stay={row["stay_id"]:[] for row in statics}
    for row in timeline: by_stay[row["stay_id"]].append(row)
    statics_by={row["stay_id"]:row for row in statics}
    builder=SyntheticCanonicalFeatureBuilder(schema_path=feature_schema_path,root=root,statics_by_stay=statics_by)
    seen=set(); results=[]
    for row in sorted(structural,key=lambda x:(x["stay_id"],x["prediction_time"])):
        key=(row["stay_id"],row["prediction_time"])
        if key in seen: raise ValueError("duplicate structural identity")
        seen.add(key); static=statics_by.get(row["stay_id"])
        if static is None or static["subject_id"]!=row["subject_id"]: raise ValueError("structural/static identity mismatch")
        context=FeatureBuildContext(subject_id=row["subject_id"],stay_id=row["stay_id"],intime=_dt(static["intime"]),outtime=_dt(static["outtime"]),events=tuple(by_stay[row["stay_id"]]),prediction_time=_dt(row["prediction_time"]),prediction_time_text=row["prediction_time"],grid_index=row["grid_index"],icu_elapsed_hours=row["icu_elapsed_hours"])
        results.append(builder.build(context))
    rows=[_mapping(item) for item in results]
    output.parent.mkdir(parents=True,exist_ok=True); temp=Path(tempfile.mkdtemp(prefix=".phase7-",dir=output.parent))
    try:
        feature_file=temp/"canonical_feature_inputs.jsonl"
        with feature_file.open("wb") as handle:
            for row in rows: handle.write(canonical_json_bytes(row))
        total_cells=len(rows)*8*builder.feature_schema.feature_dim
        observed=Counter(); sentinel=Counter(); padding=Counter(sum(item.padding_mask) for item in results)
        for item in results:
            for f,name in enumerate(builder.feature_schema.feature_names):
                observed[name]+=sum(row[f] for row in item.observation_mask)
                sentinel[name]+=sum(row[f]==builder.feature_schema.tslo_no_observation_value for row in item.tslo_hours)
        qa={"qa_version":"synthetic_feature_qa_v1","status":"ENGINEERING_DESCRIPTIVE_NO_OUTCOMES","structural_rows":len(structural),"built_examples":len(rows),"F":builder.feature_schema.feature_dim,"raw_S":len(builder.feature_schema.static_feature_names or ()),"padding_bin_distribution":{str(k):v for k,v in sorted(padding.items())},"observation_mask_rate":{k:observed[k]/(len(rows)*8) for k in builder.feature_schema.feature_names},"tslo_sentinel_rate":{k:sentinel[k]/(len(rows)*8) for k in builder.feature_schema.feature_names},"raw_missing_cells":total_cells-sum(observed.values()),"split_status":"NOT_ASSIGNED_PHASE10","label_status":"NOT_CREATED_PHASE8_OR9","preprocessor_status":"NOT_FIT_PHASE10"}
        (temp/"feature_qa.json").write_bytes(canonical_json_bytes(qa))
        schema_hash=sha256_file(feature_schema_path)
        manifest={"manifest_version":MANIFEST_VERSION,"status":"ENGINEERING_CANONICAL_FEATURE_INPUTS","project_scope_version":processed["project_scope_version"],"project_scope_sha256_current":sha256_file(root/"docs/governance/project_scope_v2.md"),"processed_manifest_path":str(processed_manifest_path.relative_to(root)),"processed_manifest_sha256":sha256_file(processed_manifest_path),"structural_index_path":str(structural_index_path.relative_to(root)),"structural_index_sha256":sha256_file(structural_index_path),"feature_schema_path":str(feature_schema_path.relative_to(root)),"feature_schema_version":builder.feature_schema.version,"feature_schema_sha256":schema_hash,"timestamp_spec_sha256":builder.spec["timestamp_spec"]["sha256"],"tensor_contract_sha256":builder.spec["tensor_contract"]["sha256"],"builder_version":builder.version,"builder_path":"src/features/synthetic.py","builder_sha256":sha256_file(root/"src/features/synthetic.py"),"example_count":len(rows),"F":builder.feature_schema.feature_dim,"raw_S":len(builder.feature_schema.static_feature_names or ()),"support_dependency_status":"NO_SUPPORT_CHANNELS_IN_V1_PHASE9_NOT_FABRICATED","labels":"NOT_CREATED","split":"NOT_ASSIGNED","preprocessor":"NOT_FIT","generation_timestamp_utc":generation_timestamp or datetime.now(timezone.utc).isoformat().replace("+00:00","Z"),"artifacts":[{"logical_name":"canonical_feature_inputs","path":str((output/feature_file.name).relative_to(root)),"sha256":sha256_file(feature_file),"semantic_sha256":semantic_sha256("canonical_feature_inputs",rows),"record_count":len(rows)},{"logical_name":"feature_qa","path":str((output/"feature_qa.json").relative_to(root)),"sha256":sha256_file(temp/"feature_qa.json"),"semantic_sha256":sha256_file(temp/"feature_qa.json"),"record_count":1}]}
        (temp/"synthetic_feature_input_manifest_v1.json").write_bytes(canonical_json_bytes(manifest)); os.replace(temp,output)
    except Exception:
        shutil.rmtree(temp,ignore_errors=True); raise
    return output/"synthetic_feature_input_manifest_v1.json"


def validate_feature_manifest(path:Path,root:Path):
    manifest=json.loads(path.read_text())
    if manifest.get("manifest_version")!=MANIFEST_VERSION: raise ValueError("feature manifest version mismatch")
    for field,path_field in (("processed_manifest_sha256","processed_manifest_path"),("structural_index_sha256","structural_index_path"),("feature_schema_sha256","feature_schema_path"),("builder_sha256","builder_path")):
        if sha256_file(root/manifest[path_field])!=manifest[field]: raise ValueError(field+" mismatch")
    loaded={}
    for artifact in manifest["artifacts"]:
        if sha256_file(root/artifact["path"])!=artifact["sha256"]: raise ValueError("feature artifact hash mismatch")
        loaded[artifact["logical_name"]]=load_jsonl(root/artifact["path"]) if artifact["logical_name"]=="canonical_feature_inputs" else json.loads((root/artifact["path"]).read_text())
    rows=loaded["canonical_feature_inputs"]
    if semantic_sha256("canonical_feature_inputs",rows)!=next(a["semantic_sha256"] for a in manifest["artifacts"] if a["logical_name"]=="canonical_feature_inputs"): raise ValueError("feature semantic hash mismatch")
    _,schema=load_feature_schema(root/manifest["feature_schema_path"],root)
    seen=set()
    for row in rows:
        item=CanonicalFeatureInput(subject_id=row["subject_id"],stay_id=row["stay_id"],prediction_time=row["prediction_time"],grid_index=row["grid_index"],icu_elapsed_hours=row["icu_elapsed_hours"],tensor_contract_version=row["tensor_contract_version"],timestamp_spec_version=row["timestamp_spec_version"],feature_schema_version=row["feature_schema_version"],feature_schema_sha256=row["feature_schema_sha256"],temporal_feature_names=tuple(row["temporal_feature_names"]),history_dtype=row["history_dtype"],history_values=tuple(tuple(x) for x in row["history_values"]),padding_mask_dtype=row["padding_mask_dtype"],padding_mask=tuple(row["padding_mask"]),observation_mask_dtype=row["observation_mask_dtype"],observation_mask=tuple(tuple(x) for x in row["observation_mask"]),tslo_hours=tuple(tuple(x) for x in row["tslo_hours"]),tslo_status=row["tslo_status"],static_features=tuple(row["static_features"]),static_feature_names=tuple(row["static_feature_names"]),static_status=row["static_status"],quality_metadata=row["quality_metadata"],case_tags=tuple(row["case_tags"]))
        validate_canonical_feature_input(item,schema); key=(item.stay_id,item.prediction_time)
        if key in seen: raise ValueError("duplicate canonical feature identity")
        seen.add(key)
    if len(rows)!=manifest["example_count"]: raise ValueError("feature example count mismatch")
    return manifest
