"""Phase-8 deterministic partial target artifact and hash-bound validation."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json, math, os, shutil, tempfile
from pathlib import Path
from typing import Optional

from data.synthetic.cohort_manifest import validate_cohort_manifest
from data.synthetic.config import canonical_json_bytes
from data.synthetic.processed_manifest import validate_processed_manifest
from data.synthetic.provenance import semantic_sha256, sha256_file
from data.synthetic.validation import load_jsonl
from features.manifest import validate_feature_manifest
from labels.icu_time import remaining_episode_time_label


MANIFEST_VERSION = "synthetic_phase8_target_manifest_v1"


def _artifact(manifest, logical_name, root):
    item = next(x for x in manifest["artifacts"] if x["logical_name"] == logical_name)
    return root / item.get("repository_relative_path", item.get("path"))


def build_partial_target_bundle(*, cohort_manifest_path: Path, processed_manifest_path: Path,
                                feature_manifest_path: Path, recovery_spec_path: Path,
                                icu_spec_path: Path, target_schema_path: Path, target_provenance_path: Path, sofa_spec_path: Path,
                                output: Path, root: Path, mode: str,
                                generation_timestamp: Optional[str] = None) -> Path:
    paths = [cohort_manifest_path, processed_manifest_path, feature_manifest_path, recovery_spec_path,
             icu_spec_path, target_schema_path, target_provenance_path, sofa_spec_path]
    paths = [p if p.is_absolute() else root / p for p in paths]
    cohort_path, processed_path, feature_path, recovery_path, icu_path, schema_path, provenance_path, sofa_path = paths
    output = output if output.is_absolute() else root / output
    cohort_raw = json.loads(cohort_path.read_text())
    cohort = validate_cohort_manifest(cohort_path, root, root / cohort_raw["cohort_spec_path"])
    processed = validate_processed_manifest(processed_path, root)
    feature = validate_feature_manifest(feature_path, root)
    recovery_spec, icu_spec, schema, sofa_spec = (json.loads(p.read_text()) for p in (recovery_path, icu_path, schema_path, sofa_path))
    if mode == "final":
        raise ValueError("final Phase-8 targets require authorized final cohort and final Phase-9-backed SOFA")
    if mode != "engineering" or not cohort["manifest_status"].startswith("ENGINEERING_"):
        raise ValueError("engineering target build requires engineering cohort")
    if recovery_spec["sofa_spec"]["sha256"] != sha256_file(sofa_path):
        raise ValueError("recovery/SOFA specification hash mismatch")
    structural = load_jsonl(_artifact(cohort, "structural_index", root))
    retained = load_jsonl(_artifact(cohort, "retained_cohort", root))
    statics = load_jsonl(_artifact(processed, "canonical_statics", root))
    features = load_jsonl(_artifact(feature, "canonical_feature_inputs", root))
    stays = {x["stay_id"]: x for x in retained}
    static_by = {x["stay_id"]: x for x in statics}
    feature_keys = {(x["subject_id"], x["stay_id"], x["prediction_time"], x["grid_index"]) for x in features}
    rows, seen = [], set()
    for row in sorted(structural, key=lambda x: (x["stay_id"], x["prediction_time"])):
        key = (row["subject_id"], row["stay_id"], row["prediction_time"], row["grid_index"])
        if key in seen or key not in feature_keys:
            raise ValueError("duplicate or orphan structural/feature identity")
        seen.add(key)
        stay, static = stays.get(row["stay_id"]), static_by.get(row["stay_id"])
        if stay is None or static is None or stay["subject_id"] != row["subject_id"] or static["outtime"] != stay["outtime"]:
            raise ValueError("episode/static/structural identity mismatch")
        label = remaining_episode_time_label(
            prediction_time=datetime.fromisoformat(row["prediction_time"].replace("Z", "+00:00")),
            outtime=datetime.fromisoformat(stay["outtime"].replace("Z", "+00:00")),
            icu_time_temporally_eligible=row["icu_time_temporally_eligible"],
        )
        rows.append({"subject_id":row["subject_id"], "stay_id":row["stay_id"], "prediction_time":row["prediction_time"],
            "grid_index":row["grid_index"], "recovery24_structurally_eligible":row["recovery24_followup_available"],
            "recovery48_structurally_eligible":row["recovery48_followup_available"], "recovery24_eligible":None,
            "recovery48_eligible":None, "delta_sofa_24":None, "delta_sofa_48":None,
            "recovery_generation_status":"BLOCKED_PHASE6_FINAL_SOFA_REQUIRES_PHASE9_SUPPORT",
            "icu_time_eligible":label.eligible, "remaining_hours":label.remaining_hours,
            "icu_time_log1p":label.log1p_remaining_hours, "target_schema_version":schema["schema_version"]})
    if len(seen) != len(feature_keys):
        raise ValueError("orphan Phase-7 feature identity")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError("target output already exists; refusing overwrite")
    temp = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        target_file = temp / "phase8_targets_partial.jsonl"
        with target_file.open("wb") as handle:
            for row in rows: handle.write(canonical_json_bytes(row))
        counts = Counter((x["recovery24_structurally_eligible"], x["recovery48_structurally_eligible"]) for x in rows)
        qa = {"qa_version":"synthetic_phase8_target_qa_v1", "scope":"ENGINEERING_PARTIAL_NO_SPLIT",
            "structural_rows":len(rows), "icu_time_eligible":sum(x["icu_time_eligible"] for x in rows),
            "icu_time_formula_validated":len(rows), "recovery24_structurally_eligible":sum(x["recovery24_structurally_eligible"] for x in rows),
            "recovery48_structurally_eligible":sum(x["recovery48_structurally_eligible"] for x in rows),
            "recovery_24_only":counts[(True,False)], "recovery_both":counts[(True,True)], "recovery_neither":counts[(False,False)],
            "recovery_generated":0, "support_fields_generated":0, "split":"NOT_ASSIGNED", "target_scaler":"NOT_FIT"}
        (temp / "target_qa.json").write_bytes(canonical_json_bytes(qa))
        published_target = output / target_file.name; published_qa = output / "target_qa.json"
        manifest = {"manifest_version":MANIFEST_VERSION, "status":"ENGINEERING_PHASE8_PARTIAL_ICU_TIME_COMPLETE_RECOVERY_BLOCKED",
            "recovery_engine_status":"IMPLEMENTED_AND_TESTED", "final_recovery_artifact_status":"BLOCKED_PHASE6_FINAL_SOFA_REQUIRES_PHASE9_SUPPORT",
            "icu_time_engine_status":"IMPLEMENTED", "final_icu_time_artifact_status":"BLOCKED_NO_AUTHORIZED_FINAL_COHORT",
            "engineering_icu_time_artifact_status":"GENERATED", "combined_phase8_status":"PARTIAL",
            "cohort_manifest_path":str(cohort_path.relative_to(root)), "cohort_manifest_sha256":sha256_file(cohort_path),
            "structural_index_path":str(_artifact(cohort,"structural_index",root).relative_to(root)), "structural_index_sha256":sha256_file(_artifact(cohort,"structural_index",root)),
            "processed_manifest_path":str(processed_path.relative_to(root)), "processed_manifest_sha256":sha256_file(processed_path),
            "canonical_timeline_path":str(_artifact(processed,"canonical_timeline",root).relative_to(root)), "canonical_timeline_sha256":sha256_file(_artifact(processed,"canonical_timeline",root)),
            "feature_manifest_path":str(feature_path.relative_to(root)), "feature_manifest_sha256":sha256_file(feature_path),
            "feature_schema_path":feature["feature_schema_path"], "feature_schema_sha256":feature["feature_schema_sha256"],
            "sofa_spec_path":str(sofa_path.relative_to(root)), "sofa_spec_sha256":sha256_file(sofa_path), "sofa_status":sofa_spec["status"],
            "recovery_spec_path":str(recovery_path.relative_to(root)), "recovery_spec_sha256":sha256_file(recovery_path),
            "icu_time_spec_path":str(icu_path.relative_to(root)), "icu_time_spec_sha256":sha256_file(icu_path),
            "target_schema_path":str(schema_path.relative_to(root)), "target_schema_sha256":sha256_file(schema_path),
            "target_provenance_path":str(provenance_path.relative_to(root)), "target_provenance_sha256":sha256_file(provenance_path),
            "builder_path":"src/labels/synthetic_manifest.py", "builder_sha256":sha256_file(root/"src/labels/synthetic_manifest.py"),
            "recovery_engine_path":"src/labels/recovery.py", "recovery_engine_sha256":sha256_file(root/"src/labels/recovery.py"),
            "icu_time_engine_path":"src/labels/icu_time.py", "icu_time_engine_sha256":sha256_file(root/"src/labels/icu_time.py"),
            "row_count":len(rows), "generation_timestamp_utc":generation_timestamp or datetime.now(timezone.utc).isoformat().replace("+00:00","Z"),
            "generation_command":"python3 scripts/build_synthetic_labels.py ... --mode engineering", "split":"NOT_ASSIGNED_PHASE10",
            "preprocessing":"NOT_FIT_PHASE10", "target_scaler":"NOT_FIT_PHASE10", "organ_support":"NOT_CREATED_PHASE9",
            "artifacts":[{"logical_name":"phase8_targets_partial","path":str(published_target.relative_to(root)),"sha256":sha256_file(target_file),"semantic_sha256":semantic_sha256("phase8_targets_partial",rows),"record_count":len(rows)},
                         {"logical_name":"target_qa","path":str(published_qa.relative_to(root)),"sha256":sha256_file(temp/"target_qa.json"),"record_count":1}]}
        (temp / "synthetic_phase8_target_manifest_v1.json").write_bytes(canonical_json_bytes(manifest))
        os.replace(temp, output)
        validate_target_manifest(output / "synthetic_phase8_target_manifest_v1.json", root)
    except Exception:
        shutil.rmtree(temp, ignore_errors=True)
        if output.exists(): shutil.rmtree(output, ignore_errors=True)
        raise
    return output / "synthetic_phase8_target_manifest_v1.json"


def validate_target_manifest(path: Path, root: Path):
    manifest = json.loads(path.read_text())
    if manifest.get("manifest_version") != MANIFEST_VERSION or manifest.get("organ_support") != "NOT_CREATED_PHASE9":
        raise ValueError("Phase-8 manifest contract mismatch")
    for hash_field, path_field in (("cohort_manifest_sha256","cohort_manifest_path"),("structural_index_sha256","structural_index_path"),
            ("processed_manifest_sha256","processed_manifest_path"),("feature_manifest_sha256","feature_manifest_path"),
            ("canonical_timeline_sha256","canonical_timeline_path"),
            ("feature_schema_sha256","feature_schema_path"),("sofa_spec_sha256","sofa_spec_path"),
            ("recovery_spec_sha256","recovery_spec_path"),("icu_time_spec_sha256","icu_time_spec_path"),
            ("target_schema_sha256","target_schema_path"),("target_provenance_sha256","target_provenance_path"),
            ("builder_sha256","builder_path"),("recovery_engine_sha256","recovery_engine_path"),("icu_time_engine_sha256","icu_time_engine_path")):
        if sha256_file(root / manifest[path_field]) != manifest[hash_field]: raise ValueError(hash_field + " mismatch")
    artifacts = {x["logical_name"]:x for x in manifest["artifacts"]}
    for item in artifacts.values():
        if sha256_file(root/item["path"]) != item["sha256"]: raise ValueError("target artifact hash mismatch")
    rows = load_jsonl(root/artifacts["phase8_targets_partial"]["path"])
    if semantic_sha256("phase8_targets_partial",rows) != artifacts["phase8_targets_partial"]["semantic_sha256"]: raise ValueError("target semantic hash mismatch")
    seen=set()
    for row in rows:
        key=(row["stay_id"],row["prediction_time"])
        if key in seen: raise ValueError("duplicate target identity")
        seen.add(key)
        if row["recovery24_eligible"] is not None or row["recovery48_eligible"] is not None or row["delta_sofa_24"] is not None or row["delta_sofa_48"] is not None: raise ValueError("blocked recovery fields must remain null")
        if row["icu_time_eligible"] is not True or not math.isfinite(row["remaining_hours"]) or row["remaining_hours"] <= 0 or not math.isclose(math.expm1(row["icu_time_log1p"]),row["remaining_hours"],rel_tol=1e-12): raise ValueError("ICU-time target formula mismatch")
    if len(rows) != manifest["row_count"]: raise ValueError("target row count mismatch")
    return manifest
