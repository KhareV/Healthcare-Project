"""Produced dataset manifest creation and fail-closed verification."""
from __future__ import annotations
import json
from pathlib import Path
from .config import GENERATOR_VERSION, RuntimeConfig
from .provenance import sha256_file, semantic_sha256
from .validation import SyntheticValidationError, load_jsonl, validate_dataset

MANIFEST_NAME = "synthetic_dataset_manifest_v1.json"

def artifact_entry(root: Path, serialized_output: Path, published_output: Path, name: str, table: str, rows):
    path = serialized_output / f"{name}.jsonl"
    published_path = published_output / f"{name}.jsonl"
    return {"logical_name":name,"repository_relative_path":str(published_path.relative_to(root)),"schema_table":table,
            "serialization_format":"JSONL_CANONICAL_UTF8","record_count":len(rows),"sha256":sha256_file(path),
            "semantic_sha256":semantic_sha256(table, rows)}

def validate_manifest(path: Path, repo_root: Path, config: RuntimeConfig | None = None, schema_path: Path | None = None):
    try: manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error: raise SyntheticValidationError(f"manifest unreadable: {error}") from error
    if manifest.get("manifest_version") != "synthetic_dataset_manifest_v1": raise SyntheticValidationError("manifest version mismatch")
    if manifest.get("generator_version") != GENERATOR_VERSION: raise SyntheticValidationError("generator version mismatch")
    if manifest.get("manifest_status") not in {"FIXTURE", "SMOKE"}: raise SyntheticValidationError("Phase 3 manifest falsely claims final or unknown status")
    if manifest.get("scientific_support_generation_status") != "DEFERRED_TO_PHASE_9": raise SyntheticValidationError("support status inconsistent")
    if config and manifest.get("generator_config_sha256") != config.sha256: raise SyntheticValidationError("generator config hash mismatch")
    if schema_path and manifest.get("raw_schema_sha256") != sha256_file(schema_path): raise SyntheticValidationError("raw schema hash mismatch")
    records = {}
    for item in manifest.get("artifacts", []):
        artifact = repo_root / item["repository_relative_path"]
        if not artifact.is_file() or sha256_file(artifact) != item["sha256"]: raise SyntheticValidationError(f"artifact missing or hash mismatch: {item['logical_name']}")
        rows = load_jsonl(artifact)
        if len(rows) != item["record_count"]: raise SyntheticValidationError(f"record count mismatch: {item['logical_name']}")
        if semantic_sha256(item["schema_table"], rows) != item["semantic_sha256"]: raise SyntheticValidationError(f"semantic hash mismatch: {item['logical_name']}")
        records[item["logical_name"]] = rows
    required = {"subjects", "episodes", "raw_events", "support_intervals"}
    if set(records) != required: raise SyntheticValidationError("manifest artifact inventory incomplete")
    if manifest["record_counts"] != {k:len(records[k]) for k in sorted(required)}: raise SyntheticValidationError("manifest aggregate counts mismatch")
    if manifest["generated_subject_count"] != len(records["subjects"]): raise SyntheticValidationError("generated subject count mismatch")
    if schema_path:
        import yaml
        schema = json.loads(schema_path.read_text())
        spec = yaml.safe_load((repo_root / "configs/synthetic/synthetic_generator_v1.yaml").read_text())
        validate_dataset(records["subjects"],records["episodes"],records["raw_events"],records["support_intervals"],schema,spec["raw_variable_inventory"])
    return manifest
