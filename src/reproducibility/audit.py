"""Environment, artifact, portability, and restricted-data audits."""

import json
import re
from pathlib import Path
from typing import Mapping, Sequence

from experiments.lineage import (
    read_artifact_index,
    read_run_registry,
    validate_artifact_lineage,
)
from vedant_infra.hashing import is_sha256, sha256_file


class ReproducibilityAuditError(RuntimeError):
    """Raised for malformed reproducibility configuration or evidence."""


EXACT_VERSION_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+==[^*xX,;<>=!~\s]+$")
ABSOLUTE_PATH_PATTERNS = (
    re.compile(r"/Users/[^/\s]+/"),
    re.compile(r"/home/[^/\s]+/"),
    re.compile(r"[A-Za-z]:\\Users\\[^\\\s]+\\"),
)
FORBIDDEN_OPERATIONS = {
    "retune-after-test",
    "choose-better-final-model",
    "recompute-threshold-on-test",
    "fit-calibrator-on-test",
}


def load_config(path: Path) -> Mapping[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ReproducibilityAuditError("unreadable reproducibility config") from error
    if not isinstance(value, Mapping) or value.get("config_version") != "reproducibility_v1":
        raise ReproducibilityAuditError("reproducibility config version mismatch")
    return value


def audit_environment_lock(root: Path, config: Mapping[str, object]):
    reference = config.get("environment_lock_ref")
    expected = config.get("environment_lock_sha256")
    if not isinstance(reference, str) or not reference:
        return {
            "status": "BLOCKED",
            "detail": "BLOCKED — EXACT ENVIRONMENT LOCK REQUIRED FOR FINAL REPRODUCIBILITY CLAIM",
            "lock_ref": None,
            "lock_sha256": None,
        }
    path = (root / reference).resolve()
    if root.resolve() not in path.parents or not path.is_file():
        return {"status": "BLOCKED", "detail": "environment lock is missing"}
    actual = sha256_file(path)
    if not isinstance(expected, str) or not is_sha256(expected) or actual != expected:
        return {"status": "BLOCKED", "detail": "environment lock hash mismatch"}
    lines = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    invalid = tuple(line for line in lines if not EXACT_VERSION_PATTERN.match(line))
    return {
        "status": "PASS" if not invalid else "BLOCKED",
        "detail": "exact versions verified" if not invalid else "wildcard or non-exact dependency entries",
        "lock_ref": reference,
        "lock_sha256": actual,
        "invalid_entries": invalid,
    }


def audit_registered_artifacts(root: Path):
    records = tuple(
        record
        for record in read_artifact_index(root / "experiments/artifacts.csv")
        if not record.artifact_id.startswith("phase21:")
    )
    failures = {}
    try:
        validate_artifact_lineage(
            records,
            read_run_registry(root / "experiments/registry.csv"),
            repository_root=root,
        )
    except Exception as error:
        failures["lineage_validation"] = str(error)
    return {
        "status": "PASS" if not failures else "BLOCKED",
        "registered_artifact_count": len(records),
        "failures": failures,
    }


def audit_path_portability(root: Path):
    operational_roots = (root / "src", root / "configs")
    findings = []
    for base in operational_roots:
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if not path.is_file() or path.suffix in {".pyc", ".pt"}:
                continue
            if path.resolve() == Path(__file__).resolve():
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            if any(pattern.search(text) for pattern in ABSOLUTE_PATH_PATTERNS):
                findings.append(str(path.relative_to(root)))
    return {
        "status": "PASS" if not findings else "BLOCKED",
        "operational_absolute_path_findings": tuple(sorted(findings)),
        "historical_evidence_paths_scanned": False,
        "detail": "immutable historical reports may contain diagnostic absolute paths but are not execution inputs",
    }


def audit_restricted_data(root: Path):
    prohibited_types = {
        "raw_mimic",
        "raw_clinical_table",
        "patient_identifying_extract",
        "credential",
    }
    records = read_artifact_index(root / "experiments/artifacts.csv")
    indexed = tuple(
        record.artifact_id
        for record in records
        if record.artifact_type.lower() in prohibited_types
    )
    sensitive_keys = re.compile(
        r'"(?:password|api_token|access_token|secret)"\s*:\s*"(?!fixture|synthetic|redacted|<)[^"\s]+"',
        re.IGNORECASE,
    )
    content_findings = []
    for base in (root / "configs", root / "artifacts"):
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if not path.is_file() or path.suffix not in {".json", ".yaml", ".yml", ".csv", ".jsonl"}:
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            if sensitive_keys.search(text):
                content_findings.append(str(path.relative_to(root)))
    findings = tuple(sorted(indexed + tuple(content_findings)))
    return {
        "status": "PASS" if not findings else "BLOCKED",
        "prohibited_index_or_secret_findings": findings,
        "audit_scope": "artifact classes plus structured config/artifact secret values",
    }


def reject_post_test_operation(operation: str) -> None:
    if operation in FORBIDDEN_OPERATIONS:
        raise ReproducibilityAuditError("post-test scientific tuning is forbidden")


def require_cross_member_evidence(payload: Mapping[str, object]) -> None:
    if payload.get("reproducer_role") in (None, "vedant", "primary_owner"):
        raise ReproducibilityAuditError("cross-member reproduction requires a non-owner")
    required = (
        "reproducer_name",
        "run_id",
        "config_hash",
        "split_hash",
        "environment_lock_sha256",
        "command",
        "expected_metrics",
        "reproduced_metrics",
    )
    if any(payload.get(field) in (None, "", {}) for field in required):
        raise ReproducibilityAuditError("cross-member reproduction evidence is incomplete")
