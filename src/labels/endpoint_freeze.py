"""Endpoint-freeze prerequisite auditing and immutable freeze records.

Real freeze activation is intentionally fail-closed. Synthetic freeze records
exercise ordering and hash binding but never authorize real label access.
"""

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Optional, Tuple

import yaml

from labels.event_dict import (
    FROZEN_STATUS,
    ORGAN_SUPPORT_FROZEN_STATUS,
    VENTILATION_FROZEN_STATUS,
    load_event_dictionary,
)
from vedant_infra.hashing import is_sha256, sha256_file


FREEZE_RECORD_VERSION = "support_endpoint_freeze_record_v1"
ENDPOINT_VERSION = "organ_support_endpoint_v1"
ENDPOINT_NAME = "24-hour New Organ-Support Initiation Risk"
ACTIVE_REAL_STATUS = "ACTIVE_REAL_APPROVED"
ACTIVE_SYNTHETIC_STATUS = "ACTIVE_SYNTHETIC_NON_AUTHORIZING"
BLOCKED_STATUS = "BLOCKED_PREREQUISITES"


class EndpointFreezeError(RuntimeError):
    """Raised when endpoint freeze or freeze-bound access is unsafe."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _load_json(path: Path) -> Mapping[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise EndpointFreezeError("unreadable JSON artifact: " + str(path)) from error
    if not isinstance(value, dict):
        raise EndpointFreezeError("JSON artifact must be an object: " + str(path))
    return value


def _safe_ref(root: Path, reference: object) -> Path:
    if not isinstance(reference, str) or not reference or Path(reference).is_absolute():
        raise EndpointFreezeError("artifact reference must be repository-relative")
    resolved_root = root.resolve()
    path = (root / reference).resolve()
    if resolved_root != path and resolved_root not in path.parents:
        raise EndpointFreezeError("artifact reference escapes repository root")
    return path


def _git_commit(root: Path) -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(root),
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    commit = result.stdout.strip()
    if len(commit) not in (40, 64) or any(
        character not in "0123456789abcdef" for character in commit
    ):
        return None
    return commit


def _audit_item(name: str, passed: bool, detail: str) -> Mapping[str, object]:
    return {"name": name, "status": "PASS" if passed else "BLOCKED", "detail": detail}


def audit_real_endpoint_freeze(
    root: Path,
    *,
    signoff_ref: str = "configs/support_endpoint_signoff_v1.json",
) -> Mapping[str, object]:
    """Audit real-freeze prerequisites without reading label/prevalence data."""

    root = root.resolve()
    event_ref = "configs/event_dict_v1.yaml"
    event_path = _safe_ref(root, event_ref)
    signoff_path = _safe_ref(root, signoff_ref)
    endpoint_ref = "src/labels/organ_support.py"
    timestamp_ref = "configs/timestamp_spec_v1.yaml"
    event_hash = sha256_file(event_path)
    items = []

    try:
        contract = load_event_dictionary(event_path)
        items.append(_audit_item("event_dictionary_parse", True, "validated"))
    except Exception as error:
        contract = None
        items.append(_audit_item("event_dictionary_parse", False, str(error)))

    raw = yaml.safe_load(event_path.read_text(encoding="utf-8"))
    unresolved = raw.get("unresolved_required_items") if isinstance(raw, dict) else None
    items.append(
        _audit_item(
            "event_dictionary_global_freeze",
            bool(
                contract
                and contract.status == FROZEN_STATUS
                and contract.real_scientific_execution_authorized
                and unresolved == []
            ),
            "requires frozen global status, authorization, and no unresolved items",
        )
    )
    items.append(
        _audit_item(
            "phase1_vasopressor_real_definition",
            bool(
                contract
                and contract.is_frozen_for_scientific_use
                and contract.scientific_qualifying_agent_keys
                and contract.scientific_boundary_policy
                and contract.mapping_provenance_version
            ),
            "requires verified qualifying mapping/activity/boundary provenance",
        )
    )
    items.append(
        _audit_item(
            "phase2_ventilation_real_definition",
            bool(
                contract
                and contract.ventilation.status == VENTILATION_FROZEN_STATUS
                and contract.ventilation.is_frozen_for_scientific_use
            ),
            "requires pinned reviewed MIMIC-Code concept and SOFA alignment",
        )
    )
    items.append(
        _audit_item(
            "phase3_composite_real_definition",
            bool(
                contract
                and contract.organ_support.status == ORGAN_SUPPORT_FROZEN_STATUS
                and contract.organ_support.real_scientific_execution_authorized
            ),
            "requires resolved tie/unknown/vocabulary policies and reviews",
        )
    )
    items.append(
        _audit_item(
            "timestamp_contract",
            timestamp_ref == "configs/timestamp_spec_v1.yaml"
            and _safe_ref(root, timestamp_ref).is_file(),
            "timestamp_spec_v1 must be present and hash-bound",
        )
    )

    try:
        signoff = _load_json(signoff_path)
    except EndpointFreezeError as error:
        signoff = {}
        items.append(_audit_item("team_signoff_schema", False, str(error)))
    else:
        schema_ok = (
            signoff.get("schema_version")
            == "support_endpoint_signoff_schema_v1_proposed"
            and signoff.get("event_dictionary_sha256") == event_hash
        )
        items.append(
            _audit_item(
                "team_signoff_schema",
                schema_ok,
                "proposed schema and exact event-dictionary hash must match",
            )
        )
    reviewers = signoff.get("reviewers") if isinstance(signoff, dict) else None
    required_reviewers = ("Sanskruti", "Vedant", "Pulkit_team_approval")
    approvals_ok = isinstance(reviewers, dict)
    if approvals_ok:
        for reviewer in required_reviewers:
            review = reviewers.get(reviewer)
            approvals_ok = bool(
                approvals_ok
                and isinstance(review, dict)
                and review.get("decision") == "APPROVED"
                and isinstance(review.get("review_timestamp_utc"), str)
                and review.get("evidence_ref")
                and review.get("reviewed_event_dictionary_sha256") == event_hash
            )
    items.append(
        _audit_item(
            "team_endpoint_signoff",
            approvals_ok,
            "genuine Sanskruti, Vedant, and Pulkit-after-team-agreement approvals required",
        )
    )
    test_ref = signoff.get("state_machine_test_evidence_ref") if signoff else None
    test_hash = signoff.get("state_machine_test_evidence_sha256") if signoff else None
    test_ok = False
    if isinstance(test_ref, str) and isinstance(test_hash, str) and is_sha256(test_hash):
        test_path = _safe_ref(root, test_ref)
        test_ok = test_path.is_file() and sha256_file(test_path) == test_hash
    items.append(
        _audit_item(
            "state_machine_test_evidence",
            test_ok,
            "reviewed Phase-3 test evidence must be hash-bound",
        )
    )
    items.append(
        _audit_item(
            "prevalence_noninspection_attestation",
            signoff.get("real_prevalence_pre_freeze_attestation")
            == "ATTESTED_NOT_ACCESSED",
            "human attestation required before real freeze",
        )
    )
    commit = _git_commit(root)
    items.append(
        _audit_item(
            "repository_commit_provenance",
            commit is not None,
            "exact immutable Git commit required for formal real freeze",
        )
    )
    blockers = tuple(
        str(item["name"]) for item in items if item["status"] == "BLOCKED"
    )
    return {
        "audit_version": "support_endpoint_freeze_audit_v1",
        "endpoint_version": ENDPOINT_VERSION,
        "endpoint_name": ENDPOINT_NAME,
        "scope": "real",
        "generated_at_utc": utc_now(),
        "event_dictionary_ref": event_ref,
        "event_dictionary_sha256": event_hash,
        "endpoint_implementation_ref": endpoint_ref,
        "endpoint_implementation_sha256": sha256_file(_safe_ref(root, endpoint_ref)),
        "timestamp_spec_ref": timestamp_ref,
        "timestamp_spec_sha256": sha256_file(_safe_ref(root, timestamp_ref)),
        "signoff_ref": signoff_ref,
        "signoff_sha256": sha256_file(signoff_path),
        "code_commit": commit,
        "items": items,
        "blockers": blockers,
        "overall": "PASS" if not blockers else "BLOCKED",
        "label_or_prevalence_data_accessed": False,
    }


def _write_immutable_json(path: Path, payload: Mapping[str, object]) -> str:
    content = (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    ).encode("utf-8")
    if path.exists():
        if path.read_bytes() != content:
            raise EndpointFreezeError("immutable artifact path already contains different bytes")
        return sha256_file(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(content)
    temporary.replace(path)
    return sha256_file(path)


def write_blocked_real_freeze_audit(
    root: Path,
    output_path: Path,
) -> str:
    audit = audit_real_endpoint_freeze(root)
    record = {
        "record_version": FREEZE_RECORD_VERSION,
        "freeze_id": "support_endpoint_real_v1_blocked_audit",
        "freeze_scope": "real",
        "freeze_status": BLOCKED_STATUS,
        "freeze_active": False,
        "freeze_timestamp_utc": audit["generated_at_utc"],
        "endpoint_version": ENDPOINT_VERSION,
        "endpoint_name": ENDPOINT_NAME,
        "audit": audit,
        "prevalence_access_authorized": False,
        "test_label_access_authorized": False,
    }
    return _write_immutable_json(output_path, record)


def create_real_endpoint_freeze(root: Path, output_path: Path) -> str:
    """Create an active real freeze only after every audited prerequisite passes."""

    audit = audit_real_endpoint_freeze(root)
    if audit["overall"] != "PASS":
        raise EndpointFreezeError(
            "ENDPOINT FREEZE = BLOCKED: " + ", ".join(audit["blockers"])
        )
    record = {
        "record_version": FREEZE_RECORD_VERSION,
        "freeze_id": "support_endpoint_real_v1",
        "freeze_scope": "real",
        "freeze_status": ACTIVE_REAL_STATUS,
        "freeze_active": True,
        "freeze_timestamp_utc": utc_now(),
        "endpoint_version": ENDPOINT_VERSION,
        "endpoint_name": ENDPOINT_NAME,
        "event_dictionary_ref": audit["event_dictionary_ref"],
        "event_dictionary_sha256": audit["event_dictionary_sha256"],
        "endpoint_implementation_ref": audit["endpoint_implementation_ref"],
        "endpoint_implementation_sha256": audit["endpoint_implementation_sha256"],
        "timestamp_spec_ref": audit["timestamp_spec_ref"],
        "timestamp_spec_sha256": audit["timestamp_spec_sha256"],
        "signoff_ref": audit["signoff_ref"],
        "signoff_sha256": audit["signoff_sha256"],
        "code_commit": audit["code_commit"],
        "prevalence_access_authorized": True,
        "test_label_access_authorized": False,
        "audit": audit,
    }
    return _write_immutable_json(output_path, record)


def create_synthetic_endpoint_freeze(root: Path, output_path: Path) -> str:
    """Create an active but non-authorizing synthetic ordering fixture."""

    root = root.resolve()
    event_ref = "configs/event_dict_v1.yaml"
    implementation_ref = "src/labels/organ_support.py"
    timestamp_ref = "configs/timestamp_spec_v1.yaml"
    record = {
        "record_version": FREEZE_RECORD_VERSION,
        "freeze_id": "support_endpoint_synthetic_v1",
        "freeze_scope": "synthetic",
        "freeze_status": ACTIVE_SYNTHETIC_STATUS,
        "freeze_active": True,
        "freeze_timestamp_utc": utc_now(),
        "endpoint_version": ENDPOINT_VERSION,
        "endpoint_name": ENDPOINT_NAME,
        "event_dictionary_ref": event_ref,
        "event_dictionary_sha256": sha256_file(_safe_ref(root, event_ref)),
        "event_dictionary_semantics_status": "DRAFT_SYNTHETIC_ONLY",
        "endpoint_implementation_ref": implementation_ref,
        "endpoint_implementation_sha256": sha256_file(
            _safe_ref(root, implementation_ref)
        ),
        "timestamp_spec_ref": timestamp_ref,
        "timestamp_spec_sha256": sha256_file(_safe_ref(root, timestamp_ref)),
        "code_commit": _git_commit(root),
        "commit_provenance_status": "BLOCKED_FOR_REAL_FREEZE",
        "review_status": "SYNTHETIC_FIXTURE_ONLY_NO_HUMAN_APPROVAL",
        "prevalence_access_authorized": True,
        "real_data_access_authorized": False,
        "test_label_access_authorized": False,
    }
    return _write_immutable_json(output_path, record)


def validate_freeze_record(
    root: Path,
    freeze_path: Path,
    *,
    expected_sha256: str,
    requested_scope: str,
) -> Mapping[str, object]:
    """Validate exact freeze bytes and current dependency hashes."""

    if not is_sha256(expected_sha256) or not freeze_path.is_file():
        raise EndpointFreezeError("freeze artifact/hash is missing or malformed")
    if sha256_file(freeze_path) != expected_sha256:
        raise EndpointFreezeError("freeze artifact hash mismatch")
    record = _load_json(freeze_path)
    if record.get("record_version") != FREEZE_RECORD_VERSION:
        raise EndpointFreezeError("unsupported freeze-record version")
    if record.get("freeze_active") is not True:
        raise EndpointFreezeError("endpoint freeze is not active")
    if record.get("freeze_scope") != requested_scope:
        raise EndpointFreezeError("freeze scope cannot authorize requested data")
    expected_status = (
        ACTIVE_REAL_STATUS if requested_scope == "real" else ACTIVE_SYNTHETIC_STATUS
    )
    if record.get("freeze_status") != expected_status:
        raise EndpointFreezeError("freeze status cannot authorize requested data")
    if requested_scope == "real" and record.get("prevalence_access_authorized") is not True:
        raise EndpointFreezeError("real prevalence access is not authorized")
    for ref_field, hash_field in (
        ("event_dictionary_ref", "event_dictionary_sha256"),
        ("endpoint_implementation_ref", "endpoint_implementation_sha256"),
        ("timestamp_spec_ref", "timestamp_spec_sha256"),
    ):
        path = _safe_ref(root, record.get(ref_field))
        expected = record.get(hash_field)
        if not isinstance(expected, str) or not is_sha256(expected):
            raise EndpointFreezeError(hash_field + " is malformed")
        if not path.is_file() or sha256_file(path) != expected:
            raise EndpointFreezeError(ref_field + " changed after endpoint freeze")
    if requested_scope == "real":
        commit = record.get("code_commit")
        if commit is None or _git_commit(root) != commit:
            raise EndpointFreezeError("real freeze commit provenance is unavailable or stale")
    return record

