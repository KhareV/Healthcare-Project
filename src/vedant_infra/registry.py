"""Experiment-registry schema validation with no experiment execution."""

import csv
import json
from datetime import datetime
from pathlib import Path, PureWindowsPath
from typing import Dict, Iterable, Mapping, Union

from .hashing import is_sha256, sha256_file


PathLike = Union[str, Path]

REGISTRY_COLUMNS = (
    "run_id",
    "timestamp_utc",
    "task",
    "model_family",
    "seed",
    "code_commit",
    "config_ref",
    "config_hash",
    "search_space_hash",
    "split_hash",
    "feature_version",
    "label_version",
    "model_artifact_ref",
    "model_sha256",
    "metrics_ref",
    "status",
    "parent_run_id",
    "notes",
    "candidate_id",
    "search_version",
    "candidate_list_hash",
    "attempt_number",
    "retry_of_run_id",
    "attempt_status_detail",
    "validation_objective",
    "run_type",
    "finalized",
    "metrics_sha256",
    "preprocessor_ref",
    "preprocessor_sha256",
    "event_dict_version",
    "parent_feature_version",
    "derived_feature_hash",
    "environment_ref",
    "environment_sha256",
    "dirty_worktree",
    "dataset_version",
    "mimic_code_commit",
    "extraction_ref",
    "extraction_sha256",
    "cohort_version",
    "feature_dictionary_ref",
    "feature_dictionary_sha256",
    "label_spec_ref",
    "split_ref",
)

COMPLETED_REQUIRED_FIELDS = (
    "run_id",
    "timestamp_utc",
    "task",
    "model_family",
    "seed",
    "code_commit",
    "config_ref",
    "config_hash",
    "split_hash",
    "feature_version",
    "label_version",
    "model_artifact_ref",
    "model_sha256",
    "metrics_ref",
)

HASH_FIELDS = (
    "config_hash",
    "search_space_hash",
    "split_hash",
    "model_sha256",
    "candidate_list_hash",
    "metrics_sha256",
    "preprocessor_sha256",
    "derived_feature_hash",
    "environment_sha256",
    "extraction_sha256",
    "feature_dictionary_sha256",
)

REFERENCE_FIELDS = (
    "config_ref",
    "model_artifact_ref",
    "metrics_ref",
    "preprocessor_ref",
    "environment_ref",
    "extraction_ref",
    "feature_dictionary_ref",
    "label_spec_ref",
    "split_ref",
)

ALLOWED_STATUSES = {"planned", "running", "completed", "failed", "aborted"}
ALLOWED_RUN_TYPES = {
    "scientific",
    "synthetic",
    "smoke",
    "development",
    "test",
    "sensitivity_smoke",
    "legacy_incomplete",
}
FINAL_STATUSES = {"completed", "failed", "aborted"}
IMMUTABLE_FIELDS = (
    "run_id",
    "task",
    "model_family",
    "seed",
    "code_commit",
    "config_ref",
    "config_hash",
    "search_space_hash",
    "split_hash",
    "feature_version",
    "label_version",
    "candidate_id",
    "run_type",
)
PROHIBITED_METADATA_KEYS = {
    "password",
    "api_token",
    "access_token",
    "secret",
    "raw_clinical_note",
    "patient_name",
}


class RegistryValidationError(ValueError):
    """Raised when the registry violates its Phase 0 schema."""


def _is_absolute_any_platform(value: str) -> bool:
    return Path(value).is_absolute() or PureWindowsPath(value).is_absolute()


def _validate_header(fieldnames: Iterable[str]) -> None:
    actual = tuple(fieldnames)
    missing = [field for field in REGISTRY_COLUMNS if field not in actual]
    if missing:
        raise RegistryValidationError(
            "registry is missing mandatory columns: " + ", ".join(missing)
        )


def _validate_record(record: Mapping[str, str], line_number: int) -> None:
    status = (record.get("status") or "").strip()
    if status and status not in ALLOWED_STATUSES:
        raise RegistryValidationError(
            "line {} has unsupported status: {}".format(line_number, status)
        )

    if status == "completed":
        missing = [
            field
            for field in COMPLETED_REQUIRED_FIELDS
            if not (record.get(field) or "").strip()
        ]
        if missing:
            raise RegistryValidationError(
                "line {} completed record is missing: {}".format(
                    line_number, ", ".join(missing)
                )
            )

    run_type = (record.get("run_type") or "").strip()
    if run_type and run_type not in ALLOWED_RUN_TYPES:
        raise RegistryValidationError(
            "line {} has unsupported run_type: {}".format(line_number, run_type)
        )
    if run_type == "scientific":
        required = (
            "timestamp_utc",
            "task",
            "model_family",
            "seed",
            "code_commit",
            "config_ref",
            "config_hash",
            "split_hash",
            "feature_version",
            "label_version",
        )
        missing = [field for field in required if not (record.get(field) or "").strip()]
        if missing:
            raise RegistryValidationError(
                "line {} scientific run is missing: {}".format(
                    line_number, ", ".join(missing)
                )
            )
        if record.get("code_commit", "").startswith("UNAVAILABLE"):
            raise RegistryValidationError("scientific run requires an exact code commit")
        commit = record.get("code_commit", "")
        if len(commit) not in (40, 64) or any(character not in "0123456789abcdef" for character in commit):
            raise RegistryValidationError("scientific run code_commit must be an exact Git hash")
    timestamp = (record.get("timestamp_utc") or "").strip()
    if timestamp:
        try:
            parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        except ValueError as error:
            raise RegistryValidationError("registry timestamp must be ISO-8601") from error
        if parsed.tzinfo is None:
            raise RegistryValidationError("registry timestamp must include timezone")

    for field in HASH_FIELDS:
        value = (record.get(field) or "").strip()
        if value and not is_sha256(value):
            raise RegistryValidationError(
                "line {} has invalid {}".format(line_number, field)
            )

    for field in REFERENCE_FIELDS:
        value = (record.get(field) or "").strip()
        if value and _is_absolute_any_platform(value):
            raise RegistryValidationError(
                "line {} {} must be repository-relative".format(
                    line_number, field
                )
            )
    searchable = " ".join(str(value).lower() for value in record.values())
    for key in PROHIBITED_METADATA_KEYS:
        if key in searchable:
            raise RegistryValidationError("registry metadata contains prohibited field: " + key)


def validate_registry(path: PathLike) -> int:
    """Validate a registry and return its number of data records.

    A header-only registry is valid. No file content is mutated.
    """

    registry_path = Path(path)
    with registry_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise RegistryValidationError("registry has no header")
        _validate_header(reader.fieldnames)
        count = 0
        rows = []
        for line_number, record in enumerate(reader, start=2):
            _validate_record(record, line_number)
            rows.append(record)
            count += 1
    ids = [row.get("run_id", "") for row in rows]
    if len(ids) != len(set(ids)):
        raise RegistryValidationError("registry contains duplicate run_id")
    known = set(ids)
    for row in rows:
        for field in ("parent_run_id", "retry_of_run_id"):
            parent = (row.get(field) or "").strip()
            if parent and parent not in known:
                raise RegistryValidationError(
                    "run {} references unknown {} {}".format(
                        row.get("run_id"), field, parent
                    )
                )
        retry = (row.get("retry_of_run_id") or "").strip()
        if retry:
            parent = next(item for item in rows if item.get("run_id") == retry)
            if parent.get("candidate_id") != row.get("candidate_id"):
                raise RegistryValidationError("retry candidate_id differs from failed attempt")
            if parent.get("status") != "failed":
                raise RegistryValidationError("retry must reference a failed run")
    return count


def make_registry_record(**values: str) -> Dict[str, str]:
    """Build an ordered blank record for tests/future callers."""

    normalized = {field: values.get(field, "") for field in REGISTRY_COLUMNS}
    if not normalized["run_type"]:
        combined = " ".join(
            str(values.get(field, "")) for field in ("task", "model_family", "notes")
        ).lower()
        normalized["run_type"] = "synthetic" if "synthetic" in combined else "development"
    if not normalized["finalized"] and normalized["status"] in FINAL_STATUSES:
        normalized["finalized"] = "true"
    return normalized


def upsert_registry_record(path: PathLike, record: Mapping[str, str]) -> None:
    """Validate and insert/replace one run by run_id deterministically."""

    registry_path = Path(path)
    normalized = make_registry_record(**dict(record))
    if not normalized["run_id"].strip():
        raise RegistryValidationError("registry record requires run_id")
    _validate_record(normalized, line_number=2)

    existing = []  # type: list
    if registry_path.exists():
        with registry_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise RegistryValidationError("registry has no header")
            _validate_header(reader.fieldnames)
            existing = [dict(row) for row in reader]

    matches = [row for row in existing if row.get("run_id") == normalized["run_id"]]
    if matches:
        prior = make_registry_record(**matches[0])
        if prior == normalized:
            return
        if prior.get("finalized") == "true" or prior.get("status") in FINAL_STATUSES:
            raise RegistryValidationError("finalized run records are immutable")
        changed = [field for field in IMMUTABLE_FIELDS if prior.get(field) != normalized.get(field)]
        if changed:
            raise RegistryValidationError(
                "run lifecycle update changed immutable fields: " + ", ".join(changed)
            )
    retained = [row for row in existing if row.get("run_id") != normalized["run_id"]]
    retained.append(normalized)
    retained.sort(key=lambda row: row["run_id"])
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = registry_path.with_name(registry_path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REGISTRY_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(retained)
    temporary.replace(registry_path)
    validate_registry(registry_path)


def migrate_registry_schema(path: PathLike, repository_root: PathLike) -> int:
    """Add Phase-16 columns and only recover metadata verifiable from files."""

    registry_path = Path(path)
    root = Path(repository_root)
    with registry_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise RegistryValidationError("registry has no header")
        rows = [dict(row) for row in reader]
    migrated = []
    for row in rows:
        values = dict(row)
        if not values.get("run_type"):
            notes = values.get("notes", "").lower()
            values["run_type"] = (
                "sensitivity_smoke" if "sensitivity_smoke" in notes else "synthetic"
            )
        if not values.get("finalized") and values.get("status") in FINAL_STATUSES:
            values["finalized"] = "true"
        metrics_ref = values.get("metrics_ref", "")
        metrics_path = root / metrics_ref if metrics_ref else None
        if metrics_path is not None and metrics_path.is_file():
            values["metrics_sha256"] = sha256_file(metrics_path)
        model_ref = values.get("model_artifact_ref", "")
        sidecar = root / (model_ref + ".metadata.json") if model_ref else None
        if sidecar is not None and sidecar.is_file():
            metadata = json.loads(sidecar.read_text(encoding="utf-8"))
            preprocessing = metadata.get("preprocessing") or {}
            preprocessor_ref = (
                metadata.get("target_scaler_artifact")
                or metadata.get("class_weight_artifact")
                or preprocessing.get("target_scaler_ref")
                or preprocessing.get("class_weight_ref")
            )
            preprocessor_hash = (
                metadata.get("target_scaler_sha256")
                or metadata.get("class_weight_sha256")
                or preprocessing.get("target_scaler_sha256")
                or preprocessing.get("class_weight_sha256")
            )
            if preprocessor_ref and (root / preprocessor_ref).is_file():
                actual = sha256_file(root / preprocessor_ref)
                if actual != preprocessor_hash:
                    raise RegistryValidationError("historical preprocessor hash mismatch")
                values["preprocessor_ref"] = preprocessor_ref
                values["preprocessor_sha256"] = actual
            if metadata.get("event_dictionary_version"):
                values["event_dict_version"] = metadata["event_dictionary_version"]
        migrated.append(make_registry_record(**values))
    temporary = registry_path.with_name(registry_path.name + ".phase16.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REGISTRY_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(migrated)
    temporary.replace(registry_path)
    return validate_registry(registry_path)
