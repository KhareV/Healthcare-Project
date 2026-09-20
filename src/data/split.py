"""Authoritative subject-level coarse temporal split implementation.

The module consumes an already-retained cohort contract. It does not query
MIMIC, select ICU stays, inspect labels, fit preprocessing, or train models.
"""

import csv
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple, Union

from vedant_infra.hashing import is_sha256, sha256_bytes, sha256_file


Identifier = Union[int, str]
PathLike = Union[str, Path]

SPLIT_SPEC_VERSION = "split_spec_v1"
SPLIT_ARTIFACT_VERSION = "split_v1"
SUPPORTED_ERA_TO_SPLIT = {
    "2008-2010": "train",
    "2011-2013": "train",
    "2014-2016": "validation",
    "2017-2019": "test",
}
SPLIT_NAMES = ("train", "validation", "test")
CSV_COLUMNS = ("subject_id", "split", "anchor_year_group")


class SplitValidationError(ValueError):
    """Raised when cohort or split content violates the frozen contract."""


class SplitIntegrityError(RuntimeError):
    """Raised when a serialized split fails integrity verification."""


@dataclass(frozen=True)
class CohortSubjectRecord:
    """Internal retained-cohort input; names are engineering contract fields."""

    subject_id: Identifier
    anchor_year_group: str
    stay_id: Optional[Identifier] = None


@dataclass(frozen=True)
class SubjectSplit:
    """One canonical subject-to-split assignment."""

    subject_id: Identifier
    split: str
    anchor_year_group: str


@dataclass(frozen=True)
class SplitQASummary:
    """Structural QA only; no labels, prevalence, or performance metrics."""

    total_input_rows: int
    total_subjects: int
    train_subjects: int
    validation_subjects: int
    test_subjects: int
    counts_by_anchor_year_group: Mapping[str, int]
    duplicate_input_count: int
    missing_subject_count: int
    missing_era_count: int
    unsupported_era_count: int
    conflicting_subject_count: int


@dataclass(frozen=True)
class CohortAudit:
    """Validated/deduplicated cohort state plus any contract errors."""

    canonical_records: Tuple[CohortSubjectRecord, ...]
    qa: SplitQASummary
    errors: Tuple[str, ...]

    @property
    def is_valid(self) -> bool:
        return not self.errors


@dataclass(frozen=True)
class SplitResult:
    """Canonical split rows and structural QA summary."""

    rows: Tuple[SubjectSplit, ...]
    qa: SplitQASummary
    split_spec_version: str = SPLIT_SPEC_VERSION


def _is_supported_identifier(value: object) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return True
    return isinstance(value, str) and bool(value.strip())


def _identifier_key(value: Identifier) -> Tuple[int, Union[int, str]]:
    if isinstance(value, int):
        return (0, value)
    return (1, value)


def era_to_split(anchor_year_group: str) -> str:
    """Map one exact frozen era string to its split."""

    if anchor_year_group not in SUPPORTED_ERA_TO_SPLIT:
        raise SplitValidationError(
            "unsupported anchor_year_group: {!r}".format(anchor_year_group)
        )
    return SUPPORTED_ERA_TO_SPLIT[anchor_year_group]


def records_from_mappings(
    rows: Iterable[Mapping[str, object]],
) -> Tuple[CohortSubjectRecord, ...]:
    """Extract only split-contract fields, intentionally ignoring extras.

    Extra label, outcome, timestamp, missingness, or model fields cannot affect
    assignment because they are never copied into the typed contract.
    """

    records = []  # type: List[CohortSubjectRecord]
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise SplitValidationError("input row {} must be a mapping".format(index))
        records.append(
            CohortSubjectRecord(
                subject_id=row.get("subject_id"),  # type: ignore[arg-type]
                anchor_year_group=row.get("anchor_year_group"),  # type: ignore[arg-type]
                stay_id=row.get("stay_id"),  # type: ignore[arg-type]
            )
        )
    return tuple(records)


def audit_cohort_records(records: Iterable[CohortSubjectRecord]) -> CohortAudit:
    """Audit and deterministically deduplicate subject-level cohort metadata."""

    materialized = tuple(records)
    errors = []  # type: List[str]
    missing_subject_count = 0
    missing_era_count = 0
    unsupported_era_count = 0
    duplicate_input_count = 0
    conflicting_subjects = set()  # type: Set[Tuple[type, Identifier]]
    identifier_types = set()  # type: Set[type]

    by_subject = {}  # type: Dict[Tuple[type, Identifier], CohortSubjectRecord]
    stay_ids = {}  # type: Dict[Tuple[type, Identifier], Set[Tuple[type, Identifier]]]

    for index, record in enumerate(materialized):
        if not isinstance(record, CohortSubjectRecord):
            errors.append("row {} is not CohortSubjectRecord".format(index))
            continue

        subject_valid = _is_supported_identifier(record.subject_id)
        if not subject_valid:
            missing_subject_count += 1
            errors.append("row {} has missing or invalid subject_id".format(index))

        era_valid = isinstance(record.anchor_year_group, str) and bool(
            record.anchor_year_group
        )
        if not era_valid:
            missing_era_count += 1
            errors.append("row {} has missing anchor_year_group".format(index))
        elif record.anchor_year_group not in SUPPORTED_ERA_TO_SPLIT:
            unsupported_era_count += 1
            errors.append(
                "row {} has unsupported anchor_year_group {!r}".format(
                    index, record.anchor_year_group
                )
            )
            era_valid = False

        stay_valid = record.stay_id is None or _is_supported_identifier(record.stay_id)
        if not stay_valid:
            errors.append("row {} has invalid stay_id".format(index))

        if not (subject_valid and era_valid and stay_valid):
            continue

        identifier_types.add(type(record.subject_id))
        subject_key = (type(record.subject_id), record.subject_id)
        existing = by_subject.get(subject_key)
        if existing is None:
            by_subject[subject_key] = record
            stay_ids[subject_key] = set()
        else:
            duplicate_input_count += 1
            if existing.anchor_year_group != record.anchor_year_group:
                conflicting_subjects.add(subject_key)

        if record.stay_id is not None:
            stay_ids[subject_key].add((type(record.stay_id), record.stay_id))
            if len(stay_ids[subject_key]) > 1:
                conflicting_subjects.add(subject_key)

    if len(identifier_types) > 1:
        errors.append("mixed subject_id types are not allowed in one split artifact")

    for subject_key in sorted(
        conflicting_subjects,
        key=lambda key: _identifier_key(key[1]),
    ):
        errors.append(
            "subject {!r} has conflicting era or retained-stay metadata".format(
                subject_key[1]
            )
        )

    canonical_records = tuple(
        sorted(by_subject.values(), key=lambda record: _identifier_key(record.subject_id))
    )
    era_counts = {
        era: sum(record.anchor_year_group == era for record in canonical_records)
        for era in SUPPORTED_ERA_TO_SPLIT
    }
    split_counts = {
        split_name: sum(
            SUPPORTED_ERA_TO_SPLIT[record.anchor_year_group] == split_name
            for record in canonical_records
        )
        for split_name in SPLIT_NAMES
    }
    qa = SplitQASummary(
        total_input_rows=len(materialized),
        total_subjects=len(canonical_records),
        train_subjects=split_counts["train"],
        validation_subjects=split_counts["validation"],
        test_subjects=split_counts["test"],
        counts_by_anchor_year_group=era_counts,
        duplicate_input_count=duplicate_input_count,
        missing_subject_count=missing_subject_count,
        missing_era_count=missing_era_count,
        unsupported_era_count=unsupported_era_count,
        conflicting_subject_count=len(conflicting_subjects),
    )
    return CohortAudit(
        canonical_records=canonical_records,
        qa=qa,
        errors=tuple(errors),
    )


def subject_sets(rows: Sequence[SubjectSplit]) -> Mapping[str, Set[Tuple[type, Identifier]]]:
    """Return subject sets for explicit pairwise-isolation checks."""

    sets = {split_name: set() for split_name in SPLIT_NAMES}
    for row in rows:
        if row.split not in sets:
            raise SplitValidationError("unsupported split name: {!r}".format(row.split))
        sets[row.split].add((type(row.subject_id), row.subject_id))
    return sets


def validate_subject_isolation(rows: Sequence[SubjectSplit]) -> None:
    """Validate one exact assignment per subject and pairwise-disjoint splits."""

    seen = set()  # type: Set[Tuple[type, Identifier]]
    for row in rows:
        if not isinstance(row, SubjectSplit):
            raise SplitValidationError("split row must be SubjectSplit")
        if not _is_supported_identifier(row.subject_id):
            raise SplitValidationError("split row has invalid subject_id")
        expected_split = era_to_split(row.anchor_year_group)
        if row.split != expected_split:
            raise SplitValidationError("split does not match frozen era mapping")
        subject_key = (type(row.subject_id), row.subject_id)
        if subject_key in seen:
            raise SplitValidationError("duplicate subject_id in split rows")
        seen.add(subject_key)

    sets = subject_sets(rows)
    if sets["train"] & sets["validation"]:
        raise SplitValidationError("train and validation subjects overlap")
    if sets["train"] & sets["test"]:
        raise SplitValidationError("train and test subjects overlap")
    if sets["validation"] & sets["test"]:
        raise SplitValidationError("validation and test subjects overlap")


def generate_subject_split(records: Iterable[CohortSubjectRecord]) -> SplitResult:
    """Generate the deterministic subject-level coarse temporal holdout."""

    audit = audit_cohort_records(records)
    if not audit.is_valid:
        raise SplitValidationError("; ".join(audit.errors))
    rows = tuple(
        SubjectSplit(
            subject_id=record.subject_id,
            split=era_to_split(record.anchor_year_group),
            anchor_year_group=record.anchor_year_group,
        )
        for record in audit.canonical_records
    )
    validate_subject_isolation(rows)
    return SplitResult(rows=rows, qa=audit.qa)


def generate_subject_split_from_mappings(
    rows: Iterable[Mapping[str, object]],
) -> SplitResult:
    """Generate from mapping rows while ignoring all non-contract columns."""

    return generate_subject_split(records_from_mappings(rows))


def serialize_split_csv(rows: Sequence[SubjectSplit]) -> bytes:
    """Serialize canonical rows deterministically as UTF-8 CSV bytes."""

    validate_subject_isolation(rows)
    ordered = sorted(rows, key=lambda row: _identifier_key(row.subject_id))
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=CSV_COLUMNS, lineterminator="\n")
    writer.writeheader()
    for row in ordered:
        writer.writerow(
            {
                "subject_id": row.subject_id,
                "split": row.split,
                "anchor_year_group": row.anchor_year_group,
            }
        )
    return stream.getvalue().encode("utf-8")


def split_content_sha256(rows: Sequence[SubjectSplit]) -> str:
    """Return SHA-256 of canonical serialized split content."""

    return sha256_bytes(serialize_split_csv(rows))


def assert_split_hash(path: PathLike, expected_sha256: str) -> None:
    """Fail closed when a split artifact does not match its expected hash."""

    if not is_sha256(expected_sha256):
        raise SplitIntegrityError("expected split SHA-256 is malformed")
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise SplitIntegrityError(
            "split SHA-256 mismatch: expected {}, found {}".format(
                expected_sha256, actual
            )
        )


def write_split_artifacts(
    result: SplitResult,
    split_path: PathLike,
    metadata_path: PathLike,
    *,
    source_cohort_version: str,
    source_cohort_sha256: str,
    split_spec_sha256: str,
    created_at_utc: str,
    code_commit: Optional[str] = None,
) -> Mapping[str, object]:
    """Write a split and metadata sidecar from an approved cohort result.

    Callers are responsible for ensuring that a final `split_v1.csv` is only
    produced from Sanskruti's frozen real retained-cohort artifact.
    """

    if result.split_spec_version != SPLIT_SPEC_VERSION:
        raise SplitValidationError("unexpected split specification version")
    if not source_cohort_version:
        raise SplitValidationError("source_cohort_version is required")
    if not is_sha256(source_cohort_sha256):
        raise SplitValidationError("source_cohort_sha256 is required and must be valid")
    if not is_sha256(split_spec_sha256):
        raise SplitValidationError("split_spec_sha256 is required and must be valid")
    if not created_at_utc:
        raise SplitValidationError("created_at_utc is required")

    content = serialize_split_csv(result.rows)
    digest = sha256_bytes(content)
    split_file = Path(split_path)
    metadata_file = Path(metadata_path)
    split_file.parent.mkdir(parents=True, exist_ok=True)
    metadata_file.parent.mkdir(parents=True, exist_ok=True)
    split_file.write_bytes(content)

    metadata = {
        "split_version": SPLIT_ARTIFACT_VERSION,
        "split_spec_version": SPLIT_SPEC_VERSION,
        "split_spec_sha256": split_spec_sha256,
        "split_file": split_file.name,
        "split_file_sha256": digest,
        "created_at_utc": created_at_utc,
        "source_cohort_version": source_cohort_version,
        "source_cohort_sha256": source_cohort_sha256,
        "code_commit": code_commit,
        "holdout_description": "coarse temporal holdout",
        "era_mapping": dict(SUPPORTED_ERA_TO_SPLIT),
        "counts": {
            "total_subjects": result.qa.total_subjects,
            "train_subjects": result.qa.train_subjects,
            "validation_subjects": result.qa.validation_subjects,
            "test_subjects": result.qa.test_subjects,
            "by_anchor_year_group": dict(result.qa.counts_by_anchor_year_group),
            "duplicate_input_count": result.qa.duplicate_input_count,
        },
        "scientific_source_version": "1.0",
    }
    metadata_file.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata


def load_split_csv(path: PathLike) -> Tuple[SubjectSplit, ...]:
    """Load and validate a serialized split artifact for integrity checks."""

    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != CSV_COLUMNS:
            raise SplitIntegrityError("split CSV header does not match contract")
        rows = tuple(
            SubjectSplit(
                subject_id=row["subject_id"],
                split=row["split"],
                anchor_year_group=row["anchor_year_group"],
            )
            for row in reader
        )
    try:
        validate_subject_isolation(rows)
    except SplitValidationError as error:
        raise SplitIntegrityError(str(error)) from error
    return rows


def verify_split_artifacts(
    split_path: PathLike, metadata_path: PathLike
) -> Mapping[str, object]:
    """Verify split hash, schema, mapping, isolation, and metadata counts."""

    try:
        metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SplitIntegrityError("split metadata is unreadable") from error
    if not isinstance(metadata, dict):
        raise SplitIntegrityError("split metadata must be a JSON object")
    if metadata.get("split_version") != SPLIT_ARTIFACT_VERSION:
        raise SplitIntegrityError("unexpected split artifact version")
    if metadata.get("split_spec_version") != SPLIT_SPEC_VERSION:
        raise SplitIntegrityError("unexpected split specification version")
    if metadata.get("holdout_description") != "coarse temporal holdout":
        raise SplitIntegrityError("incorrect holdout description")
    if metadata.get("era_mapping") != SUPPORTED_ERA_TO_SPLIT:
        raise SplitIntegrityError("metadata era mapping does not match frozen mapping")
    if metadata.get("split_file") != Path(split_path).name:
        raise SplitIntegrityError("metadata split filename does not match artifact")
    for hash_field in ("split_spec_sha256", "source_cohort_sha256"):
        hash_value = metadata.get(hash_field)
        if not isinstance(hash_value, str) or not is_sha256(hash_value):
            raise SplitIntegrityError("metadata {} is invalid".format(hash_field))
    expected_hash = metadata.get("split_file_sha256")
    if not isinstance(expected_hash, str):
        raise SplitIntegrityError("split metadata lacks split_file_sha256")
    assert_split_hash(split_path, expected_hash)
    rows = load_split_csv(split_path)
    if Path(split_path).read_bytes() != serialize_split_csv(rows):
        raise SplitIntegrityError("split CSV is not in canonical deterministic order")

    actual_counts = {
        "total_subjects": len(rows),
        "train_subjects": sum(row.split == "train" for row in rows),
        "validation_subjects": sum(row.split == "validation" for row in rows),
        "test_subjects": sum(row.split == "test" for row in rows),
        "by_anchor_year_group": {
            era: sum(row.anchor_year_group == era for row in rows)
            for era in SUPPORTED_ERA_TO_SPLIT
        },
    }
    metadata_counts = metadata.get("counts")
    if not isinstance(metadata_counts, dict):
        raise SplitIntegrityError("split metadata lacks counts")
    for key, value in actual_counts.items():
        if metadata_counts.get(key) != value:
            raise SplitIntegrityError("split metadata count mismatch for {}".format(key))
    return metadata
