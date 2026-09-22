"""Artifact index, compatibility validation, and bidirectional tracing."""

import csv
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

from vedant_infra.hashing import is_sha256, sha256_file


ARTIFACT_COLUMNS = (
    "artifact_id",
    "artifact_path",
    "artifact_type",
    "artifact_version",
    "artifact_sha256",
    "producing_run_id",
    "parent_artifact_ids",
    "task",
    "model_family",
    "split_hash",
    "feature_version",
    "label_version",
    "config_hash",
    "model_sha256",
    "calibrator_sha256",
    "creation_commit",
    "metadata_ref",
    "generating_script",
    "creation_date_utc",
    "partition",
    "probability_type",
    "prediction_population_hash",
    "calibration_method",
    "threshold_criterion",
    "threshold_value",
    "evaluator_version",
    "metric_implementation_version",
    "slice_spec_hash",
    "dataset_version",
    "mimic_code_commit",
    "extraction_sha256",
    "cohort_version",
    "feature_dictionary_sha256",
    "event_dict_version",
    "preprocessor_sha256",
    "environment_sha256",
    "run_type",
    "status",
)
ALLOWED_ARTIFACT_STATUSES = {"registered", "legacy_incomplete", "invalid", "stale"}
OUTPUT_TYPES = {
    "model_checkpoint",
    "prediction",
    "metric_table",
    "selected_model_manifest",
    "calibrator",
    "threshold",
    "ablation_table",
    "error_analysis_table",
    "evidence_plot",
    "training_log",
    "preprocessor",
}
RESULT_TYPES = {"metric_table", "ablation_table", "error_analysis_table", "evidence_plot"}
PROHIBITED_FIELDS = {
    "password",
    "api_token",
    "access_token",
    "secret",
    "raw_clinical_note",
    "patient_name",
    "subject_id",
    "stay_id",
}


class LineageValidationError(ValueError):
    """Raised for orphaned, drifted, cyclic, or incompatible provenance."""


@dataclass(frozen=True)
class ArtifactRecord:
    artifact_id: str
    artifact_path: str
    artifact_type: str
    artifact_version: str
    artifact_sha256: str
    producing_run_id: str
    parent_artifact_ids: str = ""
    task: str = ""
    model_family: str = ""
    split_hash: str = ""
    feature_version: str = ""
    label_version: str = ""
    config_hash: str = ""
    model_sha256: str = ""
    calibrator_sha256: str = ""
    creation_commit: str = ""
    metadata_ref: str = ""
    generating_script: str = ""
    creation_date_utc: str = ""
    partition: str = ""
    probability_type: str = ""
    prediction_population_hash: str = ""
    calibration_method: str = ""
    threshold_criterion: str = ""
    threshold_value: str = ""
    evaluator_version: str = ""
    metric_implementation_version: str = ""
    slice_spec_hash: str = ""
    dataset_version: str = ""
    mimic_code_commit: str = ""
    extraction_sha256: str = ""
    cohort_version: str = ""
    feature_dictionary_sha256: str = ""
    event_dict_version: str = ""
    preprocessor_sha256: str = ""
    environment_sha256: str = ""
    run_type: str = ""
    status: str = "registered"

    @property
    def parents(self) -> Tuple[str, ...]:
        return tuple(value for value in self.parent_artifact_ids.split(";") if value)


def read_run_registry(path: Path) -> Tuple[Mapping[str, str], ...]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return tuple(dict(row) for row in csv.DictReader(handle))


def read_artifact_index(path: Path) -> Tuple[ArtifactRecord, ...]:
    if not path.exists():
        return ()
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != ARTIFACT_COLUMNS:
            raise LineageValidationError("artifact index schema mismatch")
        return tuple(ArtifactRecord(**dict(row)) for row in reader)


def write_artifact_index(path: Path, records: Sequence[ArtifactRecord]) -> None:
    validate_artifact_lineage(records, (), repository_root=None, verify_files=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ARTIFACT_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(record.__dict__ for record in records)
    temporary.replace(path)


def _validate_no_restricted_metadata(values: Iterable[str]) -> None:
    rendered = " ".join(str(value).lower() for value in values)
    for field in PROHIBITED_FIELDS:
        if field in rendered:
            raise LineageValidationError("lineage metadata contains prohibited field: " + field)


def _cycle_check(records: Sequence[ArtifactRecord]) -> None:
    graph = {record.artifact_id: record.parents for record in records}
    visiting = set()
    visited = set()

    def visit(node):
        if node in visiting:
            raise LineageValidationError("artifact lineage contains a cycle")
        if node in visited:
            return
        visiting.add(node)
        for parent in graph.get(node, ()):
            visit(parent)
        visiting.remove(node)
        visited.add(node)

    for artifact_id in graph:
        visit(artifact_id)


def validate_artifact_lineage(
    records: Sequence[ArtifactRecord],
    runs: Sequence[Mapping[str, str]],
    *,
    repository_root: Optional[Path],
    verify_files: bool = True,
) -> None:
    ids = [record.artifact_id for record in records]
    paths = [record.artifact_path for record in records]
    if len(ids) != len(set(ids)):
        raise LineageValidationError("duplicate artifact_id")
    if len(paths) != len(set(paths)):
        raise LineageValidationError("duplicate immutable artifact path")
    by_id = {record.artifact_id: record for record in records}
    by_run = {row.get("run_id", ""): row for row in runs}
    for record in records:
        _validate_no_restricted_metadata(record.__dict__.values())
        if not record.artifact_id or not record.artifact_path or not record.artifact_type:
            raise LineageValidationError("artifact identity/path/type are required")
        if Path(record.artifact_path).is_absolute() or PureWindowsPath(record.artifact_path).is_absolute():
            raise LineageValidationError("artifact paths must be repository-relative")
        if not is_sha256(record.artifact_sha256):
            raise LineageValidationError("artifact hash must be SHA-256")
        for field in (
            "split_hash",
            "config_hash",
            "model_sha256",
            "calibrator_sha256",
            "prediction_population_hash",
            "slice_spec_hash",
            "extraction_sha256",
            "feature_dictionary_sha256",
            "preprocessor_sha256",
            "environment_sha256",
        ):
            value = getattr(record, field)
            if value and not is_sha256(value):
                raise LineageValidationError(field + " must be SHA-256")
        if record.status not in ALLOWED_ARTIFACT_STATUSES:
            raise LineageValidationError("unknown artifact status")
        if record.producing_run_id and record.producing_run_id not in by_run and runs:
            raise LineageValidationError("artifact points to unknown producing run")
        if record.artifact_type in OUTPUT_TYPES and not record.producing_run_id:
            raise LineageValidationError("output artifact requires producing run")
        for parent in record.parents:
            if parent not in by_id:
                raise LineageValidationError("artifact points to unknown parent artifact")
        if verify_files:
            if repository_root is None:
                raise LineageValidationError("repository root required for hash validation")
            artifact_path = repository_root / record.artifact_path
            if not artifact_path.is_file():
                raise LineageValidationError("MISSING ARTIFACT: " + record.artifact_path)
            if sha256_file(artifact_path) != record.artifact_sha256:
                raise LineageValidationError("artifact SHA-256 mismatch: " + record.artifact_id)
        run = by_run.get(record.producing_run_id)
        if run:
            for artifact_field, run_field in (
                ("task", "task"),
                ("model_family", "model_family"),
                ("split_hash", "split_hash"),
                ("feature_version", "feature_version"),
                ("label_version", "label_version"),
            ):
                artifact_value = getattr(record, artifact_field)
                run_value = run.get(run_field, "")
                if artifact_value and run_value and artifact_value != run_value:
                    raise LineageValidationError(artifact_field + " mismatch")
        _validate_artifact_class(record, by_id)
    _cycle_check(records)


def _validate_artifact_class(record: ArtifactRecord, by_id: Mapping[str, ArtifactRecord]) -> None:
    parents = [by_id[parent] for parent in record.parents if parent in by_id]
    if record.status == "legacy_incomplete":
        return
    if record.run_type in ("scientific", "scientific_sensitivity"):
        required_by_type = {
            "model_checkpoint": (
                "task", "model_family", "split_hash", "feature_version",
                "label_version", "config_hash", "creation_commit",
            ),
            "prediction": (
                "task", "model_family", "split_hash", "feature_version",
                "label_version", "partition", "prediction_population_hash",
            ),
            "calibrator": ("calibration_method", "partition"),
            "threshold": ("threshold_criterion", "threshold_value", "partition"),
            "metric_table": ("evaluator_version", "metric_implementation_version", "partition"),
            "error_analysis_table": ("evaluator_version", "metric_implementation_version", "partition", "slice_spec_hash"),
            "evidence_plot": ("generating_script", "creation_date_utc"),
            "dataset_feature": (
                "dataset_version", "extraction_sha256", "cohort_version",
                "feature_version", "feature_dictionary_sha256", "split_hash",
                "creation_commit",
            ),
        }
        missing = [
            field
            for field in required_by_type.get(record.artifact_type, ())
            if not getattr(record, field)
        ]
        if missing:
            raise LineageValidationError(
                record.artifact_type + " missing required metadata: " + ", ".join(missing)
            )
    if record.artifact_type == "prediction":
        models = [item for item in parents if item.artifact_type == "model_checkpoint"]
        if len(models) != 1 or record.model_sha256 != models[0].artifact_sha256:
            raise LineageValidationError("prediction/model binding is incompatible")
    elif record.artifact_type == "calibrator":
        models = [item for item in parents if item.artifact_type == "model_checkpoint"]
        predictions = [item for item in parents if item.artifact_type == "prediction"]
        if len(models) != 1 or len(predictions) != 1 or record.model_sha256 != models[0].artifact_sha256:
            raise LineageValidationError("calibrator/model prediction binding is incompatible")
    elif record.artifact_type == "threshold":
        calibrators = [item for item in parents if item.artifact_type == "calibrator"]
        if len(calibrators) != 1 or record.calibrator_sha256 != calibrators[0].artifact_sha256:
            raise LineageValidationError("threshold/calibrator binding is incompatible")
    elif record.artifact_type == "selected_model_manifest":
        models = [item for item in parents if item.artifact_type == "model_checkpoint"]
        if not models:
            raise LineageValidationError("selected-model manifest requires model parents")
    elif record.artifact_type in RESULT_TYPES:
        predictions = [item for item in parents if item.artifact_type == "prediction"]
        if not predictions:
            raise LineageValidationError("result artifact requires registered prediction lineage")


def register_artifact(
    index_path: Path,
    record: ArtifactRecord,
    runs: Sequence[Mapping[str, str]],
    repository_root: Path,
) -> None:
    existing = list(read_artifact_index(index_path))
    if any(item.artifact_id == record.artifact_id or item.artifact_path == record.artifact_path for item in existing):
        raise LineageValidationError("immutable artifact identity/path already registered")
    candidate = existing + [record]
    validate_artifact_lineage(candidate, runs, repository_root=repository_root)
    write_artifact_index(index_path, candidate)


def artifact_descendants(records: Sequence[ArtifactRecord], artifact_id: str) -> Tuple[str, ...]:
    if artifact_id not in {record.artifact_id for record in records}:
        raise LineageValidationError("unknown artifact")
    found = set()
    changed = True
    while changed:
        changed = False
        for record in records:
            if record.artifact_id not in found and (
                artifact_id in record.parents or any(parent in found for parent in record.parents)
            ):
                found.add(record.artifact_id)
                changed = True
    return tuple(sorted(found))


def trace_artifact(
    artifact_id: str,
    records: Sequence[ArtifactRecord],
    runs: Sequence[Mapping[str, str]],
) -> Mapping[str, object]:
    by_id = {record.artifact_id: record for record in records}
    by_run = {row.get("run_id", ""): row for row in runs}
    if artifact_id not in by_id:
        raise LineageValidationError("unknown artifact")
    ancestors = []
    seen = set()

    def collect(identifier):
        for parent in by_id[identifier].parents:
            if parent not in seen:
                seen.add(parent)
                ancestors.append(parent)
                collect(parent)

    collect(artifact_id)
    record = by_id[artifact_id]
    run = by_run.get(record.producing_run_id, {})
    return {
        "artifact": record.__dict__,
        "ancestor_artifact_ids": tuple(ancestors),
        "producing_run": dict(run),
        "configuration_ref": run.get("config_ref", ""),
        "configuration_hash": run.get("config_hash", ""),
        "split_hash": run.get("split_hash", ""),
        "feature_version": run.get("feature_version", ""),
        "label_version": run.get("label_version", ""),
        "code_commit": run.get("code_commit", ""),
    }


def trace_result(
    artifact_id: str,
    records: Sequence[ArtifactRecord],
    runs: Sequence[Mapping[str, str]],
) -> Mapping[str, object]:
    record = next((item for item in records if item.artifact_id == artifact_id), None)
    if record is None or record.artifact_type not in RESULT_TYPES:
        raise LineageValidationError("result trace requires a registered result artifact")
    return trace_artifact(artifact_id, records, runs)


def trace_run(
    run_id: str,
    records: Sequence[ArtifactRecord],
    runs: Sequence[Mapping[str, str]],
) -> Mapping[str, object]:
    by_run = {row.get("run_id", ""): row for row in runs}
    if run_id not in by_run:
        raise LineageValidationError("unknown run_id")
    direct = tuple(record.artifact_id for record in records if record.producing_run_id == run_id)
    downstream = set()
    for artifact_id in direct:
        downstream.update(artifact_descendants(records, artifact_id))
    children = tuple(
        row.get("run_id", "")
        for row in runs
        if row.get("parent_run_id") == run_id or row.get("retry_of_run_id") == run_id
    )
    return {
        "run": dict(by_run[run_id]),
        "direct_artifact_ids": direct,
        "downstream_artifact_ids": tuple(sorted(downstream)),
        "child_run_ids": children,
    }


def final_scientific_runs(runs: Sequence[Mapping[str, str]]) -> Tuple[Mapping[str, str], ...]:
    return tuple(
        row
        for row in runs
        if row.get("run_type") == "scientific"
        and row.get("status") == "completed"
        and row.get("finalized") == "true"
    )


def migrate_historical_artifacts(
    root: Path,
    runs: Sequence[Mapping[str, str]],
    index_path: Path,
) -> Tuple[ArtifactRecord, ...]:
    """Index only artifacts and metadata exactly recoverable from old run rows."""

    records = []
    preprocessors_by_path = {}
    for row in runs:
        common = {
            "producing_run_id": row.get("run_id", ""),
            "task": row.get("task", ""),
            "model_family": row.get("model_family", ""),
            "split_hash": row.get("split_hash", ""),
            "feature_version": row.get("feature_version", ""),
            "label_version": row.get("label_version", ""),
            "config_hash": row.get("config_hash", ""),
            "creation_commit": row.get("code_commit", ""),
            "run_type": row.get("run_type", ""),
        }
        preprocessor_ref = row.get("preprocessor_ref", "")
        if preprocessor_ref and preprocessor_ref not in preprocessors_by_path:
            preprocessor_path = root / preprocessor_ref
            if preprocessor_path.is_file():
                actual = sha256_file(preprocessor_path)
                if actual != row.get("preprocessor_sha256"):
                    raise LineageValidationError("historical preprocessor hash mismatch")
                preprocessor_id = row["run_id"] + ":preprocessor"
                preprocessors_by_path[preprocessor_ref] = preprocessor_id
                records.append(
                    ArtifactRecord(
                        artifact_id=preprocessor_id,
                        artifact_path=preprocessor_ref,
                        artifact_type="preprocessor",
                        artifact_version="historical_synthetic_preprocessor_v1",
                        artifact_sha256=actual,
                        status="registered",
                        **common,
                    )
                )
        model_ref = row.get("model_artifact_ref", "")
        model_path = root / model_ref if model_ref else None
        if model_path is not None and model_path.is_file():
            actual = sha256_file(model_path)
            if actual != row.get("model_sha256"):
                raise LineageValidationError("historical model artifact hash mismatch")
            records.append(
                ArtifactRecord(
                    artifact_id=row["run_id"] + ":checkpoint",
                    artifact_path=model_ref,
                    artifact_type="model_checkpoint",
                    artifact_version="historical_synthetic_checkpoint_v1",
                    artifact_sha256=actual,
                    parent_artifact_ids=preprocessors_by_path.get(preprocessor_ref, ""),
                    metadata_ref=model_ref + ".metadata.json",
                    status="registered",
                    **common,
                )
            )
        metrics_ref = row.get("metrics_ref", "")
        metrics_path = root / metrics_ref if metrics_ref else None
        if metrics_path is not None and metrics_path.is_file():
            records.append(
                ArtifactRecord(
                    artifact_id=row["run_id"] + ":metrics",
                    artifact_path=metrics_ref,
                    artifact_type="metric_table",
                    artifact_version="historical_synthetic_metrics_v1",
                    artifact_sha256=sha256_file(metrics_path),
                    parent_artifact_ids="",
                    status="legacy_incomplete",
                    **common,
                )
            )
    validate_artifact_lineage(records, runs, repository_root=root)
    write_artifact_index(index_path, records)
    return tuple(records)
