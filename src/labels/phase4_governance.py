"""Materialize and register Phase-4 blocked-real and synthetic evidence."""

import json
from pathlib import Path

from experiments.lineage import (
    ArtifactRecord,
    read_artifact_index,
    read_run_registry,
    register_artifact,
)
from labels.endpoint_freeze import (
    create_synthetic_endpoint_freeze,
    write_blocked_real_freeze_audit,
)
from labels.support_prevalence import (
    create_synthetic_label_artifact,
    generate_raw_count_report,
    json_label_loader,
)
from vedant_infra.hashing import sha256_file


def _register(root: Path, record: ArtifactRecord) -> None:
    index = root / "experiments/artifacts.csv"
    existing = read_artifact_index(index)
    matches = [item for item in existing if item.artifact_id == record.artifact_id]
    if matches:
        if matches[0].artifact_path != record.artifact_path or matches[0].artifact_sha256 != record.artifact_sha256:
            raise RuntimeError("registered Phase-4 artifact conflicts with immutable evidence")
        return
    register_artifact(index, record, read_run_registry(root / "experiments/registry.csv"), root)


def materialize_phase4_artifacts(root: Path) -> None:
    root = root.resolve()
    real_path = root / "artifacts/labels/support_endpoint_freeze_v1.json"
    synthetic_freeze = root / "artifacts/labels/synthetic/support_endpoint_freeze_synthetic_v1.json"
    synthetic_labels = root / "artifacts/labels/synthetic/support_labels_phase4_synthetic_v1.json"
    synthetic_report = root / "artifacts/labels/synthetic/support_prevalence_synthetic_v1.json"

    real_hash = sha256_file(real_path) if real_path.exists() else write_blocked_real_freeze_audit(root, real_path)
    freeze_hash = sha256_file(synthetic_freeze) if synthetic_freeze.exists() else create_synthetic_endpoint_freeze(root, synthetic_freeze)
    labels_hash = sha256_file(synthetic_labels) if synthetic_labels.exists() else create_synthetic_label_artifact(
        root,
        root / "tests/fixtures/support_endpoint_freeze/synthetic_label_rows_v1.json",
        synthetic_labels,
        freeze_path=synthetic_freeze,
        freeze_sha256=freeze_hash,
    )
    descriptor = {
        "scope": "synthetic",
        "population_id": "SYNTHETIC_PHASE4_FIXTURE_ONLY",
        "partition": "synthetic",
        "freeze_sha256": freeze_hash,
        "label_artifact_sha256": labels_hash,
        "data_version": "synthetic_phase4_population_v1",
        "split_version": "synthetic_partition_only_v1",
        "timestamp_spec_version": "timestamp_spec_v1",
    }
    report_hash = sha256_file(synthetic_report) if synthetic_report.exists() else generate_raw_count_report(
        root,
        freeze_path=synthetic_freeze,
        freeze_sha256=freeze_hash,
        scope="synthetic",
        population_id="SYNTHETIC_PHASE4_FIXTURE_ONLY",
        partition="synthetic",
        label_artifact_path=synthetic_labels,
        label_artifact_sha256=labels_hash,
        label_loader=json_label_loader,
        descriptor=descriptor,
        output_path=synthetic_report,
    )
    event_hash = sha256_file(root / "configs/event_dict_v1.yaml")
    config_hash = sha256_file(root / "configs/support_prevalence_v1.json")
    created = json.loads(synthetic_report.read_text())["generated_at_utc"]
    common = dict(task="organ_support", label_version="organ_support_endpoint_v1", event_dict_version="event_dict_v1", status="registered")
    records = (
        ArtifactRecord(
            artifact_id="phase4_support:event_dictionary:v1", artifact_path="configs/event_dict_v1.yaml",
            artifact_type="event_dictionary", artifact_version="event_dict_v1", artifact_sha256=event_hash,
            producing_run_id="", generating_script="manual_reviewed_configuration", run_type="development", **common,
        ),
        ArtifactRecord(
            artifact_id="phase4_support:real_freeze_blocked_audit:v1", artifact_path="artifacts/labels/support_endpoint_freeze_v1.json",
            artifact_type="support_endpoint_freeze_audit", artifact_version="support_endpoint_freeze_record_v1",
            artifact_sha256=real_hash, producing_run_id="", parent_artifact_ids="phase4_support:event_dictionary:v1",
            config_hash=sha256_file(root / "configs/support_endpoint_signoff_v1.json"),
            generating_script="src/labels/phase4_governance.py", run_type="development", **common,
        ),
        ArtifactRecord(
            artifact_id="phase4_support:synthetic_freeze:v1", artifact_path="artifacts/labels/synthetic/support_endpoint_freeze_synthetic_v1.json",
            artifact_type="support_endpoint_freeze", artifact_version="support_endpoint_freeze_record_v1",
            artifact_sha256=freeze_hash, producing_run_id="", parent_artifact_ids="phase4_support:event_dictionary:v1",
            generating_script="src/labels/phase4_governance.py", creation_date_utc=created, partition="synthetic", run_type="synthetic", **common,
        ),
        ArtifactRecord(
            artifact_id="phase4_support:synthetic_labels:v1", artifact_path="artifacts/labels/synthetic/support_labels_phase4_synthetic_v1.json",
            artifact_type="support_label_artifact", artifact_version="support_label_artifact_v1",
            artifact_sha256=labels_hash, producing_run_id="", parent_artifact_ids="phase4_support:synthetic_freeze:v1",
            generating_script="src/labels/phase4_governance.py", creation_date_utc=created, partition="synthetic",
            dataset_version="synthetic_phase4_population_v1", run_type="synthetic", **common,
        ),
        ArtifactRecord(
            artifact_id="phase4_support:synthetic_prevalence:v1", artifact_path="artifacts/labels/synthetic/support_prevalence_synthetic_v1.json",
            artifact_type="support_prevalence_report", artifact_version="support_prevalence_raw_counts_v1",
            artifact_sha256=report_hash, producing_run_id="", parent_artifact_ids="phase4_support:synthetic_freeze:v1;phase4_support:synthetic_labels:v1",
            config_hash=config_hash, generating_script="src/labels/phase4_governance.py", creation_date_utc=created,
            partition="synthetic", dataset_version="synthetic_phase4_population_v1", run_type="synthetic", **common,
        ),
    )
    for record in records:
        _register(root, record)


if __name__ == "__main__":
    materialize_phase4_artifacts(Path.cwd())
