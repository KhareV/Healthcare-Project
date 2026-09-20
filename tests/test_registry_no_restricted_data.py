import json

import pytest

from experiments.audit import audit_registry_lineage
from experiments.lineage import ArtifactRecord, LineageValidationError, validate_artifact_lineage, write_artifact_index
from registry_helpers import run, write_runs


def test_artifact_metadata_rejects_explicit_restricted_fields(tmp_path):
    record = ArtifactRecord(
        "bad", "artifacts/bad.json", "configuration", "v1", "a" * 64, "",
        metadata_ref="raw_clinical_note.txt",
    )
    with pytest.raises(LineageValidationError, match="prohibited"):
        validate_artifact_lineage((record,), (), repository_root=None, verify_files=False)


def test_audit_reports_orphan_missing_and_incomplete_without_reading_test_data(tmp_path):
    write_runs(tmp_path / "experiments/registry.csv", (run(),))
    orphan = tmp_path / "artifacts/smoke/orphan.txt"
    orphan.parent.mkdir(parents=True)
    orphan.write_text("metadata only", encoding="utf-8")
    write_artifact_index(tmp_path / "experiments/artifacts.csv", ())
    summary = audit_registry_lineage(tmp_path)
    assert summary["orphan_artifacts"] == ("artifacts/smoke/orphan.txt",)
    assert summary["final_test_accessed"] is False
    assert summary["g3_marker_created"] is False
