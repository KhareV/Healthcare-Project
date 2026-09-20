"""Read-only Phase-14 registry, lineage, scope, and governance integration."""

from dataclasses import replace
from pathlib import Path

import pytest

from experiments.audit import audit_registry_lineage
from experiments.lineage import (
    LineageValidationError,
    final_scientific_runs,
    write_artifact_index,
)
from registry_helpers import chain, run, write_runs
from vedant_infra.g3 import audit_g3


ROOT = Path(__file__).resolve().parents[1]


def test_repository_registry_and_lineage_are_clean_for_critical_integrity():
    summary = audit_registry_lineage(ROOT)
    for field in (
        "missing_artifacts",
        "hash_mismatches",
        "scientific_orphan_artifacts",
        "broken_parent_runs",
        "broken_artifact_parents",
        "unknown_artifact_producers",
        "incompatible_dependencies",
    ):
        assert summary[field] in ((), 0), (field, summary[field])
    assert summary["final_test_accessed"] is False
    assert summary["g3_marker_created"] is False


def test_registry_audit_detects_post_registration_hash_drift(tmp_path):
    selected = run()
    records = chain(tmp_path, selected)
    write_runs(tmp_path / "experiments/registry.csv", (selected,))
    write_artifact_index(tmp_path / "experiments/artifacts.csv", records)
    (tmp_path / "artifacts/model.bin").write_bytes(b"mutated-after-registration")
    summary = audit_registry_lineage(tmp_path)
    assert "model" in summary["hash_mismatches"]


def test_synthetic_runs_cannot_enter_final_scientific_query():
    synthetic = run(run_type="synthetic", finalized="true")
    assert final_scientific_runs((synthetic,)) == ()


def test_restricted_row_identity_cannot_be_registered(tmp_path):
    selected = run()
    model = chain(tmp_path, selected)[0]
    hostile = replace(model, artifact_id="raw_subject_id_export")
    with pytest.raises(LineageValidationError, match="prohibited field"):
        write_artifact_index(tmp_path / "experiments/artifacts.csv", (hostile,))


def test_real_g3_audit_is_non_authorizing_and_final_test_remains_unopened():
    marker = ROOT / "artifacts/governance/g3_freeze.json"
    before = marker.exists()
    report = audit_g3(ROOT, scope="real")
    assert report.overall == "BLOCKED"
    assert report.test_data_accessed is False
    assert marker.exists() is before is False
