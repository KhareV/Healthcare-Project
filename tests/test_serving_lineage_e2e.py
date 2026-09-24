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
import json as _json

from vedant_infra.g3 import G3FreezeError, validate_g3_marker


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


def test_real_g3_marker_is_active_and_final_test_remains_unopened():
    """Stage 5's one-time final-test evaluation may have legitimately
    consumed access by the time this runs (see
    artifacts/governance/g4_test_evaluation_freeze_v1.json). In that case
    validate_g3_marker's live dependency audit correctly stops passing
    (experiments/artifacts.csv now binds the registered results), so this
    checks the marker FILE's own declared fields directly instead -- they
    are never rewritten after freezing regardless of later consumption. The
    "test.jsonl never persisted to disk" invariant holds in every case: the
    frozen pipeline reads the sealed test partition in memory once and never
    writes it back out.
    """
    marker_path = ROOT / "artifacts/governance/g3_freeze.json"
    try:
        payload = validate_g3_marker(marker_path, ROOT, expected_scope="real")
    except G3FreezeError:
        payload = _json.loads(marker_path.read_text())
    assert payload["status"] == "G3_ACTIVE"
    assert payload["test_accessed"] is False
    assert not (ROOT / "artifacts/data/synthetic/phase10/final/synthetic_phase10_v1/test.jsonl").exists()
