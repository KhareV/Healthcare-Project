import csv

import pytest

from experiments.lineage import write_artifact_index
from experiments.validation_suite import ValidationSuiteError, validate_comparison_traceability
from registry_helpers import chain, run, write_runs
from vedant_infra.hashing import sha256_bytes


def test_comparison_rows_must_resolve_run_and_artifact(tmp_path):
    selected = run(
        model_artifact_ref="artifacts/model.bin",
        model_sha256=sha256_bytes(b"model"),
    )
    records = chain(tmp_path, selected)
    write_runs(tmp_path / "experiments/registry.csv", (selected,))
    write_artifact_index(tmp_path / "experiments/artifacts.csv", records)
    row = {
        "run_id": selected["run_id"],
        "model_artifact_ref": records[0].artifact_path,
        "model_sha256": records[0].artifact_sha256,
    }
    validate_comparison_traceability((row,), tmp_path)
    with pytest.raises(ValidationSuiteError, match="unregistered"):
        validate_comparison_traceability((dict(row, run_id="manual"),), tmp_path)
