import csv
from dataclasses import replace

import pytest

from registry_helpers import run, write_runs
from vedant_infra.registry import REGISTRY_COLUMNS, RegistryValidationError, validate_registry


def test_valid_scientific_run_and_required_fields(tmp_path):
    path = tmp_path / "registry.csv"
    write_runs(path, (run(),))
    assert validate_registry(path) == 1
    for field in ("run_id", "task", "model_family", "config_hash", "split_hash"):
        broken = run()
        broken[field] = ""
        write_runs(path, (broken,))
        with pytest.raises(RegistryValidationError):
            validate_registry(path)


def test_duplicate_run_id_and_unknown_parent_fail(tmp_path):
    path = tmp_path / "registry.csv"
    write_runs(path, (run(), run()))
    with pytest.raises(RegistryValidationError, match="duplicate run_id"):
        validate_registry(path)
    write_runs(path, (run(parent_run_id="missing"),))
    with pytest.raises(RegistryValidationError, match="unknown parent_run_id"):
        validate_registry(path)
