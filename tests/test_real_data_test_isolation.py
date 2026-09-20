from pathlib import Path

import pytest

from data.acceptance import DataAcceptanceError, audit_real_data, require_accepted_real_data
from data.real_adapter import load_real_bundle
from real_data_helpers import build_real_handoff_fixture
from vedant_infra.hashing import sha256_file


CONFIG = Path(__file__).resolve().parents[1] / "configs/data_acceptance_v1.json"


def test_protected_test_label_artifact_is_never_opened(tmp_path, monkeypatch):
    manifest = build_real_handoff_fixture(tmp_path)
    protected = (tmp_path / "artifacts/data/protected/test_labels_v1.parquet").resolve()
    original_text = Path.read_text

    def hostile(self, *args, **kwargs):
        if self.resolve() == protected:
            raise AssertionError("TEST_LABELS_OPENED")
        return original_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", hostile)
    bundle = load_real_bundle(tmp_path, manifest)
    assert len([row for row in bundle.structural_rows if row.split == "test"]) > 0


def test_phase19_guard_refuses_blocked_report(tmp_path):
    report = audit_real_data(tmp_path, config_path=CONFIG)
    path = tmp_path / "blocked.json"
    from data.acceptance import write_acceptance_report

    write_acceptance_report(path, report)
    with pytest.raises(DataAcceptanceError, match="not ACCEPTED"):
        require_accepted_real_data(path, sha256_file(path))


def test_acceptance_audit_does_not_create_search_attempts(tmp_path):
    manifest = build_real_handoff_fixture(tmp_path)
    registry = tmp_path / "experiments/registry.csv"
    registry.parent.mkdir(parents=True)
    registry.write_text("sentinel-no-search\n", encoding="utf-8")
    before = registry.read_bytes()
    report = audit_real_data(tmp_path, manifest_path=manifest, config_path=CONFIG)
    assert report["performance_inspected"] is False
    assert registry.read_bytes() == before
