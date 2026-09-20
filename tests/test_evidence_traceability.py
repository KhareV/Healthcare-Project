from pathlib import Path

import pytest

from evidence.audit import EvidenceAuditError, audit_generated_package, audit_manifest


ROOT = Path(__file__).resolve().parents[1]


def test_generated_package_has_no_orphans_and_all_hashes_resolve():
    result = audit_generated_package(ROOT)
    assert result["status"] == "PASS"
    assert result["traceability"]["orphan_outputs"] == 0


def test_orphan_evidence_file_is_rejected(tmp_path):
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    orphan = evidence / "orphan.png"
    orphan.write_bytes(b"fake")
    manifest = {
        "evidence_root": "evidence",
        "items": [],
    }
    with pytest.raises(EvidenceAuditError, match="orphan"):
        audit_manifest(tmp_path, manifest)
