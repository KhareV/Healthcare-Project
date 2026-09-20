import json
from pathlib import Path

import pytest

from evidence.audit import EvidenceAuditError, audit_manifest
from vedant_infra.hashing import sha256_file


def test_manual_table_cell_change_breaks_evidence_hash(tmp_path):
    source = tmp_path / "source.json"
    source.write_text(json.dumps({"metric": 1.0}), encoding="utf-8")
    output = tmp_path / "evidence/table.csv"
    output.parent.mkdir(parents=True)
    output.write_text("metric,value\nmae,1.0000\n", encoding="utf-8")
    manifest = {
        "evidence_root": "evidence",
        "items": [{
            "evidence_id": "table",
            "status": "VALIDATION-ONLY",
            "output_path": "evidence/table.csv",
            "output_sha256": sha256_file(output),
            "generator": "generator.py",
            "generated_at": "2026-09-19",
            "input_artifacts": ["source.json"],
            "input_hashes": [sha256_file(source)],
        }],
    }
    output.write_text("metric,value\nmae,9.9999\n", encoding="utf-8")
    with pytest.raises(EvidenceAuditError, match="hash mismatch"):
        audit_manifest(tmp_path, manifest)
