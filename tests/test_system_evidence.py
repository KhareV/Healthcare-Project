import json
import re
import shutil
from pathlib import Path

import pytest

from evidence.audit import EvidenceAuditError
from evidence.system import EVIDENCE_ROOT, audit


ROOT = Path(__file__).resolve().parents[1]


def _copy_governed_package(tmp_path):
    manifest_path = ROOT / EVIDENCE_ROOT / "system_evidence_manifest_v1.json"
    manifest = json.loads(manifest_path.read_text())
    paths = {str(EVIDENCE_ROOT / "system_evidence_manifest_v1.json")}
    for item in manifest["items"]:
        if item["output_path"]:
            paths.add(item["output_path"])
        paths.update(source["path"] for source in item["source_artifacts"])
    for relative in paths:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
    return manifest


def test_generated_system_evidence_is_current_indexed_and_orphan_free():
    result = audit(ROOT)
    assert result["status"] == "PASS"
    assert result["orphans"] == 0
    assert result["indexed_outputs"] >= 10


def test_system_evidence_detects_source_drift(tmp_path):
    manifest = _copy_governed_package(tmp_path)
    source = next(
        item["source_artifacts"][0]["path"]
        for item in manifest["items"]
        if item["source_artifacts"]
    )
    path = tmp_path / source
    path.write_bytes(path.read_bytes() + b"\nDRIFT")
    with pytest.raises(EvidenceAuditError, match="source hash mismatch"):
        audit(tmp_path)


def test_system_evidence_detects_orphan_output(tmp_path):
    _copy_governed_package(tmp_path)
    orphan = tmp_path / EVIDENCE_ROOT / "dashboard/unindexed.png"
    orphan.parent.mkdir(parents=True, exist_ok=True)
    orphan.write_bytes(b"not a screenshot")
    with pytest.raises(EvidenceAuditError, match="orphan"):
        audit(tmp_path)


def test_blocked_release_slots_have_no_fabricated_output_or_signoff():
    manifest = json.loads(
        (ROOT / EVIDENCE_ROOT / "system_evidence_manifest_v1.json").read_text()
    )
    blocked = {item["evidence_id"]: item for item in manifest["items"] if item["scope"] == "blocked"}
    assert {"git_tag", "real_api", "screenshots", "clean_install", "non_owner", "member_reviews"} <= set(blocked)
    assert all(item["status"] == "BLOCKED" for item in blocked.values())
    assert all(item["output_path"] is None for item in blocked.values())
    assert all(item["output_sha256"] is None for item in blocked.values())
    assert manifest["scientific_values_generated"] is False
    assert manifest["final_test_accessed"] is False


def test_system_evidence_contains_no_personal_path_secret_or_raw_mimic_fields():
    text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (ROOT / EVIDENCE_ROOT).rglob("*")
        if path.is_file()
    )
    assert not re.search(r"/Users/[^/\s]+/|/home/[^/\s]+/|[A-Za-z]:\\Users\\", text)
    assert not re.search(
        r"(?i)(password|api[_-]?token|access[_-]?token|api[_-]?key)\s*[=:]\s*[^<\s]+",
        text,
    )
    for prohibited in ('"itemid"', '"hadm_id"', '"caregiver_id"', '"patient_name"'):
        assert prohibited not in text.lower()


def test_system_evidence_uses_associational_explanation_language():
    evidence = ROOT / EVIDENCE_ROOT
    text = "\n".join(
        path.read_text(encoding="utf-8").lower()
        for path in evidence.rglob("*")
        if path.is_file() and path.suffix in {".md", ".json"}
    )
    assert "treatment driver" not in text
    assert "clinical reason" not in text
    assert "patient-specific confidence interval" not in text
