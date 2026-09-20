from pathlib import Path

from reproducibility.audit import audit_environment_lock
from vedant_infra.hashing import sha256_file


def test_missing_lock_is_blocked(tmp_path):
    result = audit_environment_lock(tmp_path, {})
    assert result["status"] == "BLOCKED"
    assert "EXACT ENVIRONMENT LOCK" in result["detail"]


def test_wildcard_lock_cannot_support_final_claim(tmp_path):
    lock = tmp_path / "environment.lock.txt"
    lock.write_text("torch==2.x\nnumpy==1.26.4\n", encoding="utf-8")
    result = audit_environment_lock(
        tmp_path,
        {
            "environment_lock_ref": "environment.lock.txt",
            "environment_lock_sha256": sha256_file(lock),
        },
    )
    assert result["status"] == "BLOCKED"
    assert result["invalid_entries"] == ("torch==2.x",)
