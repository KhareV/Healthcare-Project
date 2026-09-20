"""Cryptographic and environment provenance helpers."""
from __future__ import annotations
import hashlib, json, platform, subprocess, sys
from pathlib import Path
from typing import Mapping, Sequence
import numpy

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""): digest.update(chunk)
    return digest.hexdigest()

def semantic_sha256(table: str, rows: Sequence[Mapping]) -> str:
    encoded = json.dumps({"table": table, "rows": list(rows)}, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(encoded).hexdigest()

def source_bundle_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted((root / "src/data/synthetic").glob("*.py")):
        digest.update(str(path.relative_to(root)).encode() + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()

def git_identity(root: Path):
    def run(*args): return subprocess.run(["git", *args], cwd=root, text=True, capture_output=True, check=True).stdout.strip()
    try: return {"git_commit": run("rev-parse", "HEAD"), "git_worktree_dirty": bool(run("status", "--porcelain"))}
    except (OSError, subprocess.CalledProcessError): return {"git_commit": None, "git_worktree_dirty": None}

def environment_identity():
    return {"python": sys.version.split()[0], "numpy": numpy.__version__, "platform": platform.platform()}
