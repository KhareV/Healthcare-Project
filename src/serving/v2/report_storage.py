"""Object-storage abstraction for uploaded health-record report files.

Report *binary content* never lives in MongoDB (src/serving/v2/persistence.py
stores only report metadata) -- it lives here, addressed by an opaque,
server-generated `storage_key` that a client never sees or controls (see
api/v2_app.py's report endpoints). This module only implements the storage
side; ownership/authorization is enforced entirely by the API layer before
any of these methods are ever called.

`LocalReportStorage` is the only implementation this pass ships: files land
under a local, gitignored directory (runtime/uploads/ by default). An
S3-compatible adapter is a natural drop-in behind the same
`ReportStorageProvider` protocol, deliberately not built here -- the
application must run entirely locally with zero cloud dependency (see
docs/product_v2/HEALTH_RECORD_ARCHITECTURE.md).
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol


class ReportStorageError(RuntimeError):
    """Raised for any storage-layer failure (missing file, invalid key)."""


class ReportStorageProvider(Protocol):
    def save(self, *, key: str, data: bytes) -> None: ...
    def open(self, *, key: str) -> bytes: ...
    def delete(self, *, key: str) -> None: ...
    def exists(self, *, key: str) -> bool: ...


class LocalReportStorage:
    """Filesystem-backed storage under a fixed root directory.

    `key` is always a server-generated string of the form
    "<owner_user_id>/<report_id>_<sanitized_filename>" (see api/v2_app.py's
    upload endpoint) -- never derived from unsanitized client input. Even
    so, `_resolve` defensively refuses to resolve outside `root`, so a
    malformed or malicious key (e.g. containing "..") can never escape the
    storage directory, whatever produced it.
    """

    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _resolve(self, key: str) -> Path:
        if not key or key.startswith("/") or "\\" in key:
            raise ReportStorageError(f"invalid storage key: {key!r}")
        candidate = (self.root / key).resolve()
        if not candidate.is_relative_to(self.root):
            raise ReportStorageError(f"storage key escapes the storage root: {key!r}")
        return candidate

    def save(self, *, key: str, data: bytes) -> None:
        path = self._resolve(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    def open(self, *, key: str) -> bytes:
        path = self._resolve(key)
        if not path.is_file():
            raise ReportStorageError(f"no stored file for key: {key!r}")
        return path.read_bytes()

    def delete(self, *, key: str) -> None:
        path = self._resolve(key)
        path.unlink(missing_ok=True)

    def exists(self, *, key: str) -> bool:
        try:
            return self._resolve(key).is_file()
        except ReportStorageError:
            return False


ALLOWED_REPORT_MIME_TYPES = ("application/pdf", "image/png", "image/jpeg")
MAX_REPORT_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB, per product policy


def sanitize_filename(original: str) -> str:
    """Strips any directory component and restricts to a conservative safe
    charset, so a client-supplied filename can never influence where a file
    is written (the storage key's directory structure is always
    server-generated -- see LocalReportStorage's docstring) or contain
    characters that could confuse a downstream shell/filesystem."""

    base = original.replace("\\", "/").split("/")[-1].strip()
    safe = "".join(ch if (ch.isalnum() or ch in "._-") else "_" for ch in base)
    safe = safe.strip("._") or "report"
    return safe[:120]
