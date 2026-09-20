"""Infrastructure-only helpers for Vedant's project work.

This package intentionally contains no clinical, timestamp, split, model, or
evaluation algorithms.
"""

from .hashing import is_sha256, sha256_bytes, sha256_file
from .registry import (
    RegistryValidationError,
    migrate_registry_schema,
    upsert_registry_record,
    validate_registry,
)


_GOVERNANCE_EXPORTS = {
    "G3FreezeError",
    "assert_g3_freeze",
    "is_g3_freeze_valid",
    "audit_g3",
    "freeze_g3",
    "guarded_test_access",
    "invalidate_for_reset",
    "validate_g3_marker",
}


def __getattr__(name):
    """Load governance lazily so low-level hashing has no split import cycle."""
    if name not in _GOVERNANCE_EXPORTS:
        raise AttributeError(name)
    from . import governance

    return getattr(governance, name)

__all__ = [
    "G3FreezeError",
    "RegistryValidationError",
    "assert_g3_freeze",
    "is_g3_freeze_valid",
    "audit_g3",
    "freeze_g3",
    "guarded_test_access",
    "invalidate_for_reset",
    "validate_g3_marker",
    "is_sha256",
    "sha256_bytes",
    "sha256_file",
    "upsert_registry_record",
    "validate_registry",
    "migrate_registry_schema",
]
