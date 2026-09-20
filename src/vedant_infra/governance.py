"""Compatibility surface for the authoritative Phase-17 G3 mechanism."""

from pathlib import Path

from vedant_infra.g3 import (
    G3FreezeError,
    REAL_MARKER_RELATIVE_PATH,
    audit_g3,
    freeze_g3,
    guarded_test_access,
    invalidate_for_reset,
    load_access_state,
    require_development_change_allowed,
    validate_g3_marker,
)


def _infer_root(marker_path: Path) -> Path:
    path = marker_path.resolve()
    if path.name == "g3_freeze.json" and path.parent.name == "governance":
        return path.parents[2]
    return path.parent


def is_g3_freeze_valid(path) -> bool:
    marker = Path(path)
    if not marker.is_file():
        return False
    try:
        validate_g3_marker(
            marker, _infer_root(marker), expected_scope="real"
        )
    except G3FreezeError:
        return False
    return True


def assert_g3_freeze(path) -> None:
    marker = Path(path)
    if not marker.is_file():
        raise G3FreezeError(
            "final-test access refused: G3 marker is absent"
        )
    validate_g3_marker(
        marker, _infer_root(marker), expected_scope="real"
    )


__all__ = [
    "G3FreezeError",
    "REAL_MARKER_RELATIVE_PATH",
    "assert_g3_freeze",
    "audit_g3",
    "freeze_g3",
    "guarded_test_access",
    "invalidate_for_reset",
    "is_g3_freeze_valid",
    "load_access_state",
    "require_development_change_allowed",
    "validate_g3_marker",
]
