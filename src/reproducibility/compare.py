"""Fail-closed comparisons for frozen hashes and numeric outputs."""

import math
from pathlib import Path
from typing import Mapping, Optional, Sequence

from vedant_infra.hashing import is_sha256, sha256_file


class ReproducibilityComparisonError(ValueError):
    """Raised when two objects cannot represent the same frozen result."""


def verify_file_hash(path: Path, expected_sha256: str) -> str:
    if not is_sha256(expected_sha256):
        raise ReproducibilityComparisonError("expected artifact hash is malformed")
    if not path.is_file() or sha256_file(path) != expected_sha256:
        raise ReproducibilityComparisonError("frozen artifact hash mismatch")
    return expected_sha256


def compare_numeric_sequences(
    expected: Sequence[float],
    reproduced: Sequence[float],
    *,
    tolerance: Optional[float],
) -> None:
    if tolerance is None:
        raise ReproducibilityComparisonError(
            "BLOCKED — RETRAINING REPRODUCIBILITY TOLERANCE REQUIRED"
        )
    if not math.isfinite(float(tolerance)) or float(tolerance) < 0:
        raise ReproducibilityComparisonError("numeric tolerance must be finite and nonnegative")
    if len(expected) != len(reproduced):
        raise ReproducibilityComparisonError("reproduced output length mismatch")
    for left, right in zip(expected, reproduced):
        if not math.isfinite(float(left)) or not math.isfinite(float(right)):
            raise ReproducibilityComparisonError("reproduced outputs must be finite")
        if abs(float(left) - float(right)) > float(tolerance):
            raise ReproducibilityComparisonError("reproduced numeric output differs")


def validate_reproduction_identity(
    expected: Mapping[str, object], reproduced: Mapping[str, object]
) -> None:
    fields = (
        "run_id",
        "config_hash",
        "split_hash",
        "feature_version",
        "label_version",
        "preprocessor_sha256",
        "model_sha256",
    )
    for field in fields:
        if expected.get(field) != reproduced.get(field):
            raise ReproducibilityComparisonError(
                "reproduction identity mismatch: " + field
            )
