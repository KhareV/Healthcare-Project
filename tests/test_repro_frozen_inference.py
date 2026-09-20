import pytest

from reproducibility.compare import (
    ReproducibilityComparisonError,
    compare_numeric_sequences,
    validate_reproduction_identity,
)


def identity():
    return {
        "run_id": "run-1",
        "config_hash": "a" * 64,
        "split_hash": "b" * 64,
        "feature_version": "features-v1",
        "label_version": "labels-v1",
        "preprocessor_sha256": "c" * 64,
        "model_sha256": "d" * 64,
    }


def test_frozen_inference_requires_identity_and_approved_tolerance():
    validate_reproduction_identity(identity(), identity())
    compare_numeric_sequences((0.1, 0.9), (0.1, 0.9), tolerance=0.0)
    changed = identity()
    changed["model_sha256"] = "e" * 64
    with pytest.raises(ReproducibilityComparisonError, match="model_sha256"):
        validate_reproduction_identity(identity(), changed)
    with pytest.raises(ReproducibilityComparisonError, match="TOLERANCE REQUIRED"):
        compare_numeric_sequences((0.1,), (0.1,), tolerance=None)
