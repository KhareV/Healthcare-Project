from pathlib import Path

import pytest

from evaluation.ablations import (
    AblationValidationError,
    derive_feature_manifest,
    evaluate_recovery_ablation,
    reject_ablation_test_fields,
)
from evaluation.metrics import PredictionRecord
from evaluation.selection_validation import sha256_file
from ablation_helpers import contract, variant


def test_test_sentinel_reference_is_rejected_without_reading_file(tmp_path):
    sentinel = Path(tmp_path) / "fake_test_predictions.json"
    sentinel.write_text("DO_NOT_READ_TEST_SENTINEL", encoding="utf-8")
    with pytest.raises(AblationValidationError, match="TEST-LABELLED FIELD") as caught:
        reject_ablation_test_fields({"test_prediction_artifact": str(sentinel)})
    assert "DO_NOT_READ_TEST_SENTINEL" not in str(caught.value)


def test_evaluator_requires_validation_partition_attestation(tmp_path):
    candidate = variant(tmp_path)
    manifest = derive_feature_manifest(candidate, contract())
    artifact = Path(tmp_path) / "ablation.model"
    artifact.write_bytes(b"synthetic")
    rows = (PredictionRecord("A", "2020-01-01", 1.0, 0.5, True),)
    with pytest.raises(AblationValidationError, match="TEST ACCESS"):
        evaluate_recovery_ablation(
            variant=candidate,
            manifest=manifest,
            run_id="attack",
            run_artifact_ref=str(artifact),
            run_artifact_sha256=sha256_file(artifact),
            recovery24=rows,
            recovery48=rows,
            source_partition="test",
            test_accessed=False,
        )
