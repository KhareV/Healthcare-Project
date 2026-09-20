from pathlib import Path

import pytest

from evaluation.error_analysis import (
    ErrorAnalysisValidationError,
    load_prediction_artifact,
)


def test_pre_phase20_rejects_test_manifest_before_reading_sentinel(tmp_path):
    sentinel = tmp_path / "test_predictions.json"
    sentinel.write_text("DO_NOT_READ_TEST_SENTINEL", encoding="utf-8")
    with pytest.raises(ErrorAnalysisValidationError, match="DEFERRED UNTIL PHASE 20"):
        load_prediction_artifact(
            {"partition": "test", "prediction_ref": str(sentinel), "test_accessed": False}
        )
    assert sentinel.read_text(encoding="utf-8") == "DO_NOT_READ_TEST_SENTINEL"


def test_validation_loader_rejects_nonvalidation_payload(tmp_path):
    path = Path(tmp_path) / "wrong.json"
    path.write_text('{"partition": "train"}', encoding="utf-8")
    with pytest.raises(ErrorAnalysisValidationError, match="only validation"):
        load_prediction_artifact({"partition": "validation", "prediction_ref": str(path)})
