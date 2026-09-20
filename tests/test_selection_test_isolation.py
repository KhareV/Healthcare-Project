from pathlib import Path

import pytest

from evaluation.select import select_task, selection_report
from evaluation.selection_validation import SelectionValidationError, reject_test_fields
from selection_helpers import pair, replace_candidate


def test_test_labelled_fields_are_structurally_rejected_without_reading_artifact(tmp_path):
    sentinel = Path(tmp_path) / "forbidden.txt"
    sentinel.write_text("DO_NOT_READ_TEST_SENTINEL", encoding="utf-8")
    with pytest.raises(SelectionValidationError, match="TEST-LABELLED FIELD") as caught:
        reject_test_fields({"test_artifact_ref": str(sentinel)})
    assert "DO_NOT_READ_TEST_SENTINEL" not in str(caught.value)


def test_test_access_attestation_and_partition_are_hard_failures(tmp_path):
    xgb, gru = pair(tmp_path, "recovery", 1.0, 2.0)
    with pytest.raises(SelectionValidationError, match="TEST ACCESS FORBIDDEN"):
        select_task(
            task="recovery",
            xgboost=replace_candidate(xgb, test_accessed=True),
            gru=gru,
        )
    with pytest.raises(SelectionValidationError, match="TEST ACCESS FORBIDDEN"):
        select_task(
            task="recovery",
            xgboost=replace_candidate(xgb, source_partition="test"),
            gru=gru,
        )


def test_selection_report_contains_validation_evidence_only(tmp_path):
    xgb, gru = pair(tmp_path, "recovery", 1.0, 2.0)
    report = selection_report(select_task(task="recovery", xgboost=xgb, gru=gru))
    serialized = str(report).lower()
    assert "test_metric" not in serialized
    assert report["test_accessed"] is False

