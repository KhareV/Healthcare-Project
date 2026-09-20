import pytest

from evaluation.select import select_task
from evaluation.selection_validation import SelectionValidationError
from selection_helpers import information, pair, replace_candidate


def test_feature_information_mismatch_blocks_selection(tmp_path):
    xgb, gru = pair(tmp_path, "recovery", 1.0, 2.0)
    gru = replace_candidate(
        gru,
        information_manifest=information("gru", feature_schema_version="incompatible"),
    )
    with pytest.raises(SelectionValidationError, match="INFORMATION MISMATCH"):
        select_task(task="recovery", xgboost=xgb, gru=gru)


def test_split_hash_mismatch_blocks_selection(tmp_path):
    xgb, gru = pair(tmp_path, "recovery", 1.0, 2.0)
    gru = replace_candidate(gru, split_hash="other-split")
    with pytest.raises(SelectionValidationError, match="split_hash"):
        select_task(task="recovery", xgboost=xgb, gru=gru)


def test_validation_population_mismatch_blocks_selection(tmp_path):
    xgb, gru = pair(tmp_path, "organ_support", 0.3, 0.4)
    counts = dict(gru.counts)
    counts["N_ICU_stays"] -= 1
    gru = replace_candidate(gru, counts=counts)
    with pytest.raises(SelectionValidationError, match="population parity"):
        select_task(task="organ_support", xgboost=xgb, gru=gru)


def test_metric_implementation_mismatch_blocks_selection(tmp_path):
    xgb, gru = pair(tmp_path, "recovery", 1.0, 2.0)
    gru = replace_candidate(gru, metric_implementation_version="custom_notebook_metric")
    with pytest.raises(SelectionValidationError, match="Phase-9 metric"):
        select_task(task="recovery", xgboost=xgb, gru=gru)

