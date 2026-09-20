import pytest

from evaluation.select import select_task
from evaluation.selection_validation import SelectionValidationError
from selection_helpers import pair, replace_candidate


def test_support_uses_raw_auprc_not_auroc_or_f1(tmp_path):
    xgb, gru = pair(
        tmp_path,
        "organ_support",
        0.31,
        0.40,
        xgb_secondary={"validation_auroc": 0.99, "validation_f1": 0.90},
        gru_secondary={"validation_auroc": 0.51, "validation_f1": 0.20},
    )
    result = select_task(task="organ_support", xgboost=xgb, gru=gru)
    assert result.selected.family == "gru"
    assert result.primary_metric == "validation_auprc"


def test_calibrated_support_candidate_cannot_enter_selection(tmp_path):
    xgb, gru = pair(tmp_path, "organ_support", 0.31, 0.40)
    with pytest.raises(SelectionValidationError, match="raw uncalibrated"):
        select_task(
            task="organ_support",
            xgboost=replace_candidate(xgb, probability_type="calibrated"),
            gru=gru,
        )

