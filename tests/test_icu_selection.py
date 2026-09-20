from evaluation.select import select_task
from selection_helpers import pair


def test_icu_uses_weighted_median_absolute_error_in_hours(tmp_path):
    xgb, gru = pair(
        tmp_path,
        "icu_stay_time",
        8.0,
        10.0,
        xgb_secondary={"validation_weighted_mae_hours": 100.0},
        gru_secondary={"validation_weighted_mae_hours": 1.0},
    )
    result = select_task(task="icu_stay_time", xgboost=xgb, gru=gru)
    assert result.selected.family == "xgboost"
    assert result.primary_metric == "validation_median_absolute_error_hours"

