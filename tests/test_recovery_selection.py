from evaluation.select import select_task
from selection_helpers import pair


def test_recovery_uses_mae24_even_when_every_secondary_metric_disagrees(tmp_path):
    xgb, gru = pair(
        tmp_path,
        "recovery",
        1.0,
        2.0,
        xgb_secondary={"validation_recovery48_mae": 100.0, "validation_recovery24_rmse": 99.0},
        gru_secondary={"validation_recovery48_mae": 0.1, "validation_recovery24_rmse": 0.2},
    )
    result = select_task(task="recovery", xgboost=xgb, gru=gru)
    assert result.selected.family == "xgboost"
    assert result.primary_metric == "validation_recovery24_mae"

