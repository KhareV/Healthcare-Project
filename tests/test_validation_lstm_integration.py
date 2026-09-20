import pytest

from experiments.validation_suite import (
    ValidationSuiteError,
    audit_lstm_sensitivity,
    audit_selected_gru_reruns,
)
from validation_suite_helpers import TASKS, complete_reruns


def test_one_selected_gru_rerun_and_one_lstm_per_task_pass():
    reruns = complete_reruns("selected_config_rerun", "gru", "search-")
    audit_selected_gru_reruns(reruns, {task: "search-" + task for task in TASKS})
    lstm = complete_reruns("sensitivity", "lstm", "selected_config_rerun-")
    audit_lstm_sensitivity(
        lstm, {task: "selected_config_rerun-" + task for task in TASKS}
    )


def test_extra_lstm_or_candidate_identity_fails():
    rows = complete_reruns("sensitivity", "lstm", "rerun-")
    rows.append(dict(rows[0], run_id="extra"))
    with pytest.raises(ValidationSuiteError, match="exactly one LSTM"):
        audit_lstm_sensitivity(rows, {task: "rerun-" + task for task in TASKS})
