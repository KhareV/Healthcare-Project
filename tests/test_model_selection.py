from dataclasses import replace

import pytest

from evaluation.select import select_task
from evaluation.selection_validation import SelectionValidationError
from selection_helpers import pair, replace_candidate


def test_only_best_of_xgboost_and_gru_can_be_selected(tmp_path):
    xgb, gru = pair(tmp_path, "recovery", 1.2, 1.4)
    result = select_task(task="recovery", xgboost=xgb, gru=gru)
    assert result.selected.family == "xgboost"
    assert result.selected.family not in {"naive", "lstm"}


@pytest.mark.parametrize("forbidden_family", ["naive", "lstm"])
def test_reference_and_sensitivity_families_rejected_from_family_slots(tmp_path, forbidden_family):
    xgb, gru = pair(tmp_path, "recovery", 1.2, 1.4)
    attacked = replace_candidate(xgb, family=forbidden_family)
    with pytest.raises(SelectionValidationError, match="fixed family slots"):
        select_task(task="recovery", xgboost=attacked, gru=gru)


def test_exact_cross_family_tie_blocks_until_policy_is_frozen(tmp_path):
    xgb, gru = pair(tmp_path, "recovery", 1.2, 1.2)
    with pytest.raises(SelectionValidationError, match="TIE POLICY REQUIRED"):
        select_task(task="recovery", xgboost=xgb, gru=gru)


def test_unfrozen_metric_tolerance_is_rejected(tmp_path):
    xgb, gru = pair(tmp_path, "recovery", 1.2, 1.3)
    with pytest.raises(SelectionValidationError, match="METRIC TIE TOLERANCE"):
        select_task(task="recovery", xgboost=xgb, gru=gru, tie_tolerance=1e-6)


def test_manually_mistyped_within_family_winner_is_recomputed_and_rejected(tmp_path):
    xgb, gru = pair(tmp_path, "recovery", 1.2, 1.3)
    wrong = xgb.search.results[1]
    attacked_search = replace(
        xgb.search,
        referenced_best_candidate_id=wrong.candidate_id,
    )
    with pytest.raises(SelectionValidationError, match="recomputed best"):
        select_task(
            task="recovery",
            xgboost=replace_candidate(xgb, search=attacked_search),
            gru=gru,
        )


def test_incomplete_or_outside_exact_30_search_is_rejected(tmp_path):
    xgb, gru = pair(tmp_path, "recovery", 1.2, 1.3)
    incomplete = replace(xgb.search, results=xgb.search.results[:-1])
    with pytest.raises(SelectionValidationError, match="COMPLETED SCIENTIFIC"):
        select_task(
            task="recovery",
            xgboost=replace_candidate(xgb, search=incomplete),
            gru=gru,
        )
    with pytest.raises(SelectionValidationError, match="config hash"):
        select_task(
            task="recovery",
            xgboost=replace_candidate(xgb, config_hash="not-in-frozen-search"),
            gru=gru,
        )


def test_summary_metric_cannot_be_manually_changed_from_frozen_result(tmp_path):
    xgb, gru = pair(tmp_path, "recovery", 1.2, 1.3)
    with pytest.raises(SelectionValidationError, match="metric does not match"):
        select_task(
            task="recovery",
            xgboost=replace_candidate(
                xgb,
                metrics={"validation_recovery24_mae": 0.0001},
            ),
            gru=gru,
        )
