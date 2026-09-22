import copy
import json
from pathlib import Path

import pytest

from experiments.stage2_finalization import (
    Stage2ContractError, assert_prediction_parity, choose_family,
    derive_lstm_config, selection_key, validate_exact_run_set, validate_lstm_config,
)
from training.checkpoint import _validate_metadata


ROOT = Path(__file__).resolve().parents[1]


def _config():
    return {"hidden_dim": 128, "num_layers": 1, "dropout": .2, "learning_rate": .001,
            "batch_size": 64, "weight_decay": .0001}


def test_exact_selection_rules_and_family_tie_break():
    recovery_x = {"mae24": 1., "mae48": 2., "rmse24": 3., "rmse48": 4.}
    recovery_g = {"mae24": 1.1, "mae48": 1., "rmse24": 1., "rmse48": 1.}
    assert choose_family("recovery", recovery_x, recovery_g) == "xgboost"
    assert choose_family("recovery", recovery_x, recovery_x) == "gru"
    support_x = {"auprc": .7, "brier": .2, "auroc": .8, "probability_type": "raw_uncalibrated"}
    support_g = {"auprc": .6, "brier": .1, "auroc": .9, "probability_type": "raw_uncalibrated"}
    assert choose_family("organ_support", support_x, support_g) == "xgboost"
    with pytest.raises(Stage2ContractError, match="uncalibrated"):
        selection_key("organ_support", "gru", {**support_g, "probability_type": "calibrated"})


def test_lstm_derivation_changes_only_cell_and_forbids_search():
    source = _config(); derived = derive_lstm_config(source)
    for key, value in source.items():
        assert derived[key] == value
    assert derived["recurrent_cell"] == "lstm"
    assert derived["participates_in_search"] is False
    validate_lstm_config(source, derived)
    bad = dict(derived); bad["candidates"] = []
    with pytest.raises(Stage2ContractError):
        validate_lstm_config(source, bad)


def test_exactly_one_lstm_per_task_guard():
    rows = [{"task": task, "family": "lstm", "sensitivity_only": True,
             "participates_in_search": False, "test_accessed": False}
            for task in ("recovery", "icu_time", "organ_support")]
    validate_exact_run_set(rows)
    with pytest.raises(Stage2ContractError, match="exactly one"):
        validate_exact_run_set(rows + [copy.deepcopy(rows[0])])


def test_authoritative_prediction_population_parity():
    pairs = {
        "recovery": (
            [ROOT / "artifacts/search/xgb/phase12/recovery/xgb-recovery-014/attempt-1/validation_predictions_24h.jsonl",
             ROOT / "artifacts/search/xgb/phase12/recovery/xgb-recovery-014/attempt-1/validation_predictions_48h.jsonl"],
            ROOT / "artifacts/search/gru/final_v2/recovery/gru-recovery-024/attempt-1/validation_predictions.jsonl"),
        "icu_time": ([ROOT / "artifacts/search/xgb/phase12/icu_time/xgb-icu-time-028/attempt-1/validation_predictions.jsonl"],
                     ROOT / "artifacts/search/gru/final_v2/icu_time/gru-icu-time-026/attempt-1/validation_predictions.jsonl"),
        "organ_support": ([ROOT / "artifacts/search/xgb/phase12/organ_support/xgb-support-024/attempt-1/validation_predictions.jsonl"],
                          ROOT / "artifacts/search/gru/final_v2/organ_support/gru-support-001/attempt-1/validation_predictions.jsonl"),
    }
    for task, (xgb, gru) in pairs.items():
        assert assert_prediction_parity(task, xgb, gru)["status"] == "PASS"


def test_frozen_sources_are_validation_only_and_v2_is_authoritative():
    xgb = json.loads((ROOT / "artifacts/search/xgb/phase12/best_xgb_candidates_v1.json").read_text())
    gru = json.loads((ROOT / "artifacts/search/gru/final_v2/best_gru_candidates_v1.json").read_text())
    master = json.loads((ROOT / "artifacts/search/gru/final_v2/manifests/search_manifest_v1.json").read_text())
    assert xgb["test_accessed"] is False and gru["test_accessed"] is False
    assert gru["search_version"] == "vedant_final_gru_validation_search_v2"
    assert master["terminal_candidate_counts"] == {task: {"complete": 30, "failed": 0}
                                                    for task in ("recovery", "icu_time", "organ_support")}
    assert all("/final_v2/" in item["checkpoint_path"] for item in gru["tasks"].values())


def test_checkpoint_accepts_scientific_sensitivity_as_exclusive_role():
    metadata = {"run_id": "r", "task": "recovery", "model_family": "lstm", "seed": 1,
                "epoch": 1, "config_hash": "a" * 64, "tensor_contract_version": "t",
                "feature_schema_version": "f", "split_hash": "b" * 64, "code_commit": "c" * 40,
                "validation_value": 1.0, "synthetic_smoke_test": False,
                "synthetic_scientific_search": False, "synthetic_scientific_sensitivity": True}
    _validate_metadata(metadata)
    metadata["synthetic_scientific_search"] = True
    with pytest.raises(Exception, match="exactly one"):
        _validate_metadata(metadata)
