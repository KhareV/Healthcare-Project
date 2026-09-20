"""Auditable fixed GRU-to-LSTM comparison-config derivation."""

import json
from pathlib import Path
from typing import Mapping, Union

from vedant_infra.hashing import sha256_file


PathLike = Union[str, Path]
MAPPED_TRAINING_FIELDS = (
    "seed",
    "batch_size",
    "num_workers",
    "max_epochs",
    "device",
    "optimizer",
    "early_stopping",
)
FORBIDDEN_SEARCH_KEYS = {
    "search_space",
    "trial_count",
    "n_trials",
    "grid_search",
    "random_search",
    "optuna",
    "candidate_values",
    "configuration_budget",
}
ALLOWED_FIXED_SEQUENCE_FIELDS = {"feature_names", "horizon_order"}


class LSTMComparisonConfigError(ValueError):
    """Raised when fixed-comparison provenance or parity is violated."""


def derive_lstm_comparison_config(
    source_gru_config: Mapping[str, object],
    *,
    source_gru_config_ref: str,
    source_gru_config_hash: str,
    source_gru_checkpoint_ref: str,
    source_gru_checkpoint_hash: str,
    lstm_run_id: str,
    output_directory: str,
) -> Mapping[str, object]:
    """Map every meaningful smoke GRU setting and change only model family/cell."""

    if source_gru_config.get("model_family") != "gru":
        raise LSTMComparisonConfigError("source model family must be gru")
    if source_gru_config.get("synthetic_smoke_test") is not True:
        raise LSTMComparisonConfigError("pre-selection derivation requires a synthetic GRU config")
    mapped_training = {
        field: source_gru_config[field] for field in MAPPED_TRAINING_FIELDS
    }
    return {
        "config_version": "{}_derived_lstm_v1".format(source_gru_config["config_version"]),
        "status": "SYNTHETIC_LSTM_SENSITIVITY_SMOKE_NOT_SCIENTIFIC_RUN",
        "run_id": lstm_run_id,
        "task": source_gru_config["task"],
        "model_family": "lstm",
        "recurrent_cell": "lstm",
        "sensitivity_only": True,
        "scientific_sensitivity": False,
        "source_selection_status": "DEVELOPMENT_SMOKE_NOT_SELECTED",
        "source_selected_gru_run_id": None,
        "source_gru_run_id": source_gru_config["run_id"],
        "source_gru_config_ref": source_gru_config_ref,
        "source_gru_config_hash": source_gru_config_hash,
        "source_gru_checkpoint_ref": source_gru_checkpoint_ref,
        "source_gru_checkpoint_hash": source_gru_checkpoint_hash,
        "model": dict(source_gru_config["model"]),
        "mapped_training": mapped_training,
        "fixture_path": source_gru_config["fixture_path"],
        "synthetic_split_path": source_gru_config["synthetic_split_path"],
        "dataset_sha256": sha256_file(source_gru_config["fixture_path"]),
        "split_sha256": sha256_file(source_gru_config["synthetic_split_path"]),
        "tensor_contract_path": source_gru_config["tensor_contract_path"],
        "tensor_contract_version": source_gru_config["tensor_contract_version"],
        "feature_schema_version": source_gru_config["feature_schema_version"],
        "feature_names": list(source_gru_config["feature_names"]),
        "feature_schema_status": source_gru_config["feature_schema_status"],
        "synthetic_missing_fill_value": source_gru_config["synthetic_missing_fill_value"],
        "seed_policy": source_gru_config.get("seed_status"),
        "label_version": source_gru_config.get("label_version"),
        "support_label_path": source_gru_config.get("support_label_path"),
        "preprocessing": None,
        "task_artifact": None,
        "unmapped_parameters": {},
        "search": None,
        "participates_in_lstm_search": False,
        "participates_in_serving_selection": False,
        "test_examples_allowed": False,
        "output_directory": output_directory,
        "checkpoint_filename": "{}_lstm_smoke.pt".format(source_gru_config["task"]),
        "log_filename": "training_log.jsonl",
        "metrics_filename": "smoke_metrics.json",
        "registry_path": source_gru_config["registry_path"],
        "code_commit": source_gru_config["code_commit"],
    }


def assert_no_lstm_search_space(config: Mapping[str, object]) -> None:
    def walk(value, path="root"):
        if isinstance(value, Mapping):
            for key, child in value.items():
                if key.lower() in FORBIDDEN_SEARCH_KEYS:
                    raise LSTMComparisonConfigError(
                        "LSTM comparison contains forbidden search key: {}".format(path + "." + key)
                    )
                if key in ALLOWED_FIXED_SEQUENCE_FIELDS and isinstance(child, list):
                    continue
                walk(child, path + "." + key)
        elif isinstance(value, list):
            raise LSTMComparisonConfigError(
                "fixed LSTM comparison cannot contain candidate arrays: {}".format(path)
            )
    walk(config)
    if config.get("search") is not None or config.get("participates_in_lstm_search") is not False:
        raise LSTMComparisonConfigError("LSTM search must be disabled")


def validate_comparison_against_source(
    comparison: Mapping[str, object], source: Mapping[str, object]
) -> None:
    assert_no_lstm_search_space(comparison)
    if comparison.get("model_family") != "lstm" or comparison.get("recurrent_cell") != "lstm":
        raise LSTMComparisonConfigError("comparison must change recurrent cell to LSTM")
    if comparison.get("sensitivity_only") is not True:
        raise LSTMComparisonConfigError("LSTM must be sensitivity-only")
    if comparison.get("scientific_sensitivity") is not False:
        raise LSTMComparisonConfigError("smoke comparison cannot be scientific")
    if comparison.get("source_selected_gru_run_id") is not None:
        raise LSTMComparisonConfigError("smoke GRU cannot be called selected")
    if comparison.get("model") != source.get("model"):
        raise LSTMComparisonConfigError("meaningful model hyperparameters changed")
    expected_training = {field: source[field] for field in MAPPED_TRAINING_FIELDS}
    if comparison.get("mapped_training") != expected_training:
        raise LSTMComparisonConfigError("meaningful training hyperparameters changed")
    for field in (
        "task",
        "fixture_path",
        "synthetic_split_path",
        "tensor_contract_version",
        "feature_schema_version",
        "feature_names",
        "synthetic_missing_fill_value",
    ):
        if comparison.get(field) != source.get(field):
            raise LSTMComparisonConfigError("comparison changed {}".format(field))
    if comparison.get("dataset_sha256") != sha256_file(source["fixture_path"]):
        raise LSTMComparisonConfigError("comparison dataset hash mismatch")
    if comparison.get("split_sha256") != sha256_file(source["synthetic_split_path"]):
        raise LSTMComparisonConfigError("comparison split hash mismatch")
    if comparison.get("seed_policy") != source.get("seed_status"):
        raise LSTMComparisonConfigError("comparison seed policy mismatch")


def load_and_validate_comparison(path: PathLike):
    comparison_path = Path(path)
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    source_path = Path(comparison["source_gru_config_ref"])
    if sha256_file(source_path) != comparison["source_gru_config_hash"]:
        raise LSTMComparisonConfigError("source GRU config hash mismatch")
    source = json.loads(source_path.read_text(encoding="utf-8"))
    validate_comparison_against_source(comparison, source)
    return comparison, source
