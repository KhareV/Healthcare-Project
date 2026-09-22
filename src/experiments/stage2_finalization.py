"""Governed Stage-2 family selection and fixed LSTM sensitivity contracts."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Mapping, Sequence


STAGE2_VERSION = "vedant_finalization_stage2_v1"
TASKS = ("recovery", "icu_time", "organ_support")
EXPECTED_GRU_WINNERS = {
    "recovery": "gru-recovery-024",
    "icu_time": "gru-icu-time-026",
    "organ_support": "gru-support-001",
}
EXPECTED_XGB_WINNERS = {
    "recovery": "xgb-recovery-014",
    "icu_time": "xgb-icu-time-028",
    "organ_support": "xgb-support-024",
}
EXPECTED_G1_SHA256 = "1fbcc1b0859c0052eb867aec15e89bd3f7df503ac318b30f92dd98ff9eeccf90"
EXPECTED_PHASE14_SHA256 = "3c4fa9a9efd25145ab1daa23c03632aa0cfd0b04d3ee0b4254c1cb180fff63d4"
EXPECTED_GRU_BEST_SHA256 = "aea37e8d85b6894d5efd33b1a7d2e1969f683f02b8cef222d3504783d5917254"
MEANINGFUL_GRU_KEYS = (
    "hidden_dim", "num_layers", "dropout", "learning_rate", "batch_size", "weight_decay"
)


class Stage2ContractError(RuntimeError):
    """Raised when Stage-2 authority, parity, or fixed-run rules are violated."""


def canonical_sha256(value: object) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def selection_key(task: str, family: str, metrics: Mapping[str, object]):
    """Return the exact validation-only family ranking key, including tie break."""
    if family not in ("gru", "xgboost"):
        raise Stage2ContractError("serving-family competition is restricted to XGBoost and GRU")
    if task == "recovery":
        return (metrics["mae24"], metrics["mae48"], metrics["rmse24"], metrics["rmse48"], family)
    if task == "icu_time":
        return (metrics["median_absolute_error_hours"], metrics["mae_hours"], metrics["rmse_hours"], family)
    if task == "organ_support":
        if metrics.get("probability_type", "raw_uncalibrated") != "raw_uncalibrated":
            raise Stage2ContractError("support family selection requires raw uncalibrated probabilities")
        return (-metrics["auprc"], metrics["brier"], -metrics["auroc"], family)
    raise Stage2ContractError("unsupported Stage-2 task")


def choose_family(task: str, xgb_metrics: Mapping[str, object], gru_metrics: Mapping[str, object]):
    rows = (("xgboost", xgb_metrics), ("gru", gru_metrics))
    return min(rows, key=lambda row: selection_key(task, row[0], row[1]))[0]


def derive_lstm_config(gru_candidate: Mapping[str, object]) -> Mapping[str, object]:
    """Change only the recurrent cell while preserving the best-GRU settings."""
    missing = [key for key in MEANINGFUL_GRU_KEYS if key not in gru_candidate]
    if missing:
        raise Stage2ContractError("best-GRU config missing: " + ", ".join(missing))
    result = {key: gru_candidate[key] for key in MEANINGFUL_GRU_KEYS}
    result.update({
        "family": "lstm",
        "recurrent_cell": "lstm",
        "optimizer": "adamw",
        "max_epochs": 60,
        "early_stopping_patience": 8,
        "huber_delta": 1.0,
        "include_observation_mask": True,
        "include_tslo": True,
        "unidirectional": True,
        "recurrent_dropout": 0.0,
        "sensitivity_only": True,
        "participates_in_search": False,
        "test_accessed": False,
    })
    return result


def validate_lstm_config(gru_candidate: Mapping[str, object], lstm: Mapping[str, object]) -> None:
    expected = derive_lstm_config(gru_candidate)
    if dict(lstm) != dict(expected):
        raise Stage2ContractError("LSTM config is not the exact fixed-cell best-GRU derivative")
    forbidden = ("search_space", "candidates", "candidate_grid", "trials", "tuning")
    if any(key in lstm for key in forbidden):
        raise Stage2ContractError("LSTM sensitivity cannot contain search controls")


def prediction_key(row: Mapping[str, object]):
    return (row["subject_id"], row["stay_id"], row["prediction_time"], row["grid_index"])


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        return tuple(json.loads(line) for line in handle if line.strip())


def assert_prediction_parity(task: str, xgb_paths: Sequence[Path], gru_path: Path) -> Mapping[str, object]:
    """Verify row keys, targets, and eligibility rather than trusting differing serializers."""
    gru_rows = read_jsonl(gru_path)
    if task == "recovery":
        if len(xgb_paths) != 2:
            raise Stage2ContractError("recovery parity requires the two independent XGB horizons")
        by_horizon = {row["horizon"]: row for row in gru_rows}
        # Duplicate keys are expected across independent horizons, never within one horizon.
        grouped = {h: {} for h in ("24h", "48h")}
        for row in gru_rows:
            key = prediction_key(row)
            if key in grouped[row["horizon"]]:
                raise Stage2ContractError("duplicate GRU recovery validation key")
            grouped[row["horizon"]][key] = row
        counts = {}
        for horizon, path in zip(("24h", "48h"), xgb_paths):
            xgb = read_jsonl(path)
            xmap = {prediction_key(row): row for row in xgb}
            if len(xmap) != len(xgb) or set(xmap) != set(grouped[horizon]):
                raise Stage2ContractError("XGB/GRU recovery validation row-key mismatch")
            for key, xrow in xmap.items():
                grow = grouped[horizon][key]
                if (bool(xrow["eligible"]) != bool(grow["eligible"])
                        or xrow.get("true_target") != grow.get("true_target")):
                    raise Stage2ContractError("XGB/GRU recovery target or eligibility mismatch")
            counts[horizon] = len(xmap)
        return {"status": "PASS", "row_counts": counts}
    if len(xgb_paths) != 1:
        raise Stage2ContractError("single-output parity requires one XGB prediction file")
    xgb = read_jsonl(xgb_paths[0])
    xmap, gmap = {prediction_key(row): row for row in xgb}, {prediction_key(row): row for row in gru_rows}
    if len(xmap) != len(xgb) or len(gmap) != len(gru_rows) or set(xmap) != set(gmap):
        raise Stage2ContractError("XGB/GRU validation row-key mismatch")
    for key, xrow in xmap.items():
        grow = gmap[key]
        target_fields = ("true_target", "true_target_log1p") if task == "icu_time" else ("true_target",)
        if bool(xrow["eligible"]) != bool(grow["eligible"]):
            raise Stage2ContractError("XGB/GRU eligibility mismatch")
        for field in target_fields:
            left, right = xrow.get(field), grow.get(field)
            same = left == right
            if task == "icu_time" and left is not None and right is not None:
                # Phase-11 XGBoost rows preserve the same target after float32
                # flattening; GRU rows preserve the upstream JSON precision.
                tolerance = 1e-4 if field == "true_target" else 1e-6
                same = math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=tolerance)
            if not same:
                raise Stage2ContractError("XGB/GRU target mismatch: " + field)
    return {"status": "PASS", "row_count": len(xmap)}


def validate_exact_run_set(entries: Sequence[Mapping[str, object]]) -> None:
    if len(entries) != 3 or {entry.get("task") for entry in entries} != set(TASKS):
        raise Stage2ContractError("exactly one LSTM sensitivity run is required per task")
    for entry in entries:
        if (entry.get("family") != "lstm" or entry.get("sensitivity_only") is not True
                or entry.get("participates_in_search") is not False
                or entry.get("test_accessed") is not False):
            raise Stage2ContractError("LSTM scientific sensitivity governance violation")
