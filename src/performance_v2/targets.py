"""Shared target/eligibility extraction for Performance-v2 Phase 1 rows.

Row shape is exactly what performance_v2.data_loading.load_dev_rows returns
(TRAIN/VALIDATION only). All functions here are pure/read-only.
"""

from __future__ import annotations

import math
from typing import Mapping, Sequence, Tuple

from evaluation.metrics import PredictionRecord


def recovery_targets(rows: Sequence[Mapping[str, object]], horizon: str) -> Tuple[Mapping[str, object], ...]:
    """Eligible rows for one recovery horizon, target = raw unclipped delta SOFA."""

    key_target = "delta_sofa_" + horizon
    key_eligible = "recovery" + horizon + "_eligible"
    return tuple(row for row in rows if row.get(key_eligible) and row.get(key_target) is not None)


def icu_hours_target(row: Mapping[str, object]) -> float:
    """Ground-truth remaining ICU-like hours, exactly invertible from the
    frozen log1p training target (no clamping needed: the stored value is
    already a valid non-negative log1p transform of a real duration)."""

    return math.expm1(float(row["icu_time_log1p"]))


def icu_rows(rows: Sequence[Mapping[str, object]]) -> Tuple[Mapping[str, object], ...]:
    return tuple(row for row in rows if row.get("icu_time_eligible") and row.get("icu_time_log1p") is not None)


def support_rows(rows: Sequence[Mapping[str, object]]) -> Tuple[Mapping[str, object], ...]:
    return tuple(row for row in rows if row.get("organ_support_eligible") and row.get("organ_support_label") is not None)


def make_records(rows: Sequence[Mapping[str, object]], targets: Sequence[float], predictions: Sequence[float]) -> Tuple[PredictionRecord, ...]:
    assert len(rows) == len(targets) == len(predictions)
    return tuple(
        PredictionRecord(stay_id=row["stay_id"], prediction_time=row["prediction_time"], target=target, prediction=prediction, eligible=True)
        for row, target, prediction in zip(rows, targets, predictions)
    )
