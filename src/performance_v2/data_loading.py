"""TRAIN/VALIDATION-only row loading for Performance-v2 Phase 1.

ABSOLUTE RULE: every function here refuses to return a row whose ``split``
field is anything other than ``"train"`` or ``"validation"``. There is no
parameter to opt into ``"test"``. This is the single choke point Phase-1
diagnostics load rows through, specifically so v1 TEST rows can never
leak into feature engineering, baseline fitting, oracle training, or
error-analysis code by accident.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterator, Mapping, Tuple

ALLOWED_SPLITS = ("train", "validation")

PRE_SPLIT_PATH = "artifacts/data/synthetic/pre_split/final/phase9_final_v1/pre_split_scientific_package.jsonl"
STATICS_PATH = "artifacts/data/synthetic/timelines/final/phase9_final_v1/canonical_statics.jsonl"
SPLIT_PATH = "artifacts/splits/synthetic_split_v2.csv"


def _utc(value: str) -> datetime:
    return datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(timezone.utc)


def load_jsonl(path: Path) -> Iterator[Mapping[str, object]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_statics(root: Path) -> Dict[str, Mapping[str, object]]:
    """stay_id -> {age_years, sex_category, cardiac_condition_group, intime, outtime}."""

    return {row["stay_id"]: row for row in load_jsonl(root / STATICS_PATH)}


def load_subject_splits(root: Path) -> Dict[str, str]:
    """subject_id -> split, from the frozen, G3-bound split assignment file.

    The pre-split scientific package's own ``split`` field is always null
    (it predates split assignment, hence the name); the authoritative split
    membership lives only in artifacts/splits/synthetic_split_v2.csv.
    """

    with (root / SPLIT_PATH).open(newline="", encoding="utf-8") as handle:
        return {row["subject_id"]: row["split"] for row in csv.DictReader(handle)}


def load_dev_rows(root: Path, *, splits: Tuple[str, ...] = ALLOWED_SPLITS) -> Tuple[Mapping[str, object], ...]:
    """Every prediction-cutoff row for the requested TRAIN/VALIDATION splits,
    enriched with lawful structural fields derivable at the cutoff itself:

    - elapsed_episode_hours_at_t = (prediction_time - intime) in hours.
      Both prediction_time and intime are known at the cutoff; this never
      uses outtime.
    - cutoff_index = the row's own grid_index (already lawful/known: it is
      the ordinal position of this prediction cutoff, not a future value).
    - hours_since_first_eligible_cutoff = elapsed_episode_hours_at_t - 24
      (the contract's first eligible cutoff is always +24h).

    Raises if any requested split is outside ALLOWED_SPLITS, or if any
    loaded row's own ``split`` field is not in ALLOWED_SPLITS -- this is the
    enforcement point for the "never touch v1 TEST" rule.
    """

    for split in splits:
        if split not in ALLOWED_SPLITS:
            raise ValueError("load_dev_rows only allows train/validation splits, got: " + repr(split))

    statics = load_statics(root)
    subject_splits = load_subject_splits(root)
    rows = []
    for row in load_jsonl(root / PRE_SPLIT_PATH):
        split = subject_splits.get(row["subject_id"])
        if split not in splits:
            continue
        if split not in ALLOWED_SPLITS:
            raise RuntimeError("refusing non-train/validation row inside load_dev_rows: split=" + repr(split))
        static = statics[row["stay_id"]]
        intime = _utc(static["intime"])
        prediction_time = _utc(row["prediction_time"])
        elapsed_hours = (prediction_time - intime).total_seconds() / 3600.0
        enriched = dict(row)
        enriched["split"] = split
        enriched["elapsed_episode_hours_at_t"] = elapsed_hours
        enriched["cutoff_index"] = int(row["grid_index"])
        enriched["hours_since_first_eligible_cutoff"] = elapsed_hours - 24.0
        enriched["age_years"] = static["age_years"]
        enriched["sex_category"] = static["sex_category"]
        enriched["cardiac_condition_group"] = static["cardiac_condition_group"]
        enriched["intime"] = static["intime"]
        rows.append(enriched)
    return tuple(rows)


def assert_no_test_rows(rows) -> None:
    bad = {row.get("split") for row in rows} - set(ALLOWED_SPLITS)
    if bad:
        raise RuntimeError("v1 TEST-partition rows detected where only TRAIN/VALIDATION are permitted: " + repr(bad))
