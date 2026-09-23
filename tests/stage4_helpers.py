"""Shared fixtures/helpers for the Stage-4 real serving integration suite."""

import csv
import json
import shutil
from datetime import datetime
from pathlib import Path

from data.timestamps import RetainedICUStay, generate_prediction_rows_for_stay


ROOT = Path(__file__).resolve().parents[1]


def copy_repository(tmp_path: Path) -> Path:
    """A cheap (APFS copy-on-write) full repository copy for hostile tests
    that must tamper with a real frozen artifact without touching the
    working tree."""

    dest = tmp_path / "repo"
    shutil.copytree(ROOT, dest, ignore=shutil.ignore_patterns(".git"))
    return dest


def safe_train_stay(root: Path = ROOT) -> str:
    split = {}
    with (root / "artifacts/splits/synthetic_split_v2.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            split[row["subject_id"]] = row["split"]
    with (root / "artifacts/data/synthetic/cohorts/final/phase9_final_v1/retained_cohort.jsonl").open() as handle:
        for line in handle:
            row = json.loads(line)
            if split.get(row["subject_id"]) == "train":
                return row["stay_id"]
    raise AssertionError("no train-split stay found")


def sealed_test_stay(root: Path = ROOT) -> str:
    split = {}
    with (root / "artifacts/splits/synthetic_split_v2.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            split[row["subject_id"]] = row["split"]
    with (root / "artifacts/data/synthetic/cohorts/final/phase9_final_v1/retained_cohort.jsonl").open() as handle:
        for line in handle:
            row = json.loads(line)
            if split.get(row["subject_id"]) == "test":
                return row["stay_id"]
    raise AssertionError("no test-split stay found")


def legal_cutoff(stay_id: str, root: Path = ROOT, *, index: int = 0) -> str:
    with (root / "artifacts/data/synthetic/cohorts/final/phase9_final_v1/retained_cohort.jsonl").open() as handle:
        for line in handle:
            row = json.loads(line)
            if row["stay_id"] == stay_id:
                intime = datetime.fromisoformat(row["intime"].replace("Z", "+00:00"))
                outtime = datetime.fromisoformat(row["outtime"].replace("Z", "+00:00"))
                rows = generate_prediction_rows_for_stay(
                    RetainedICUStay(subject_id=row["subject_id"], stay_id=stay_id, intime=intime, outtime=outtime)
                )
                return rows[index].prediction_time.isoformat()
    raise AssertionError("stay not found in retained cohort")
