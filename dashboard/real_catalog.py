"""Real (train/validation-safe) demo catalog for the Stage-4 replay dashboard.

Builds a small, deterministic set of demo ``ReplayStay`` entries directly
from the accepted cohort/timeline artifacts, for the same
``dashboard.catalog.DashboardCatalog`` the synthetic fixture already uses.
Never includes a sealed final-test subject: candidate stays are filtered
against the accepted split before any other step, using the same
``serving.real.history.load_stay_split_index`` helper the API's own guard
uses.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence, Tuple

from data.timestamps import RetainedICUStay, generate_prediction_rows_for_stay
from dashboard.catalog import DashboardCatalog, ReplayStay, TimelineEvent
from serving.real.history import load_stay_split_index


LAB_CONCEPTS = {"platelet_count", "bilirubin_total", "creatinine", "lactate"}


def _instant(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _utc_offset(value: datetime) -> str:
    # "+00:00", not "Z": matches every downstream consumer this catalog's
    # cutoffs actually flow through (serving.history's truncator, which
    # cannot parse "Z" under Python 3.9's datetime.fromisoformat).
    return value.astimezone(timezone.utc).isoformat()


def build_real_dashboard_catalog(
    root: Path,
    *,
    current_sofa_provider,
    split_by_stay: Optional[dict] = None,
    demo_stay_ids: Optional[Sequence[str]] = None,
    limit: int = 5,
) -> DashboardCatalog:
    root = root.resolve()
    retained_cohort_path = root / "artifacts/data/synthetic/cohorts/final/phase9_final_v1/retained_cohort.jsonl"
    split_path = root / "artifacts/splits/synthetic_split_v2.csv"
    if split_by_stay is None:
        split_by_stay = load_stay_split_index(retained_cohort_path=retained_cohort_path, split_path=split_path)

    stays_by_id = {}
    with retained_cohort_path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            stays_by_id[row["stay_id"]] = row

    if demo_stay_ids is None:
        candidates = sorted(
            stay_id for stay_id, split in split_by_stay.items() if split == "train"
        )
        demo_stay_ids = candidates[:limit]

    raw_manifest = json.loads(
        (root / "artifacts/data/synthetic/final/phase9_final_v1/synthetic_dataset_manifest_v1.json").read_text()
    )
    by_logical = {item["logical_name"]: root / item["repository_relative_path"] for item in raw_manifest["artifacts"]}

    wanted = set(demo_stay_ids)
    events_by_stay: dict = {stay_id: [] for stay_id in wanted}
    with by_logical["raw_events"].open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["stay_id"] not in wanted:
                continue
            category = "lab" if row["concept_code"] in LAB_CONCEPTS else "vital"
            events_by_stay[row["stay_id"]].append(
                TimelineEvent(
                    prediction_available_time=row["event_time"],
                    category=category,
                    label=row["concept_code"],
                    value=row["value_numeric"],
                    observed=True,
                )
            )
    with by_logical["support_intervals"].open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["stay_id"] not in wanted:
                continue
            events_by_stay[row["stay_id"]].append(
                TimelineEvent(
                    prediction_available_time=row["interval_start"],
                    category="support_state",
                    label=row["support_type"],
                    value=row["support_category"],
                    observed=True,
                )
            )

    stays: list = []
    for stay_id in demo_stay_ids:
        row = stays_by_id[stay_id]
        intime = _instant(row["intime"])
        outtime = _instant(row["outtime"])
        legal = generate_prediction_rows_for_stay(
            RetainedICUStay(subject_id=row["subject_id"], stay_id=stay_id, intime=intime, outtime=outtime)
        )
        legal_cutoffs = tuple(_utc_offset(entry.prediction_time) for entry in legal)
        current_by_cutoff = {}
        sofa_version = None
        for cutoff in legal_cutoffs:
            state = current_sofa_provider.current_sofa(stay_id=stay_id, prediction_time=cutoff)
            current_by_cutoff[cutoff] = state.value
            sofa_version = state.sofa_version
        stays.append(
            ReplayStay(
                subject_id=row["subject_id"],
                stay_id=stay_id,
                intime=_utc_offset(intime),
                legal_cutoffs=legal_cutoffs,
                events=tuple(
                    sorted(events_by_stay[stay_id], key=lambda event: _instant(event.prediction_available_time))
                ),
                current_sofa_by_cutoff=current_by_cutoff,
                sofa_version=sofa_version or "unknown_sofa_version",
                current_sofa_source_version="REAL_STAGE4_CANONICAL_TIMELINE_CURRENT_SOFA_V1",
                current_sofa_source_sha256=raw_manifest["artifacts"][0]["sha256"],
            )
        )
    return DashboardCatalog(stays, scope="real")
