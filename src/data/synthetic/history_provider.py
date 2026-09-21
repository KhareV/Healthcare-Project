"""Storage-backed Phase-5 adapter for Pulkit's existing HistoryProvider seam."""
from __future__ import annotations
import json
from pathlib import Path
from serving.history import InMemorySyntheticHistoryProvider, StoredStayTimeline, TimelineContract, SYNTHETIC_POINT_EVENT_SEMANTICS, SYNTHETIC_TIMELINE_SCOPE
from .processed_manifest import validate_processed_manifest
from .validation import load_jsonl, parse_utc

class CanonicalTimelineHistoryProvider(InMemorySyntheticHistoryProvider):
    def __init__(self, manifest_path: Path, root: Path):
        manifest=validate_processed_manifest(manifest_path,root)
        artifacts={a["logical_name"]:root/a["repository_relative_path"] for a in manifest["artifacts"]}
        stored_events=load_jsonl(artifacts["canonical_timeline"]); statics=load_jsonl(artifacts["canonical_statics"])
        phase3=json.loads((root/manifest["phase3_manifest_path"]).read_text())
        phase3_artifacts={a["logical_name"]:root/a["repository_relative_path"] for a in phase3["artifacts"]}
        stored_support=load_jsonl(phase3_artifacts["support_intervals"])
        # Python 3.9's datetime.fromisoformat (used by the frozen Pulkit
        # truncator) does not parse ``Z``.  Adapt only the equivalent UTC
        # spelling at this boundary; the canonical artifact remains unchanged.
        events=[]
        for stored in stored_events:
            row=dict(stored)
            if row["event_time"].endswith("Z"):
                row["event_time"]=row["event_time"][:-1]+"+00:00"
            events.append(row)
        by_stay={row["stay_id"]:[] for row in statics}
        for row in events: by_stay[row["stay_id"]].append(row)
        support_by_stay={row["stay_id"]:[] for row in statics}
        for stored in stored_support:
            row=dict(stored)
            for field in ("interval_start","interval_end"):
                if row[field].endswith("Z"): row[field]=row[field][:-1]+"+00:00"
            if row["stay_id"] in support_by_stay: support_by_stay[row["stay_id"]].append(row)
        contract=TimelineContract(version=manifest["processed_schema_version"],scope=SYNTHETIC_TIMELINE_SCOPE,stay_id_field="stay_id",event_time_field="event_time",event_time_semantics=SYNTHETIC_POINT_EVENT_SEMANTICS,stateful_intervals_present=False)
        timeline_hash=next(a["sha256"] for a in manifest["artifacts"] if a["logical_name"]=="canonical_timeline")
        timelines=[StoredStayTimeline.create(subject_id=row["subject_id"],stay_id=row["stay_id"],intime=parse_utc(row["intime"]),outtime=parse_utc(row["outtime"]),events=by_stay[row["stay_id"]],support_intervals=support_by_stay[row["stay_id"]],contract=contract,source_version=manifest["processed_schema_version"],source_sha256=timeline_hash) for row in statics]
        super().__init__(timelines)
