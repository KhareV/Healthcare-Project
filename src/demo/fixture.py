"""Validate the frozen Phase-15 demo with existing history/timestamp contracts."""

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Mapping, Tuple

from dashboard.catalog import DashboardCatalog, ReplayStay, TimelineEvent
from serving.history import (
    InMemorySyntheticHistoryProvider,
    StoredStayTimeline,
    SyntheticPointEventHistoryTruncator,
    TimelineContract,
)
from serving.recovery import CurrentSOFAState
from vedant_infra.hashing import is_sha256, sha256_file


CLASSIFICATION = "SYNTHETIC NON-SENSITIVE DEMO FIXTURE — NON-SCIENTIFIC OUTPUTS"
FIXTURE_VERSION = "demo_patient_v1"
DEFAULT_FIXTURE = Path("data/demo/demo_patient_v1.json")
DEFAULT_METADATA = Path("data/demo/demo_patient_v1.metadata.json")
ABSOLUTE_PATH = re.compile(
    r"(?:/" + r"Users/[^/\s]+/|/" + r"home/[^/\s]+/|[A-Za-z]:\\" + r"Users\\)"
)
SECRET_VALUE = re.compile(
    r'(?i)(?:password|api[_-]?token|access[_-]?token|secret|api[_-]?key)\s*[=:]\s*[^<\s][^\s]*'
)


class DemoFixtureError(RuntimeError):
    pass


@dataclass(frozen=True)
class ValidatedDemoFixture:
    fixture_path: Path
    metadata_path: Path
    fixture_sha256: str
    metadata_sha256: str
    timelines: Tuple[StoredStayTimeline, ...]
    legal_cutoffs: Tuple[str, ...]
    current_sofa: Mapping[str, float]

    @property
    def stay_id(self) -> object:
        return self.timelines[0].stay_id


class DemoCurrentSOFAProvider:
    """Synthetic SOFA(t) provider bound to the validated official demo input."""

    def __init__(self, validated: ValidatedDemoFixture) -> None:
        self._validated = validated

    def current_sofa(self, *, stay_id: object, prediction_time: str) -> CurrentSOFAState:
        if stay_id != self._validated.stay_id or prediction_time not in self._validated.current_sofa:
            raise DemoFixtureError("demo current SOFA request is outside the fixture")
        timeline = self._validated.timelines[0]
        return CurrentSOFAState(
            stay_id=stay_id,
            prediction_time=prediction_time,
            value=self._validated.current_sofa[prediction_time],
            sofa_version="SYNTHETIC_SOFA_AT_T_PHASE15_V1",
            source_version="demo_patient_v1_current_sofa",
            source_sha256=timeline.source_sha256,
            component_observed=(True, True, False, True, False, True),
        )


def _json(path: Path, label: str):
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DemoFixtureError(label + " is unreadable") from error
    if not isinstance(value, dict):
        raise DemoFixtureError(label + " must be an object")
    return value


def validate_demo_fixture(
    root: Path,
    *,
    fixture_ref: Path = DEFAULT_FIXTURE,
    metadata_ref: Path = DEFAULT_METADATA,
) -> ValidatedDemoFixture:
    root = root.resolve()
    fixture_path = (root / fixture_ref).resolve()
    metadata_path = (root / metadata_ref).resolve()
    if root not in fixture_path.parents or root not in metadata_path.parents:
        raise DemoFixtureError("demo paths must remain inside repository root")
    payload = _json(fixture_path, "demo fixture")
    metadata = _json(metadata_path, "demo metadata")
    fixture_hash = sha256_file(fixture_path)
    if (
        payload.get("classification") != CLASSIFICATION
        or payload.get("fixture_version") != FIXTURE_VERSION
        or payload.get("scope") != "synthetic"
        or payload.get("contains_restricted_raw_mimic") is not False
    ):
        raise DemoFixtureError("demo fixture scope/provenance is invalid")
    if (
        metadata.get("metadata_version") != "demo_fixture_metadata_v1"
        or metadata.get("fixture_version") != FIXTURE_VERSION
        or metadata.get("fixture_ref") != fixture_ref.as_posix()
        or metadata.get("fixture_sha256") != fixture_hash
        or metadata.get("scope") != "synthetic"
        or metadata.get("contains_restricted_raw_mimic") is not False
    ):
        raise DemoFixtureError("demo metadata/hash binding is invalid")
    text = fixture_path.read_text(encoding="utf-8") + metadata_path.read_text(encoding="utf-8")
    if ABSOLUTE_PATH.search(text) or SECRET_VALUE.search(text):
        raise DemoFixtureError("demo package contains an absolute path or secret-like value")
    if any(term in text.lower() for term in ('"itemid"', '"hadm_id"', '"caregiver_id"')):
        raise DemoFixtureError("demo fixture contains a prohibited raw-clinical field")
    if any(key.startswith("prediction_") for key in payload):
        raise DemoFixtureError("demo input cannot contain precomputed predictions")

    raw_contract = payload.get("contract")
    raw_timelines = payload.get("timelines")
    feature_schema = payload.get("feature_schema")
    if not isinstance(raw_contract, dict) or not isinstance(raw_timelines, list) or len(raw_timelines) != 1:
        raise DemoFixtureError("demo timeline contract must contain exactly one stay")
    if not isinstance(feature_schema, dict) or feature_schema.get("version") != metadata.get("feature_schema_version"):
        raise DemoFixtureError("demo feature schema identity is incompatible")
    contract = TimelineContract(**raw_contract)
    item = raw_timelines[0]
    try:
        timeline = StoredStayTimeline.create(
            subject_id=item["subject_id"], stay_id=item["stay_id"],
            intime=datetime.fromisoformat(item["intime"]),
            outtime=datetime.fromisoformat(item["outtime"]),
            events=item["events"], contract=contract,
            source_version=item["source_version"], source_sha256=item["source_sha256"],
        )
    except (KeyError, TypeError, ValueError) as error:
        raise DemoFixtureError("demo stay is malformed") from error
    if not str(timeline.subject_id).startswith("SYNTHETIC_") or not str(timeline.stay_id).startswith("SYNTHETIC_"):
        raise DemoFixtureError("demo identifiers must be unmistakably synthetic")
    if not is_sha256(timeline.source_sha256):
        raise DemoFixtureError("demo timeline source hash is invalid")
    InMemorySyntheticHistoryProvider((timeline,))
    feature_names = set(feature_schema.get("feature_names", ()))
    if not feature_names or any(row.get("synthetic_feature") not in feature_names for row in timeline.events):
        raise DemoFixtureError("demo event feature is outside the declared schema")

    cutoffs = tuple(item.get("approved_cutoffs", ()))
    if len(cutoffs) < 3 or cutoffs != tuple(metadata.get("legal_cutoffs", ())):
        raise DemoFixtureError("demo requires at least three metadata-bound cutoffs")
    if datetime.fromisoformat(cutoffs[0]) != timeline.intime + timedelta(hours=24):
        raise DemoFixtureError("first demo cutoff must be ICU admission plus 24 hours")
    if any(
        datetime.fromisoformat(right) - datetime.fromisoformat(left) != timedelta(hours=6)
        for left, right in zip(cutoffs, cutoffs[1:])
    ):
        raise DemoFixtureError("demo cutoffs must follow the six-hour cadence")
    truncator = SyntheticPointEventHistoryTruncator()
    retained_counts = tuple(len(truncator.truncate(timeline, cutoff).events) for cutoff in cutoffs)
    if tuple(sorted(retained_counts)) != retained_counts or len(set(retained_counts)) < 3:
        raise DemoFixtureError("demo events do not exercise three replay histories")
    if not any(row[contract.event_time_field] > cutoffs[-1] for row in timeline.events):
        raise DemoFixtureError("demo requires a future row after the third cutoff")
    current = item.get("current_sofa")
    if not isinstance(current, dict) or set(current) != set(cutoffs):
        raise DemoFixtureError("demo current-SOFA contract is incomplete")
    for cutoff, value in current.items():
        CurrentSOFAState(
            stay_id=timeline.stay_id, prediction_time=cutoff, value=value,
            sofa_version="SYNTHETIC_SOFA_AT_T_PHASE15_V1",
            source_version="demo_patient_v1_current_sofa",
            source_sha256=timeline.source_sha256,
        ).validate()
    return ValidatedDemoFixture(
        fixture_path, metadata_path, fixture_hash, sha256_file(metadata_path),
        (timeline,), cutoffs, dict(current),
    )


def dashboard_catalog_for_demo(validated: ValidatedDemoFixture) -> DashboardCatalog:
    timeline = validated.timelines[0]
    events = tuple(
        TimelineEvent(
            prediction_available_time=str(row["synthetic_available_time"]),
            category=str(row["synthetic_category"]),
            label=str(row["synthetic_label"]),
            value=row["synthetic_value"], observed=True,
        )
        for row in timeline.events
    )
    return DashboardCatalog(
        (
            ReplayStay(
                subject_id=str(timeline.subject_id), stay_id=str(timeline.stay_id),
                intime=timeline.intime.isoformat(), legal_cutoffs=validated.legal_cutoffs,
                events=events, current_sofa_by_cutoff=validated.current_sofa,
                sofa_version="SYNTHETIC_SOFA_AT_T_PHASE15_V1",
            ),
        ),
        scope="synthetic",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the Phase-15 synthetic demo input")
    parser.add_argument("command", choices=("validate",))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    result = validate_demo_fixture(args.root)
    print(json.dumps({
        "status": "PASS", "scope": "synthetic", "fixture_version": FIXTURE_VERSION,
        "fixture_sha256": result.fixture_sha256,
        "metadata_sha256": result.metadata_sha256,
        "stay_id": result.stay_id, "legal_cutoffs": result.legal_cutoffs,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
