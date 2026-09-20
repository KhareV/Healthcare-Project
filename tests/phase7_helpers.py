"""Explicitly synthetic Phase-7 fixture utilities, never production science."""

import json
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path

from data.schema import (
    CanonicalExample,
    FeatureSchemaReference,
    TaskEligibility,
    TaskTargets,
)
from data.support_labels import SupportLabelOverlayDataset, load_synthetic_support_contract
from phase4_helpers import tensor_dataset
from serving.history import (
    InMemorySyntheticHistoryProvider,
    StoredStayTimeline,
    SyntheticPointEventHistoryTruncator,
    TimelineContract,
)
from serving.preprocessing import CanonicalHistoryInputProvider


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUPPORT_FIXTURE = PROJECT_ROOT / "tests/fixtures/support/support_labels_synthetic_v1.json"
TIMELINE_FIXTURE = PROJECT_ROOT / "tests/fixtures/serving/history/synthetic_timeline_phase7_v1.json"
SCHEMA_FIXTURE = PROJECT_ROOT / "tests/fixtures/serving/features/synthetic_feature_schema_phase7_v1.json"
MAIN_STAY = "SYNTHETIC_PHASE6_STAY"
MAIN_CUTOFF = "2026-01-04T12:00:00+00:00"
LATER_CUTOFF = "2026-01-04T18:00:00+00:00"
EARLY_CUTOFF = "2026-01-02T12:00:00+00:00"


# Retained for the earlier Vedant Phase-7 support-model tests that share this
# historical helper module name.
def support_contract():
    return load_synthetic_support_contract(SUPPORT_FIXTURE)


def support_dataset(partition):
    return SupportLabelOverlayDataset(tensor_dataset(partition), support_contract())


def parse_time(value):
    return datetime.fromisoformat(value)


def load_schema():
    payload = json.loads(SCHEMA_FIXTURE.read_text())
    return FeatureSchemaReference(
        version=payload["version"],
        feature_names=tuple(payload["feature_names"]),
        status="SYNTHETIC_PHASE7_RESOLVED_FOR_ENGINEERING_TESTS_ONLY",
        tslo_no_observation_value=payload["tslo_no_observation_value"],
        tslo_no_observation_source=payload["tslo_no_observation_source"],
        static_feature_names=tuple(payload["static_feature_names"]),
    )


def load_timelines(payload=None):
    payload = payload or json.loads(TIMELINE_FIXTURE.read_text())
    raw_contract = payload["contract"]
    contract = TimelineContract(**raw_contract)
    return tuple(
        StoredStayTimeline.create(
            subject_id=item["subject_id"],
            stay_id=item["stay_id"],
            intime=parse_time(item["intime"]),
            outtime=parse_time(item["outtime"]),
            events=item["events"],
            contract=contract,
            source_version=item["source_version"],
            source_sha256=item["source_sha256"],
        )
        for item in payload["timelines"]
    )


class SyntheticCanonicalFeatureBuilder:
    """Test-only canonical builder shared by direct and serving paths.

    Its behavior is declared by the synthetic feature fixture.  It is not a
    substitute for Sanskruti's unavailable production ``build_features``.
    """

    def __init__(self, schema=None):
        self.schema = schema or load_schema()
        self.calls = []

    def __call__(self, history):
        self.calls.append(history)
        t = history.prediction_row.prediction_time
        window_start = t - timedelta(hours=48)
        feature_names = self.schema.feature_names
        feature_index = {name: index for index, name in enumerate(feature_names)}
        events = []
        for row in history.events:
            name = row.get("synthetic_feature")
            if name not in feature_index:
                raise ValueError("unregistered synthetic feature")
            event_time = parse_time(row[history.contract.event_time_field])
            if event_time <= window_start or event_time > t or event_time < history.intime:
                continue
            events.append((event_time, int(row["synthetic_sequence"]), row))
        events.sort(key=lambda item: (item[0], item[1]))

        values = []
        observations = []
        padding = []
        tslo = []
        last_genuine = [None] * len(feature_names)
        for bin_index in range(8):
            start = window_start + timedelta(hours=6 * bin_index)
            end = start + timedelta(hours=6)
            padded = end <= history.intime
            padding.append(padded)
            if padded:
                values.append(tuple(None for _ in feature_names))
                observations.append(tuple(False for _ in feature_names))
                tslo.append(tuple(float(self.schema.tslo_no_observation_value) for _ in feature_names))
                continue
            row_values = []
            row_observed = []
            row_tslo = []
            for name in feature_names:
                candidates = [item for item in events if item[2]["synthetic_feature"] == name and start < item[0] <= end]
                if candidates:
                    latest = candidates[-1]
                    row_values.append(float(latest[2]["synthetic_value"]))
                    row_observed.append(True)
                    last_genuine[feature_index[name]] = latest[0]
                else:
                    row_values.append(None)
                    row_observed.append(False)
                last = last_genuine[feature_index[name]]
                row_tslo.append(
                    float(self.schema.tslo_no_observation_value)
                    if last is None
                    else (end - last).total_seconds() / 3600.0
                )
            values.append(tuple(row_values))
            observations.append(tuple(row_observed))
            tslo.append(tuple(row_tslo))
        return CanonicalExample(
            subject_id=history.subject_id,
            stay_id=history.stay_id,
            prediction_time=t.isoformat(),
            grid_index=history.prediction_row.grid_index,
            icu_elapsed_hours=history.prediction_row.icu_elapsed_hours,
            anchor_year_group="2014-2016",
            split="validation",
            tensor_contract_version="tensor_contract_v1",
            timestamp_spec_version="timestamp_spec_v1",
            feature_schema_version=self.schema.version,
            history_dtype="float32",
            history_values=tuple(values),
            padding_mask_dtype="bool",
            padding_mask=tuple(padding),
            observation_mask_dtype="bool",
            observation_mask=tuple(observations),
            tslo_hours=tuple(tslo),
            tslo_status="SYNTHETIC_PHASE7_TSLO_RESOLVED_FOR_TEST_ONLY",
            static_features=(7.0,),
            static_feature_names=self.schema.static_feature_names,
            static_status="SYNTHETIC_PHASE7_STATIC_RESOLVED_FOR_TEST_ONLY",
            eligibility=TaskEligibility(False, False, False, False),
            targets=TaskTargets(None, None, None, None),
            case_tags=("SYNTHETIC_PHASE7_NOT_CLINICAL",),
        )


def build_provider(timelines=None, builder=None, schema=None):
    schema = schema or load_schema()
    builder = builder or SyntheticCanonicalFeatureBuilder(schema)
    provider = CanonicalHistoryInputProvider(
        history_provider=InMemorySyntheticHistoryProvider(timelines or load_timelines()),
        history_truncator=SyntheticPointEventHistoryTruncator(),
        build_features=builder,
        feature_schema=schema,
    )
    return provider, builder


def truncate(timeline=None, cutoff=MAIN_CUTOFF):
    timeline = timeline or load_timelines()[0]
    return SyntheticPointEventHistoryTruncator().truncate(timeline, cutoff)


class FrozenSyntheticPreprocessor:
    """Test spy representing a serialized train-fit transform-only artifact."""

    task = "synthetic_task"
    artifact_sha256 = "3" * 64

    def __init__(self):
        self.transform_calls = 0
        self.fit_calls = 0
        self.fit_transform_calls = 0

    def transform(self, value):
        self.transform_calls += 1
        return value

    def fit(self, *_args, **_kwargs):
        self.fit_calls += 1
        raise AssertionError("serving must not fit")

    def fit_transform(self, *_args, **_kwargs):
        self.fit_transform_calls += 1
        raise AssertionError("serving must not fit_transform")
