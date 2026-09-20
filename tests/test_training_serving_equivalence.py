import json
import random

from data.schema import temporal_information_views, unflatten_temporal
from phase7_helpers import (
    MAIN_CUTOFF,
    MAIN_STAY,
    TIMELINE_FIXTURE,
    FrozenSyntheticPreprocessor,
    SyntheticCanonicalFeatureBuilder,
    build_provider,
    load_schema,
    load_timelines,
    truncate,
)
from serving.history import StoredStayTimeline


def _serving_view(provider, family="gru"):
    return provider.get_canonical_input(
        stay_id=MAIN_STAY,
        prediction_time=MAIN_CUTOFF,
        task="recovery",
        family=family,
        feature_version=load_schema().version,
    )


def test_modeling_and_serving_use_same_builder_and_equal_model_input():
    schema = load_schema()
    builder = SyntheticCanonicalFeatureBuilder(schema)
    direct_features = builder(truncate())
    preprocessor = FrozenSyntheticPreprocessor()
    direct = preprocessor.transform(temporal_information_views(direct_features)["gru"])
    provider, same_builder = build_provider(builder=builder, schema=schema)
    serving = preprocessor.transform(_serving_view(provider, "gru"))
    assert same_builder is builder
    assert len(builder.calls) == 2
    assert direct == serving


def test_full_history_and_explicit_pretruncation_are_identical():
    full_provider, _ = build_provider()
    full = _serving_view(full_provider)
    timeline = load_timelines()[0]
    through_t = tuple(
        row for row in timeline.events if row["synthetic_available_time"] <= MAIN_CUTOFF
    )
    explicit = StoredStayTimeline.create(
        subject_id=timeline.subject_id,
        stay_id=timeline.stay_id,
        intime=timeline.intime,
        outtime=timeline.outtime,
        events=through_t,
        contract=timeline.contract,
        source_version=timeline.source_version + "_EXPLICITLY_TRUNCATED",
        source_sha256="4" * 64,
    )
    explicit_provider, _ = build_provider(timelines=(explicit,))
    assert full == _serving_view(explicit_provider)


def test_future_mutate_delete_add_and_reorder_cannot_change_t_input():
    payload = json.loads(TIMELINE_FIXTURE.read_text())
    baseline, _ = build_provider()
    expected = _serving_view(baseline)
    events = payload["timelines"][0]["events"]
    retained = [row for row in events if row["synthetic_available_time"] <= MAIN_CUTOFF]
    hostile_future = [
        {
            "synthetic_stay_key": MAIN_STAY,
            "synthetic_available_time": "2026-01-05T18:00:00+00:00",
            "synthetic_feature": "SYNTHETIC_SIGNAL_A",
            "synthetic_value": -1e300,
            "synthetic_sequence": 1000,
            "future_sofa_48h": 1e300,
            "support_target": 1,
        }
    ]
    payload["timelines"][0]["events"] = hostile_future + list(reversed(retained))
    changed, _ = build_provider(timelines=load_timelines(payload))
    assert _serving_view(changed) == expected


def test_raw_row_order_invariance_is_owned_by_shared_builder():
    timelines = list(load_timelines())
    timeline = timelines[0]
    shuffled = list(timeline.events)
    random.Random(7391).shuffle(shuffled)
    timelines[0] = StoredStayTimeline.create(
        subject_id=timeline.subject_id,
        stay_id=timeline.stay_id,
        intime=timeline.intime,
        outtime=timeline.outtime,
        events=shuffled,
        contract=timeline.contract,
        source_version=timeline.source_version,
        source_sha256="5" * 64,
    )
    expected, _ = build_provider()
    actual, _ = build_provider(timelines=tuple(timelines))
    assert _serving_view(actual) == _serving_view(expected)


def test_xgboost_is_only_flattened_gru_canonical_information():
    provider, _ = build_provider()
    gru = _serving_view(provider, "gru")
    xgb = _serving_view(provider, "xgboost")
    feature_dim = load_schema().feature_dim
    assert unflatten_temporal(xgb["history_values"], feature_dim) == gru["history_values"]
    assert unflatten_temporal(xgb["observation_mask"], feature_dim) == gru["observation_mask"]
    assert unflatten_temporal(xgb["tslo_hours"], feature_dim) == gru["tslo_hours"]
    assert xgb["padding_mask"] == gru["padding_mask"]
    assert xgb["static_features"] == gru["static_features"]
    assert set(xgb) == set(gru)


def test_data_quality_comes_from_canonical_masks_not_confidence():
    provider, _ = build_provider()
    _serving_view(provider)
    quality = provider.data_quality(stay_id=MAIN_STAY, prediction_time=MAIN_CUTOFF)
    assert quality["total_bins"] == 8
    assert quality["observed_feature_values"] == 4
    assert "confidence" not in quality
    assert "reliability" not in quality
