from dataclasses import replace

import pytest

from phase7_helpers import (
    EARLY_CUTOFF,
    MAIN_CUTOFF,
    MAIN_STAY,
    SyntheticCanonicalFeatureBuilder,
    build_provider,
    load_schema,
    load_timelines,
    truncate,
)
from serving.history import StoredStayTimeline
from serving.preprocessing import ServingFeatureContractError


def _view(provider, cutoff=MAIN_CUTOFF):
    return provider.get_canonical_input(
        stay_id=MAIN_STAY,
        prediction_time=cutoff,
        task="organ_support",
        family="gru",
        feature_version=load_schema().version,
    )


def test_left_boundary_excluded_and_epsilon_included():
    view = _view(build_provider()[0])
    # First active bin starts at t-48h. Exact-left value 480 is excluded;
    # the event one second later is the genuine value.
    assert view["history_values"][0][0] == 1.0
    assert view["observation_mask"][0][0] is True


def test_t_minus_60h_cannot_seed_forward_fill_or_tslo():
    timeline = load_timelines()[0]
    only_old_b = tuple(
        row
        for row in timeline.events
        if row["synthetic_feature"] == "SYNTHETIC_SIGNAL_B"
        and row["synthetic_available_time"] == "2026-01-02T00:00:00+00:00"
    )
    reduced = StoredStayTimeline.create(
        subject_id=timeline.subject_id,
        stay_id=timeline.stay_id,
        intime=timeline.intime,
        outtime=timeline.outtime,
        events=only_old_b,
        contract=timeline.contract,
        source_version=timeline.source_version,
        source_sha256="6" * 64,
    )
    view = _view(build_provider(timelines=(reduced,))[0])
    assert all(row[1] is None for row in view["history_values"])
    assert all(row[1] is False for row in view["observation_mask"])
    assert all(row[1] == 999.0 for row in view["tslo_hours"])


def test_pre_icu_bins_are_padding_not_clinical_missingness():
    provider, _ = build_provider()
    view = _view(provider, EARLY_CUTOFF)
    assert view["padding_mask"] == (True, True, True, True, False, False, False, False)
    assert all(value is None for row in view["history_values"][:4] for value in row)
    assert all(not value for row in view["observation_mask"][:4] for value in row)
    assert view["padding_mask"][3] is True
    assert view["padding_mask"][4] is False
    assert view["observation_mask"][4] == (False, False)


def test_future_outcome_and_audit_metadata_never_enters_projection():
    timeline = load_timelines()[0]
    baseline = _view(build_provider()[0])
    changed_rows = []
    for row in timeline.events:
        changed = dict(row)
        changed.update(
            future_sofa_24h=9999,
            future_sofa_48h=9999,
            remaining_icu_hours=9999,
            total_icu_los=9999,
            future_support_onset=True,
            support_target=1,
            censor_reason="SYNTHETIC_FUTURE_AUDIT_ONLY",
        )
        changed_rows.append(changed)
    changed_timeline = StoredStayTimeline.create(
        subject_id=timeline.subject_id,
        stay_id=timeline.stay_id,
        intime=timeline.intime,
        outtime=timeline.outtime,
        events=changed_rows,
        contract=timeline.contract,
        source_version=timeline.source_version,
        source_sha256="7" * 64,
    )
    assert _view(build_provider(timelines=(changed_timeline,))[0]) == baseline


def test_unregistered_extra_feature_fails_closed():
    timeline = load_timelines()[0]
    bad = dict(timeline.events[4])
    bad["synthetic_feature"] = "UNREGISTERED_MODEL_CHANNEL"
    changed = StoredStayTimeline.create(
        subject_id=timeline.subject_id,
        stay_id=timeline.stay_id,
        intime=timeline.intime,
        outtime=timeline.outtime,
        events=(bad,) + timeline.events,
        contract=timeline.contract,
        source_version=timeline.source_version,
        source_sha256="8" * 64,
    )
    with pytest.raises(ValueError, match="unregistered synthetic feature"):
        _view(build_provider(timelines=(changed,))[0])


def test_missing_model_feature_channel_fails_closed():
    base_builder = SyntheticCanonicalFeatureBuilder()

    def missing_channel(history):
        example = base_builder(history)
        return replace(
            example,
            history_values=tuple(tuple(row[:1]) for row in example.history_values),
            observation_mask=tuple(tuple(row[:1]) for row in example.observation_mask),
            tslo_hours=tuple(tuple(row[:1]) for row in example.tslo_hours),
        )

    missing_channel.schema = load_schema()
    provider, _ = build_provider(builder=missing_channel)
    with pytest.raises(Exception, match="feature dimension"):
        _view(provider)


def test_wrong_schema_version_and_feature_order_fail_before_model():
    schema = load_schema()
    provider, _ = build_provider(schema=schema)
    with pytest.raises(ServingFeatureContractError, match="feature version"):
        provider.get_canonical_input(
            stay_id=MAIN_STAY,
            prediction_time=MAIN_CUTOFF,
            task="recovery",
            family="gru",
            feature_version="DRIFTED_SCHEMA",
        )
    reordered = replace(schema, feature_names=tuple(reversed(schema.feature_names)))
    # Builder and validator disagree: serving may not silently reorder channels.
    builder = SyntheticCanonicalFeatureBuilder(schema)
    with pytest.raises(ServingFeatureContractError, match="schema/order"):
        build_provider(builder=builder, schema=reordered)


def test_target_bearing_canonical_output_is_rejected():
    base_builder = SyntheticCanonicalFeatureBuilder()

    def malicious(history):
        example = base_builder(history)
        return replace(
            example,
            eligibility=replace(example.eligibility, organ_support_eligible=True),
            targets=replace(example.targets, organ_support_target=1),
        )

    malicious.schema = load_schema()
    provider, _ = build_provider(builder=malicious)
    with pytest.raises(ServingFeatureContractError, match="targets"):
        _view(provider)


def test_no_prediction_lookup_or_result_cache_in_serving_path():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    source = "\n".join(
        (root / path).read_text()
        for path in ("src/serving/history.py", "src/serving/preprocessing.py", "src/serving/pipeline.py")
    ).lower()
    for forbidden in ("predictions.csv", "final_predictions", "prediction_lookup"):
        assert forbidden not in source
    assert "cache" not in (root / "src/serving/history.py").read_text().lower()
