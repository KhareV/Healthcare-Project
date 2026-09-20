import pytest

from dashboard.view_models import build_dashboard_view
from tests.dashboard_helpers import controller


def test_quality_counts_are_exact_and_padding_is_distinct_from_missingness():
    replay = controller()
    replay.select_stay("SYNTHETIC_PHASE13_STAY_A")
    view = build_dashboard_view(
        replay.select_cutoff(replay.available_cutoffs[0]), synthetic=True
    )
    assert view.quality == {
        "total_bins": 8,
        "observed_bins": 6,
        "padding_bins": 2,
        "total_feature_values": 80,
        "observed_feature_values": 48,
        "missing_feature_values": 32,
    }


def test_view_model_does_not_default_missing_quality_fields():
    replay = controller()
    replay.select_stay("SYNTHETIC_PHASE13_STAY_A")
    snapshot = replay.select_cutoff(replay.available_cutoffs[0])
    snapshot.prediction["data_quality"].pop("padding_bins")
    with pytest.raises((ValueError, KeyError)):
        build_dashboard_view(snapshot, synthetic=True)
