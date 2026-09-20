import json

import pytest

from dashboard.app import DEFAULT_FIXTURE
from dashboard.catalog import load_synthetic_dashboard_catalog
from dashboard.replay import ReplayContractError
from dashboard.replay import ReplayController
from tests.dashboard_helpers import FakeDashboardAPI, controller


def test_forward_backward_replay_calls_api_once_and_hides_later_events():
    api = FakeDashboardAPI()
    replay = controller(api)
    replay.select_stay("SYNTHETIC_PHASE13_STAY_A")
    a, b, c = replay.available_cutoffs
    snapshots = [replay.select_cutoff(value) for value in (a, b, c, a)]
    assert [len(item.timeline) for item in snapshots] == [2, 3, 4, 2]
    assert all(
        event.prediction_available_time <= snapshot.prediction_time
        for snapshot in snapshots
        for event in snapshot.timeline
    )
    assert [call for call in api.calls if call[0] == "POST"] == [
        ("POST", "/predict", "SYNTHETIC_PHASE13_STAY_A", value)
        for value in (a, b, c, a)
    ]


def test_switching_stays_clears_cutoff_and_snapshot():
    replay = controller()
    replay.select_stay("SYNTHETIC_PHASE13_STAY_A")
    replay.select_cutoff(replay.available_cutoffs[0])
    replay.select_stay("SYNTHETIC_PHASE13_STAY_B")
    assert replay.selected_cutoff is None
    assert replay.snapshot is None
    assert replay.available_cutoffs == (
        "2026-03-02T12:00:00+00:00",
        "2026-03-02T18:00:00+00:00",
        "2026-03-03T00:00:00+00:00",
    )


def test_illegal_cutoff_never_calls_api():
    api = FakeDashboardAPI()
    replay = controller(api)
    replay.select_stay("SYNTHETIC_PHASE13_STAY_A")
    with pytest.raises(ReplayContractError, match="not legal"):
        replay.select_cutoff("2026-02-03T01:00:00+00:00")
    assert api.calls == []
    assert replay.snapshot is None


def test_malformed_response_clears_stale_snapshot_and_fails_closed():
    api = FakeDashboardAPI()
    replay = controller(api)
    replay.select_stay("SYNTHETIC_PHASE13_STAY_A")
    replay.select_cutoff(replay.available_cutoffs[0])
    api.mutate = lambda kind, value: value.pop("data_quality") if kind == "prediction" else None
    with pytest.raises(ReplayContractError, match="contract"):
        replay.select_cutoff(replay.available_cutoffs[1])
    assert replay.snapshot is None
    assert replay.last_error == "replay data is unavailable or incompatible"


def test_future_timeline_perturbation_does_not_change_earlier_snapshot(tmp_path):
    payload = json.loads(DEFAULT_FIXTURE.read_text())
    changed = json.loads(DEFAULT_FIXTURE.read_text())
    changed["stays"][0]["events"][-1]["synthetic_value"] = -123456.0
    original_path = tmp_path / "original.json"
    changed_path = tmp_path / "changed.json"
    original_path.write_text(json.dumps(payload))
    changed_path.write_text(json.dumps(changed))

    snapshots = []
    for path in (original_path, changed_path):
        replay = ReplayController(
            load_synthetic_dashboard_catalog(path), FakeDashboardAPI()
        )
        replay.select_stay("SYNTHETIC_PHASE13_STAY_A")
        snapshots.append(replay.select_cutoff(replay.available_cutoffs[0]))
    assert snapshots[0].timeline == snapshots[1].timeline
    assert snapshots[0].current_sofa.value == snapshots[1].current_sofa.value
    assert snapshots[0].prediction == snapshots[1].prediction


def test_cross_stay_snapshots_share_no_timeline_or_prediction_state():
    replay = controller()
    replay.select_stay("SYNTHETIC_PHASE13_STAY_A")
    old = replay.select_cutoff(replay.available_cutoffs[-1])
    replay.select_stay("SYNTHETIC_PHASE13_STAY_B")
    new = replay.select_cutoff(replay.available_cutoffs[0])
    assert old.stay_id != new.stay_id
    assert {item.label for item in old.timeline}.isdisjoint(
        {item.label for item in new.timeline}
    )


@pytest.mark.parametrize(
    "mutation",
    (
        lambda kind, value: value["tasks"]["recovery"].update(
            feature_version="INCOMPATIBLE"
        ) if kind == "metadata" else None,
        lambda kind, value: value["organ_support"].update(
            calibrator_sha256="INCOMPATIBLE"
        ) if kind == "metadata" else None,
        lambda kind, value: value.update(serving_scope="real")
        if kind == "metadata" else None,
    ),
)
def test_incompatible_response_identity_is_not_corrected_by_ui(mutation):
    replay = controller(FakeDashboardAPI(mutate=mutation))
    replay.select_stay("SYNTHETIC_PHASE13_STAY_A")
    with pytest.raises(ReplayContractError):
        replay.select_cutoff(replay.available_cutoffs[0])
    assert replay.snapshot is None
