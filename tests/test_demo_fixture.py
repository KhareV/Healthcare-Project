import json
import shutil

import pytest

from dashboard.view_models import build_dashboard_view
from demo.fixture import (
    DEFAULT_FIXTURE,
    DEFAULT_METADATA,
    DemoCurrentSOFAProvider,
    DemoFixtureError,
    dashboard_catalog_for_demo,
    validate_demo_fixture,
)
from integration_helpers import build_integrated_system


def test_official_demo_fixture_hash_schema_and_three_cutoffs(project_root=None):
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    value = validate_demo_fixture(root)
    assert value.fixture_sha256 == "625289d3eaff3482cc94d4190bed61adb8f81c1a8eb24f2e148397b6a85d1a43"
    assert value.stay_id == "SYNTHETIC_DEMO_PATIENT_V1_STAY"
    assert value.legal_cutoffs == (
        "2030-01-02T00:00:00+00:00",
        "2030-01-02T06:00:00+00:00",
        "2030-01-02T12:00:00+00:00",
    )


def test_fixture_mutation_without_metadata_hash_update_is_rejected(tmp_path):
    root = tmp_path
    target = root / "data/demo"
    target.mkdir(parents=True)
    source_root = __import__("pathlib").Path(__file__).resolve().parents[1]
    shutil.copy2(source_root / DEFAULT_FIXTURE, target / DEFAULT_FIXTURE.name)
    shutil.copy2(source_root / DEFAULT_METADATA, target / DEFAULT_METADATA.name)
    path = target / DEFAULT_FIXTURE.name
    payload = json.loads(path.read_text())
    payload["timelines"][0]["events"][0]["synthetic_value"] = -999.0
    path.write_text(json.dumps(payload))
    with pytest.raises(DemoFixtureError, match="metadata/hash"):
        validate_demo_fixture(root)


def test_official_demo_passes_pipeline_api_and_dashboard_replay(tmp_path):
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    demo = validate_demo_fixture(root)
    system = build_integrated_system(
        tmp_path,
        timelines=demo.timelines,
        current_sofa_provider=DemoCurrentSOFAProvider(demo),
        expected_sofa_version="SYNTHETIC_SOFA_AT_T_PHASE15_V1",
        dashboard_catalog_override=dashboard_catalog_for_demo(demo),
    )
    controller = system.controller()
    controller.select_stay(demo.stay_id)
    snapshots = [controller.select_cutoff(cutoff) for cutoff in demo.legal_cutoffs]
    assert [item.prediction_time for item in snapshots] == list(demo.legal_cutoffs)
    assert [len(item.timeline) for item in snapshots] == [2, 3, 4]
    for snapshot in snapshots:
        view = build_dashboard_view(snapshot, synthetic=True)
        assert view.recovery.current_sofa == demo.current_sofa[snapshot.prediction_time]
        assert all(
            event.prediction_available_time <= snapshot.prediction_time
            for event in snapshot.timeline
        )
    assert [call for call in system.dashboard_api.calls if call[0] == "POST"] == [
        ("POST", "/predict", demo.stay_id, cutoff)
        for cutoff in demo.legal_cutoffs
    ]
    assert system.runtime.calibrator.fit_calls == 0


def test_demo_fixture_contains_no_prediction_lookup():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    payload = json.loads((root / DEFAULT_FIXTURE).read_text())
    text = json.dumps(payload).lower()
    for prohibited in ("prediction_t1", "prediction_t2", "prediction_t3", "precomputed_prediction"):
        assert prohibited not in text
