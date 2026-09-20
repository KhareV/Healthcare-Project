"""Pulkit Phase-14 structural end-to-end integration acceptance tests."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from fastapi.testclient import TestClient

from api.schemas import PredictionResponse
from dashboard.app import create_dashboard_app
from dashboard.view_models import REPLAY_BANNER, build_dashboard_view
from integration_helpers import LEGAL_CUTOFFS, build_integrated_system
from models.icu_time_postprocess import remaining_icu_hours_from_log_prediction
from phase7_helpers import MAIN_CUTOFF, MAIN_STAY, TIMELINE_FIXTURE, load_timelines
from serving.prediction_schema import validate_response


ROOT = Path(__file__).resolve().parents[1]


def request(cutoff=MAIN_CUTOFF):
    return {"stay_id": MAIN_STAY, "prediction_time": cutoff}


def test_raw_history_to_pipeline_api_and_dashboard_happy_path(tmp_path):
    system = build_integrated_system(tmp_path)
    direct = system.pipeline.predict(request())
    http = system.api_client.post("/predict", json=request())
    assert http.status_code == 200
    api_value = http.json()
    assert api_value == json.loads(json.dumps(direct))
    assert PredictionResponse.model_validate(api_value).model_dump() == api_value
    assert validate_response(api_value, synthetic=True) == api_value

    controller = system.controller()
    controller.select_stay(MAIN_STAY)
    snapshot = controller.select_cutoff(MAIN_CUTOFF)
    view = build_dashboard_view(snapshot, synthetic=True)
    assert snapshot.prediction == api_value
    assert view.recovery.current_sofa == 4.0
    assert view.recovery.delta_24h == api_value["recovery"]["delta_24h"]
    assert view.recovery.delta_48h == api_value["recovery"]["delta_48h"]
    assert view.recovery.predicted_sofa_24h == api_value["recovery"]["reconstructed_sofa_24h"]
    assert view.recovery.predicted_sofa_48h == api_value["recovery"]["reconstructed_sofa_48h"]
    assert view.icu_stay_time_hours == api_value["icu_stay_time_hours"]
    assert view.support_probability == api_value["organ_support_probability_calibrated"]
    assert view.quality == api_value["data_quality"]

    assert [item.family for item in view.task_metadata] == ["gru", "xgboost", "gru"]
    assert [item.method for item in view.explanations] == [
        "integrated_gradients", "tree_shap", "integrated_gradients"
    ]
    assert system.runtime.calibrator.transform_calls == 3
    assert system.runtime.calibrator.fit_calls == 0
    assert all(value.transform_calls == 3 for value in system.runtime.preprocessors.values())
    assert [call[0] for call in system.runtime.explanations.calls[-3:]] == [
        "recovery", "icu_stay_time", "organ_support"
    ]


def test_three_cutoff_replay_uses_exact_api_path_without_output_change_requirement(tmp_path):
    system = build_integrated_system(tmp_path)
    controller = system.controller()
    controller.select_stay(MAIN_STAY)
    snapshots = [controller.select_cutoff(cutoff) for cutoff in LEGAL_CUTOFFS]
    assert [item.prediction_time for item in snapshots] == list(LEGAL_CUTOFFS)
    assert [call for call in system.dashboard_api.calls if call[0] == "POST"] == [
        ("POST", "/predict", MAIN_STAY, cutoff) for cutoff in LEGAL_CUTOFFS
    ]
    for snapshot in snapshots:
        assert all(
            event.prediction_available_time <= snapshot.prediction_time
            for event in snapshot.timeline
        )
    # Synthetic models intentionally return equal values; this is not a failure.
    assert len({item.prediction["icu_stay_time_hours"] for item in snapshots}) == 1


def test_full_history_explicit_truncation_and_future_perturbation_are_equal(tmp_path):
    payload = json.loads(TIMELINE_FIXTURE.read_text())
    cutoff_safe = json.loads(TIMELINE_FIXTURE.read_text())
    cutoff_safe["timelines"][0]["events"] = [
        row for row in cutoff_safe["timelines"][0]["events"]
        if row["synthetic_available_time"] <= MAIN_CUTOFF
    ]
    hostile = json.loads(TIMELINE_FIXTURE.read_text())
    for row in hostile["timelines"][0]["events"]:
        if row["synthetic_available_time"] > MAIN_CUTOFF:
            row["synthetic_value"] = -1e200
    hostile["timelines"][0]["events"].append(
        {
            "synthetic_stay_key": MAIN_STAY,
            "synthetic_available_time": "2026-01-05T18:00:00+00:00",
            "synthetic_feature": "SYNTHETIC_SIGNAL_A",
            "synthetic_value": 1e200,
            "synthetic_sequence": 999,
        }
    )
    systems = [
        build_integrated_system(tmp_path / name, timelines=load_timelines(value))
        for name, value in (("full", payload), ("truncated", cutoff_safe), ("hostile", hostile))
    ]
    outputs = [system.pipeline.predict(request()) for system in systems]
    assert outputs[0] == outputs[1] == outputs[2]
    for task in ("recovery", "icu_stay_time", "organ_support"):
        prepared = [system.runtime.predictors[task].last_prepared_input for system in systems]
        assert prepared[0] == prepared[1] == prepared[2]
    for system in systems:
        controller = system.controller()
        controller.select_stay(MAIN_STAY)
        snapshot = controller.select_cutoff(MAIN_CUTOFF)
        assert snapshot.prediction == outputs[0]
        assert snapshot.prediction["data_quality"] == outputs[0]["data_quality"]


def test_stay_cutoff_and_api_error_isolation(tmp_path):
    system = build_integrated_system(tmp_path)
    unknown = system.api_client.post(
        "/predict", json={"stay_id": "SYNTHETIC_UNKNOWN", "prediction_time": MAIN_CUTOFF}
    )
    illegal = system.api_client.post(
        "/predict", json={"stay_id": MAIN_STAY, "prediction_time": "2026-01-02T12:00:01+00:00"}
    )
    assert unknown.status_code == 404
    assert illegal.status_code == 422


def test_canonical_icu_recovery_calibration_and_metadata_contracts(tmp_path):
    system = build_integrated_system(tmp_path)
    result = system.pipeline.predict(request())
    expected_hours = float(
        remaining_icu_hours_from_log_prediction(torch.tensor(3.0)).item()
    )
    assert result["icu_stay_time_hours"] == pytest.approx(expected_hours)
    assert result["recovery"] == {
        "delta_24h": 0.25,
        "delta_48h": -0.5,
        "reconstructed_sofa_24h": 4.25,
        "reconstructed_sofa_48h": 3.5,
    }
    assert result["organ_support_probability_calibrated"] == 0.25
    assert system.pipeline.bundle.support_threshold == 0.5
    assert system.runtime.calibrator.fit_calls == 0
    assert result["model_metadata"]["manifest_version"] == system.pipeline.bundle.manifest_version


def test_actual_dashboard_render_uses_bound_api_response(tmp_path):
    system = build_integrated_system(tmp_path)
    app = create_dashboard_app(system.controller, synthetic=True)
    response = TestClient(app).get(
        "/", params={"stay_id": MAIN_STAY, "prediction_time": MAIN_CUTOFF}
    )
    assert response.status_code == 200
    assert REPLAY_BANNER in response.text
    for exact in ("4.25", "3.5", "0.25", "integrated_gradients", "tree_shap"):
        assert exact in response.text
    assert "prediction confidence" not in response.text.lower()
    assert "precomputed" not in response.text.lower()


def test_clean_process_integration_has_no_hidden_import_state(tmp_path):
    script = """
from pathlib import Path
from tempfile import TemporaryDirectory
from integration_helpers import build_integrated_system
from phase7_helpers import MAIN_CUTOFF, MAIN_STAY
with TemporaryDirectory() as directory:
    system = build_integrated_system(Path(directory))
    result = system.pipeline.predict({'stay_id': MAIN_STAY, 'prediction_time': MAIN_CUTOFF})
    assert result['schema_version'] == 'prediction_schema_v1'
    print(result['mode'])
"""
    environment = dict(os.environ)
    environment["PYTHONPATH"] = "src:tests:."
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "RETROSPECTIVE_SEQUENTIAL_REPLAY"


def test_integration_sources_are_portable_and_do_not_access_final_test():
    sources = "\n".join(
        path.read_text() for path in (ROOT / "tests").glob("*integration*.py")
    )
    assert "/" + "Users/" not in sources
    assert "C:\\" + "Users\\" not in sources
    source = (ROOT / "tests/test_integration.py").read_text().lower()
    assert "final_" + "test_entrypoint" not in source
    assert "freeze_" + "g3" not in source
