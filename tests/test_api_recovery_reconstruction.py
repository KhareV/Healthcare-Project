import json

from api_helpers import REQUEST, build_pipeline, client_for
from phase7_helpers import MAIN_CUTOFF, MAIN_STAY, TIMELINE_FIXTURE, build_provider, load_timelines
from recovery_helpers import SOFA_VERSION, SpyCurrentSOFAProvider, SpyOriginalUnitDeltaAdapter, state
from serving.postprocessing import CanonicalICUTimeServingPostprocessor
from serving.recovery import RecoveryServingPostprocessor
from serving.pipeline import PredictionPipeline
from serving_helpers import FixtureRuntime, build_kwargs, copy_serving_fixture


def phase12_pipeline(tmp_path, *, current_provider=None, input_provider=None):
    root = copy_serving_fixture(tmp_path)
    runtime = FixtureRuntime()
    if input_provider is not None:
        runtime.input_provider = input_provider
    current_provider = current_provider or SpyCurrentSOFAProvider(
        state(
            value=10.0,
            stay_id=REQUEST["stay_id"],
            prediction_time=REQUEST["prediction_time"],
        )
    )
    delta_adapter = SpyOriginalUnitDeltaAdapter()
    recovery = RecoveryServingPostprocessor(
        current_sofa_provider=current_provider,
        delta_adapter=delta_adapter,
        expected_sofa_version=SOFA_VERSION,
    )
    pipeline = PredictionPipeline.build(
        **build_kwargs(root, runtime),
        input_provider=runtime.input_provider,
        postprocessor=CanonicalICUTimeServingPostprocessor(recovery),
        explanation_adapters=runtime.explanation_adapters,
    )
    return pipeline, runtime, current_provider, delta_adapter


def test_api_and_direct_pipeline_recovery_are_identical_and_schema_valid(tmp_path):
    pipeline, _, current, adapter = phase12_pipeline(tmp_path)
    direct = pipeline.predict(REQUEST)
    http = client_for(pipeline).post("/predict", json=REQUEST)
    assert http.status_code == 200
    assert http.json()["recovery"] == json.loads(json.dumps(direct["recovery"]))
    assert direct["recovery"] == {
        "delta_24h": 0.25,
        "delta_48h": -0.5,
        "reconstructed_sofa_24h": 10.25,
        "reconstructed_sofa_48h": 9.5,
    }
    assert current.calls == [
        (REQUEST["stay_id"], REQUEST["prediction_time"]),
        (REQUEST["stay_id"], REQUEST["prediction_time"]),
    ]
    assert len(adapter.calls) == 2


def test_future_event_perturbation_cannot_change_phase12_api_reconstruction(tmp_path):
    payload = json.loads(TIMELINE_FIXTURE.read_text())
    request = {"stay_id": MAIN_STAY, "prediction_time": MAIN_CUTOFF}
    baseline_input = build_provider(timelines=load_timelines(payload))[0]
    baseline_state = SpyCurrentSOFAProvider(
        state(value=8.0, stay_id=MAIN_STAY, prediction_time=MAIN_CUTOFF)
    )
    baseline, _, _, _ = phase12_pipeline(
        tmp_path / "baseline", current_provider=baseline_state,
        input_provider=baseline_input,
    )
    expected = client_for(baseline).post("/predict", json=request).json()

    events = payload["timelines"][0]["events"]
    retained = [row for row in events if row["synthetic_available_time"] <= MAIN_CUTOFF]
    payload["timelines"][0]["events"] = [
        {
            "synthetic_stay_key": MAIN_STAY,
            "synthetic_available_time": "2026-01-05T18:00:00+00:00",
            "synthetic_feature": "SYNTHETIC_SIGNAL_A",
            "synthetic_value": -1e300,
            "synthetic_sequence": 1000,
            "actual_future_sofa": 24,
        }
    ] + list(reversed(retained))
    changed_input = build_provider(timelines=load_timelines(payload))[0]
    changed_state = SpyCurrentSOFAProvider(
        state(value=8.0, stay_id=MAIN_STAY, prediction_time=MAIN_CUTOFF)
    )
    changed, _, _, _ = phase12_pipeline(
        tmp_path / "changed", current_provider=changed_state,
        input_provider=changed_input,
    )
    actual = client_for(changed).post("/predict", json=request).json()
    assert actual == expected
    assert baseline_state.calls == changed_state.calls == [(MAIN_STAY, MAIN_CUTOFF)]


def test_api_contains_no_reconstruction_arithmetic():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    source = (root / "api/main.py").read_text().lower()
    assert "reconstruct_absolute_sofa" not in source
    assert "current_sofa" not in source
    assert "delta_24h +" not in source
    assert "delta_48h +" not in source
