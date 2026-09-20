from dataclasses import asdict
from pathlib import Path

import pytest

from serving.recovery import RecoveryServingError
from serving.recovery_display import recovery_display_data
from api_helpers import REQUEST, build_pipeline
from recovery_helpers import state


def test_dashboard_data_transform_only_binds_validated_prediction_values(tmp_path):
    _, pipeline, _ = build_pipeline(tmp_path)
    response = pipeline.predict(REQUEST)
    current = state(
        value=4.0,
        stay_id=REQUEST["stay_id"],
        prediction_time=REQUEST["prediction_time"],
    )
    first = recovery_display_data(response, current, synthetic=True)
    second = recovery_display_data(response, current, synthetic=True)
    assert first == second
    assert first.current_sofa == 4.0
    assert first.delta_24h == response["recovery"]["delta_24h"]
    assert first.predicted_sofa_48h == response["recovery"]["reconstructed_sofa_48h"]
    assert first.current_label.startswith("Observed current")
    assert first.predicted_24h_label.startswith("Predicted")
    assert not hasattr(first, "actual_sofa_24h")
    assert "confidence" not in asdict(first)


def test_dashboard_data_rejects_current_sofa_from_another_cutoff(tmp_path):
    _, pipeline, _ = build_pipeline(tmp_path)
    response = pipeline.predict(REQUEST)
    current = state(
        stay_id=REQUEST["stay_id"],
        prediction_time="2026-01-02T18:00:00+00:00",
    )
    with pytest.raises(RecoveryServingError, match="another prediction cutoff"):
        recovery_display_data(response, current, synthetic=True)


def test_view_transform_contains_no_ui_network_file_or_reconstruction_logic():
    root = Path(__file__).resolve().parents[1]
    source = (root / "src/serving/recovery_display.py").read_text().lower()
    for forbidden in (
        "streamlit", "plotly", "requests.", "httpx", "open(", "read_text",
        "torch", "reconstruct_absolute_sofa", "delta_24h +", "delta_48h +",
    ):
        assert forbidden not in source
    assert (root / "dashboard/app.py").is_file()
