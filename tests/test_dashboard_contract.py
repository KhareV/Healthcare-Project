import pytest
from fastapi.testclient import TestClient

from dashboard.app import create_dashboard_app
from dashboard.api_client import DashboardAPIError
from dashboard.view_models import REPLAY_BANNER
from tests.dashboard_helpers import FakeDashboardAPI, controller


def test_dashboard_renders_all_required_panels_from_one_api_response():
    api = FakeDashboardAPI()
    replay = controller(api)
    cutoff = replay.available_stays and (
        "2026-02-02T06:00:00+00:00"
    )
    client = TestClient(create_dashboard_app(lambda: controller(api), synthetic=True))
    response = client.get(
        "/", params={"stay_id": "SYNTHETIC_PHASE13_STAY_A", "prediction_time": cutoff}
    )
    assert response.status_code == 200
    text = response.text
    for required in (
        REPLAY_BANNER,
        "SYNTHETIC DEMO / NON-SCIENTIFIC",
        "Replay overview",
        "History available at cutoff",
        "Observed current SOFA at cutoff",
        "Predicted SOFA +24h",
        "Predicted SOFA +48h",
        "Remaining ICU stay time",
        "Remaining time until current ICU stay ends",
        "New Organ-Support Initiation Risk",
        "Calibrated probability",
        "Top contributors to this model prediction",
        "Data quality",
        "Per-task model metadata",
    ):
        assert required in text
    assert "Synthetic future lab" not in text
    assert len([call for call in api.calls if call[0] == "POST"]) == 1


def test_default_application_is_fail_closed_and_not_synthetic():
    client = TestClient(create_dashboard_app())
    response = client.get("/")
    assert response.status_code == 503
    assert REPLAY_BANNER in response.text
    assert "SYNTHETIC DEMO" not in response.text


def test_dashboard_source_has_no_model_or_scientific_reimplementation():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    source = "\n".join(
        path.read_text().lower() for path in (root / "dashboard").glob("*.py")
    )
    for forbidden in (
        "predictionpipeline", "torch.load", "xgboost", "treeexplainer",
        "integratedgradients", "isotonic", "sofa_at(", "choose_support_threshold",
        "outtime-t", "hospital discharge",
    ):
        assert forbidden not in source


def test_recovery_icu_support_and_mixed_explanation_values_are_passed_through():
    client = TestClient(create_dashboard_app(lambda: controller(), synthetic=True))
    response = client.get(
        "/",
        params={
            "stay_id": "SYNTHETIC_PHASE13_STAY_A",
            "prediction_time": "2026-02-02T00:00:00+00:00",
        },
    )
    text = response.text
    for exact in ("10.0", "4.25", "3.5", "42.0 hours", "0.25", "0.4"):
        assert exact in text
    assert "recovery — gru / integrated_gradients" in text
    assert "icu_stay_time — xgboost / tree_shap" in text
    assert "organ_support — gru / integrated_gradients" in text
    assert "actual SOFA" not in text
    assert "outtime" not in text.lower()


def test_service_errors_render_safe_message_without_stale_panels():
    class FailedAPI(FakeDashboardAPI):
        def get_model_metadata(self):
            raise DashboardAPIError(
                503,
                "artifact_unavailable",
                "serving artifact dependency unavailable",
            )

    client = TestClient(
        create_dashboard_app(lambda: controller(FailedAPI()), synthetic=True)
    )
    response = client.get(
        "/",
        params={
            "stay_id": "SYNTHETIC_PHASE13_STAY_A",
            "prediction_time": "2026-02-02T00:00:00+00:00",
        },
    )
    assert response.status_code == 422
    assert "serving artifact dependency unavailable" in response.text
    assert "<h2>Recovery trajectory</h2>" not in response.text
    assert "Traceback" not in response.text


def test_unknown_stay_is_clean_and_does_not_substitute_another_stay():
    client = TestClient(create_dashboard_app(lambda: controller(), synthetic=True))
    response = client.get("/", params={"stay_id": "UNKNOWN"})
    assert response.status_code == 422
    assert "replay selection is unavailable or incompatible" in response.text
    assert "Replay overview" not in response.text


@pytest.mark.parametrize(
    ("status", "message"),
    (
        (404, "selected stay is unavailable"),
        (422, "selected cutoff is illegal"),
        (503, "serving artifacts are unavailable"),
    ),
)
def test_api_error_states_render_without_prediction_panels(status, message):
    class FailedAPI(FakeDashboardAPI):
        def get_model_metadata(self):
            raise DashboardAPIError(status, "request_failed", message)

    client = TestClient(
        create_dashboard_app(lambda: controller(FailedAPI()), synthetic=True)
    )
    response = client.get(
        "/",
        params={
            "stay_id": "SYNTHETIC_PHASE13_STAY_A",
            "prediction_time": "2026-02-02T00:00:00+00:00",
        },
    )
    assert message in response.text
    assert "<h2>Recovery trajectory</h2>" not in response.text
