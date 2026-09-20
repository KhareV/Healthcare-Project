import copy
import json
from pathlib import Path

from dashboard.app import DEFAULT_FIXTURE
from dashboard.catalog import load_synthetic_dashboard_catalog
from dashboard.replay import ReplayController


ROOT = Path(__file__).resolve().parents[1]
RESPONSE_FIXTURE = ROOT / "tests/fixtures/prediction_schema/mixed_family_response_v1.json"


def response_for(cutoff):
    payload = json.loads(RESPONSE_FIXTURE.read_text())["payload"]
    payload["prediction_time"] = cutoff
    return payload


def metadata():
    response = response_for("2026-02-02T00:00:00+00:00")
    tasks = {}
    for task, version in response["model_versions"].items():
        explanation = response["explanation_features"][task]
        tasks[task] = {
            "family": version["family"],
            "model_version": version["model_version"],
            "artifact_sha256": version["artifact_sha256"],
            "preprocessor_sha256": version["preprocessor_sha256"],
            "feature_version": response["model_metadata"]["feature_version"][task],
            "label_version": response["model_metadata"]["label_version"][task],
            "split_hash": "SYNTHETIC_SPLIT_HASH_NOT_REAL",
            "explanation_method": explanation["explanation_method"],
        }
    return {
        "mode": "RETROSPECTIVE_SEQUENTIAL_REPLAY",
        "serving_scope": "synthetic",
        "manifest_version": response["model_metadata"]["manifest_version"],
        "manifest_sha256": "SYNTHETIC_MANIFEST_HASH_NOT_REAL",
        "split_version": response["model_metadata"]["split_version"],
        "tasks": tasks,
        "organ_support": {
            "calibrator_sha256": response["model_versions"]["organ_support"]["calibrator_sha256"],
            "threshold_sha256": "SYNTHETIC_THRESHOLD_HASH_NOT_REAL",
            "threshold": 0.4,
            "comparator": "greater_than_or_equal",
        },
    }


class FakeDashboardAPI:
    def __init__(self, mutate=None):
        self.calls = []
        self.mutate = mutate

    def get_model_metadata(self):
        self.calls.append(("GET", "/model-metadata"))
        value = metadata()
        if self.mutate:
            self.mutate("metadata", value)
        return value

    def get_health(self):
        self.calls.append(("GET", "/health"))
        return {
            "status": "alive", "ready": True, "serving_scope": "synthetic",
            "mode": "RETROSPECTIVE_SEQUENTIAL_REPLAY",
        }

    def predict(self, stay_id, prediction_time):
        self.calls.append(("POST", "/predict", stay_id, prediction_time))
        value = response_for(prediction_time)
        if self.mutate:
            self.mutate("prediction", value)
        return value


def controller(api=None):
    return ReplayController(
        load_synthetic_dashboard_catalog(DEFAULT_FIXTURE),
        api or FakeDashboardAPI(),
    )
