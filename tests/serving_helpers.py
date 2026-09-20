import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import torch

from experiments.search_governance import canonical_sha256
from explainability.router import AdapterExplanation
from models.recovery_output import reconstruct_absolute_sofa
from vedant_infra.hashing import sha256_file


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_RELATIVE = Path("tests/fixtures/serving")
MANIFEST_NAME = "selected_models_synthetic_phase6_v1.json"


def copy_serving_fixture(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    target = root / FIXTURE_RELATIVE
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(PROJECT_ROOT / FIXTURE_RELATIVE, target)
    experiments = root / "experiments"
    experiments.mkdir(parents=True, exist_ok=True)
    shutil.copy2(PROJECT_ROOT / "experiments/artifacts.csv", experiments / "artifacts.csv")
    return root


def manifest_path(root: Path) -> Path:
    return root / FIXTURE_RELATIVE / MANIFEST_NAME


def read_json(path: Path):
    return json.loads(path.read_text())


def write_json(path: Path, payload) -> str:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return sha256_file(path)


def rehash_manifest(root: Path, mutate=None) -> str:
    path = manifest_path(root)
    payload = read_json(path)
    if mutate is not None:
        mutate(payload)
    payload.pop("manifest_sha256", None)
    payload["manifest_sha256"] = canonical_sha256(payload)
    return write_json(path, payload)


@dataclass
class MockPredictor:
    task: str
    family: str
    model_version: str
    artifact_sha256: str
    predict_calls: int = 0
    last_prepared_input: object = None

    def predict(self, prepared_input):
        self.predict_calls += 1
        self.last_prepared_input = prepared_input
        if self.task == "recovery":
            return (0.25, -0.5)
        if self.task == "icu_stay_time":
            return 3.0
        if self.task == "organ_support":
            return 0.25
        raise AssertionError("unknown task")


@dataclass
class MockPreprocessor:
    task: str
    artifact_sha256: str
    transform_calls: int = 0

    def transform(self, prepared_input):
        self.transform_calls += 1
        return prepared_input


@dataclass
class MockCalibrator:
    artifact_sha256: str
    transform_calls: int = 0
    fit_calls: int = 0

    def transform(self, raw_probability):
        self.transform_calls += 1
        return float(raw_probability)

    def fit(self, *_args, **_kwargs):
        self.fit_calls += 1
        raise AssertionError("serving must never fit calibration")


class MockInputProvider:
    def __init__(self):
        self.calls = []

    def get_canonical_input(
        self, *, stay_id, prediction_time, task, family, feature_version
    ):
        self.calls.append((stay_id, prediction_time, task, family, feature_version))
        return {"synthetic": True, "task": task, "baseline_sofa": 4.0}

    def data_quality(self, *, stay_id, prediction_time):
        return {
            "total_bins": 8,
            "observed_bins": 6,
            "padding_bins": 2,
            "total_feature_values": 80,
            "observed_feature_values": 48,
            "missing_feature_values": 32,
        }


class MockPostprocessor:
    def recovery(self, raw_output, prepared_input, **_context):
        delta24, delta48 = raw_output
        baseline = prepared_input["baseline_sofa"]
        display = reconstruct_absolute_sofa(
            torch.tensor([baseline], dtype=torch.float64),
            torch.tensor([[delta24, delta48]], dtype=torch.float64),
            clip_for_display=True,
        )[0]
        return {
            "delta_24h": delta24,
            "delta_48h": delta48,
            "reconstructed_sofa_24h": float(display[0].item()),
            "reconstructed_sofa_48h": float(display[1].item()),
        }

    def icu_stay_time_hours(self, raw_output):
        assert raw_output == 3.0
        return 42.0


class MockExplanationProvider:
    synthetic = True

    def __init__(self):
        self.calls = []
        self.contexts = []

class MockExplanationAdapter:
    synthetic = True

    def __init__(self, method, family, collector):
        self.method = method
        self.supported_families = (family,)
        self.collector = collector

    def explain(self, context):
        assert context.synthetic is True
        self.collector.calls.append((context.task, context.family, context.model_sha256))
        self.collector.contexts.append(context)
        return AdapterExplanation(
            task=context.task,
            family=context.family,
            method=self.method,
            model_sha256=context.model_sha256,
            prediction_time=context.prediction_time,
            manifest_version=context.manifest_version,
            manifest_sha256=context.manifest_sha256,
            feature_schema_version=context.feature_schema_version,
            synthetic=True,
            items=({"feature_name": "SYNTHETIC_FEATURE_" + context.task.upper(), "attribution": 0.125},),
        )


class FixtureRuntime:
    def __init__(self):
        self.loader_calls = []
        self.predictors = {}
        self.preprocessors = {}
        self.calibrator = None
        self.input_provider = MockInputProvider()
        self.postprocessor = MockPostprocessor()
        self.explanations = MockExplanationProvider()
        self.explanation_adapters = {
            "integrated_gradients": MockExplanationAdapter(
                "integrated_gradients", "gru", self.explanations
            ),
            "tree_shap": MockExplanationAdapter(
                "tree_shap", "xgboost", self.explanations
            ),
        }

    def model_loader(self, family):
        def load(task, path, metadata, identity):
            assert path.is_file()
            self.loader_calls.append((family, task))
            predictor = MockPredictor(
                task, family, identity.model_version, identity.artifact_sha256
            )
            self.predictors[task] = predictor
            return predictor
        return load

    def preprocessor_loader(self, task, path, metadata, identity):
        assert metadata["fit_partition"] == "train"
        preprocessor = MockPreprocessor(task, identity.preprocessor_sha256)
        self.preprocessors[task] = preprocessor
        return preprocessor

    def calibrator_loader(self, path, metadata):
        assert metadata["transform"] == "identity_for_orchestration_test_only"
        self.calibrator = MockCalibrator(sha256_file(path))
        return self.calibrator

    @property
    def model_loaders(self):
        return {"gru": self.model_loader("gru"), "xgboost": self.model_loader("xgboost")}


def build_kwargs(root: Path, runtime: FixtureRuntime):
    manifest = manifest_path(root)
    return {
        "root": root,
        "manifest_ref": str(manifest.relative_to(root)),
        "expected_manifest_sha256": sha256_file(manifest),
        "scope": "synthetic",
        "model_loaders": runtime.model_loaders,
        "preprocessor_loader": runtime.preprocessor_loader,
        "calibrator_loader": runtime.calibrator_loader,
    }
