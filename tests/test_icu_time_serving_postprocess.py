import math

import pytest
import torch

import serving.postprocessing as postprocessing
from api_helpers import REQUEST, build_pipeline, client_for
from models.icu_time_postprocess import remaining_icu_hours_from_log_prediction
from serving.postprocessing import CanonicalICUTimeServingPostprocessor, ServingPostprocessError
from serving_helpers import MockPostprocessor


def postprocessor():
    return CanonicalICUTimeServingPostprocessor(MockPostprocessor())


@pytest.mark.parametrize(
    "raw,expected",
    ((-1.0, 0.0), (0.0, 0.0), (math.log(2.0), 1.0), (math.log(11.0), 10.0)),
)
def test_serving_and_evaluation_share_exact_icu_transform(raw, expected):
    serving = postprocessor().icu_stay_time_hours(raw)
    evaluation = float(
        remaining_icu_hours_from_log_prediction(torch.tensor(raw, dtype=torch.float64)).item()
    )
    assert serving == evaluation == pytest.approx(expected)


def test_serving_calls_the_canonical_function_object(monkeypatch):
    calls = []

    def spy(value):
        calls.append(value.clone())
        return torch.tensor(7.0, dtype=torch.float64)

    monkeypatch.setattr(postprocessing, "remaining_icu_hours_from_log_prediction", spy)
    assert postprocessor().icu_stay_time_hours(2.0) == 7.0
    assert len(calls) == 1 and calls[0].item() == 2.0


@pytest.mark.parametrize("raw", (float("nan"), float("inf"), -float("inf")))
def test_nonfinite_icu_output_fails_closed(raw):
    with pytest.raises(ServingPostprocessError):
        postprocessor().icu_stay_time_hours(raw)


def test_nonfinite_icu_output_becomes_safe_http_503(tmp_path):
    _, pipeline, runtime = build_pipeline(tmp_path)
    original = runtime.predictors["icu_stay_time"].predict

    def invalid(_prepared):
        runtime.predictors["icu_stay_time"].predict_calls += 1
        return float("nan")

    runtime.predictors["icu_stay_time"].predict = invalid
    response = client_for(pipeline).post("/predict", json=REQUEST)
    assert response.status_code == 503
    assert "NaN" not in response.text and "Infinity" not in response.text


def test_icu_terminology_is_current_stay_hours_only():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    text = "\n".join(
        (root / path).read_text().lower()
        for path in ("api/main.py", "api/schemas.py", "src/serving/postprocessing.py")
    )
    for forbidden in ("hospital discharge eta", "survival time", "recovery time", "discharge prediction"):
        assert forbidden not in text
