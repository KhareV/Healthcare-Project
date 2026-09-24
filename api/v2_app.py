"""V2 FastAPI composition root: real, model-backed retrospective replay API
for the four frozen Performance-V2 XGBoost tasks."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from serving.v2.explanations import explain
from serving.v2.guard import UnknownDemoStayError
from serving.v2.runtime import IllegalCutoffError, V2ServingError, V2ServingRuntime

logger = logging.getLogger("api.v2")


class PredictRequest(BaseModel):
    stay_id: str = Field(..., min_length=1)
    prediction_time: str = Field(..., min_length=1)

    model_config = {"extra": "forbid"}


def _explanation_payload(runtime: V2ServingRuntime, task: str, matrix, feature_names):
    positive, negative, diagnostics = explain(runtime.models[task], matrix, feature_names)
    return {
        "top_positive_contributors": [{"feature_name": item.feature_name, "label": item.readable_label, "attribution": item.attribution} for item in positive],
        "top_negative_contributors": [{"feature_name": item.feature_name, "label": item.readable_label, "attribution": item.attribution} for item in negative],
        "method": "tree_shap",
        "diagnostics": diagnostics,
    }


def build_v2_app(root: Path) -> FastAPI:
    root = Path(root).resolve()
    runtime = V2ServingRuntime(root)
    app = FastAPI(title="Performance-V2 Retrospective Replay API")

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "ready": True,
            "mode": "RETROSPECTIVE_SEQUENTIAL_REPLAY",
            "scope": "PERFORMANCE_V2_SYNTHETIC_BENCHMARK",
            "tasks": ["recovery24", "recovery48", "icu_stay_time", "organ_support"],
        }

    @app.get("/model-metadata")
    def model_metadata():
        return {
            "schema_version": "v2_prediction_response_v1",
            "model_versions": {
                "recovery24": {"family": "xgboost", "feature_variant": "B_MIN", "artifact_sha256": runtime.model_hashes["recovery24"]},
                "recovery48": {"family": "xgboost", "feature_variant": "B_MIN", "artifact_sha256": runtime.model_hashes["recovery48"]},
                "icu_stay_time": {"family": "xgboost", "feature_variant": "B_PLUS_F", "artifact_sha256": runtime.model_hashes["icu_stay_time"]},
                "organ_support": {"family": "xgboost", "feature_variant": "B_FULL", "artifact_sha256": runtime.model_hashes["organ_support"]},
            },
            "support_threshold": 0.39781983118092895,
            "explanation_method": "tree_shap",
            "mode": "RETROSPECTIVE_SEQUENTIAL_REPLAY",
        }

    @app.post("/predict")
    def predict(request: PredictRequest):
        prediction = runtime.predict(request.stay_id, request.prediction_time)
        explanations = {
            task: _explanation_payload(runtime, task, prediction.matrices[task], prediction.feature_names[task])
            for task in ("recovery24", "recovery48", "icu_stay_time", "organ_support")
        }
        data_quality = _data_quality(prediction.feature_row)
        return {
            "schema_version": "v2_prediction_response_v1",
            "mode": "RETROSPECTIVE_SEQUENTIAL_REPLAY",
            "stay_id": prediction.stay_id,
            "prediction_time": prediction.prediction_time,
            "grid_index": prediction.grid_index,
            "elapsed_icu_hours": prediction.icu_elapsed_hours,
            "current_sofa": prediction.current_sofa,
            "recovery": {
                "delta_24h": prediction.recovery24_delta,
                "delta_48h": prediction.recovery48_delta,
                "sofa_hat_24h": prediction.recovery24_sofa,
                "sofa_hat_48h": prediction.recovery48_sofa,
                "model_metadata": {"family": "xgboost", "feature_variant": "B_MIN", "artifact_sha256_24h": prediction.model_hashes["recovery24"], "artifact_sha256_48h": prediction.model_hashes["recovery48"]},
            },
            "icu_stay_time": {
                "remaining_hours": prediction.icu_remaining_hours,
                "raw_log_prediction": prediction.icu_raw_log_prediction,
                "model_metadata": {"family": "xgboost", "feature_variant": "B_PLUS_F", "artifact_sha256": prediction.model_hashes["icu_stay_time"]},
            },
            "organ_support": {
                "raw_probability": prediction.support_raw_probability,
                "probability_24h": prediction.support_calibrated_probability,
                "threshold": prediction.support_threshold,
                "alert": prediction.support_alert,
                "model_metadata": {"family": "xgboost", "feature_variant": "B_FULL", "artifact_sha256": prediction.model_hashes["organ_support"]},
            },
            "explanations": explanations,
            "data_quality": data_quality,
            "temporal_window": {
                "channel_names": list(prediction.feature_row["temporal_feature_names"]),
                "observation_mask": [list(bin_row) for bin_row in prediction.feature_row["observation_mask"]],
                "padding_mask": list(prediction.feature_row.get("padding_mask", [False] * len(prediction.feature_row["observation_mask"]))),
            },
            "versions": {
                "selected_models_v2_sha256": "6fed05f3c71ce648658b2f864e69fa9f80f3ec8538f53a4e370fa8a4eaf176a8",
                "v2_model_freeze_sha256": "f5feb29cab2e25d6e8bff860af52ac505f40951578aa73547ec420ba8fcda570",
            },
        }

    @app.exception_handler(UnknownDemoStayError)
    def _unknown_stay(_request, _exc):
        return JSONResponse(status_code=404, content={"detail": "unknown stay_id"})

    @app.exception_handler(IllegalCutoffError)
    def _illegal_cutoff(_request, _exc):
        return JSONResponse(status_code=422, content={"detail": "illegal prediction_time"})

    @app.exception_handler(V2ServingError)
    def _serving_error(_request, _exc):
        logger.error("V2 serving artifact error")
        return JSONResponse(status_code=503, content={"detail": "serving artifacts unavailable"})

    @app.exception_handler(Exception)
    def _unhandled(_request, exc: Exception):
        logger.error("unhandled V2 API error: %s", type(exc).__name__)
        return JSONResponse(status_code=500, content={"detail": "internal error"})

    app.state.v2_runtime = runtime
    return app


def app() -> FastAPI:
    """uvicorn --factory entrypoint: `uvicorn api.v2_app:app --factory`."""

    return build_v2_app(Path(__file__).resolve().parents[1])


def _data_quality(row) -> dict:
    mask = row["observation_mask"]
    padding = row.get("padding_mask", [False] * len(mask))
    total_bins = len(mask)
    padding_bins = sum(1 for p in padding if p)
    observed_bins = sum(1 for bin_row, p in zip(mask, padding) if not p and any(bin_row))
    total_values = sum(len(bin_row) for bin_row in mask)
    observed_values = sum(1 for bin_row in mask for v in bin_row if v)
    return {
        "total_bins": total_bins,
        "observed_bins": observed_bins,
        "padding_bins": padding_bins,
        "total_feature_values": total_values,
        "observed_feature_values": observed_values,
        "missing_feature_values": total_values - observed_values,
        "observed_feature_fraction": observed_values / total_values if total_values else None,
    }
