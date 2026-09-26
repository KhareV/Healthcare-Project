"""V2 FastAPI composition root: real, model-backed retrospective replay API
for the four frozen Performance-V2 XGBoost tasks."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from serving.v2.ai_recommendation import generate_assistant_response, generate_recommendation
from serving.v2.auth import AuthError, ClerkAuthenticator
from serving.v2.custom_record import (
    CustomRecordError, build_custom_record, canonical_concepts, get_custom_record,
    list_custom_records, rehydrate_if_needed,
)
from serving.v2.explanations import explain
from serving.v2.guard import UnknownDemoStayError, load_demo_manifest
from serving.v2.persistence import MongoPersistence
from serving.v2.runtime import IllegalCutoffError, V2ServingError, V2ServingRuntime

logger = logging.getLogger("api.v2")


class PredictRequest(BaseModel):
    stay_id: str = Field(..., min_length=1)
    prediction_time: str = Field(..., min_length=1)

    model_config = {"extra": "forbid"}


class AssistantRequest(BaseModel):
    stay_id: str = Field(..., min_length=1)
    prediction_time: str = Field(..., min_length=1)
    question: Optional[str] = Field(default=None, max_length=500)
    previous_prediction_time: Optional[str] = Field(default=None)

    model_config = {"extra": "forbid"}


class CustomObservationIn(BaseModel):
    concept: str = Field(..., min_length=1)
    hours_since_admission: float
    value: float

    model_config = {"extra": "forbid"}


class CustomSupportIntervalIn(BaseModel):
    kind: str = Field(..., min_length=1)  # "vasopressor" | "ventilation"
    agent: Optional[str] = None  # required for vasopressor
    rate: Optional[float] = None  # ug/kg/min, required for vasopressor
    start_hour: float
    end_hour: Optional[float] = None  # null = currently active

    model_config = {"extra": "forbid"}


class CustomRecordRequest(BaseModel):
    patient_alias: str = Field(..., min_length=1, max_length=64)
    age_years: int = Field(..., ge=0, le=120)
    sex_category: str = Field(..., min_length=1, max_length=32)
    observations: list[CustomObservationIn] = Field(..., min_length=1)
    support_intervals: list[CustomSupportIntervalIn] = Field(default_factory=list)

    model_config = {"extra": "forbid"}


class ConditionIn(BaseModel):
    label: str = Field(..., min_length=1, max_length=200)
    diagnosed_year: Optional[int] = Field(default=None, ge=1900, le=2100)
    status: str = Field(default="active", min_length=1, max_length=32)

    model_config = {"extra": "forbid"}


def _load_demo_patient_aliases(root: Path) -> dict:
    """Product-layer alias map (stay_id -> DEMO-CARDIAC-NNN + display fields).

    Additive, non-scientific: read from artifacts/performance_v2/product/
    demo_patients_v1.json (built by scripts/product_v2_build_demo_patients.py).
    Falls back to raw stay_id if the product artifact is missing so the app
    never crashes on this being absent.
    """

    path = root / "artifacts" / "performance_v2" / "product" / "demo_patients_v1.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {row["stay_id"]: row for row in payload.get("demo_patients", [])}


def _explanation_payload(runtime: V2ServingRuntime, task: str, matrix, feature_names):
    positive, negative, diagnostics = explain(runtime.models[task], matrix, feature_names)
    return {
        "top_positive_contributors": [{"feature_name": item.feature_name, "label": item.readable_label, "attribution": item.attribution} for item in positive],
        "top_negative_contributors": [{"feature_name": item.feature_name, "label": item.readable_label, "attribution": item.attribution} for item in negative],
        "method": "tree_shap",
        "diagnostics": diagnostics,
    }


def _build_prediction_payload(runtime: V2ServingRuntime, stay_id: str, prediction_time: str, persistence: Optional[MongoPersistence] = None) -> dict:
    prediction = runtime.predict(stay_id, prediction_time)
    explanations = {
        task: _explanation_payload(runtime, task, prediction.matrices[task], prediction.feature_names[task])
        for task in ("recovery24", "recovery48", "icu_stay_time", "organ_support")
    }
    data_quality = _data_quality(prediction.feature_row)
    payload = {
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
    owner = runtime.ephemeral_owner(stay_id)
    if owner is not None and persistence is not None and persistence.enabled:
        # Only custom records get a persisted prediction-history trail --
        # demo patients are already permanently described by the frozen
        # manifest artifact, so there is nothing new to make durable there.
        # A write failure here must never break the response itself.
        try:
            persistence.save_prediction_run(
                owner_user_id=owner, stay_id=stay_id, prediction_time=prediction_time,
                snapshot={
                    "current_sofa": prediction.current_sofa,
                    "delta_sofa_24": prediction.recovery24_delta,
                    "delta_sofa_48": prediction.recovery48_delta,
                    "predicted_sofa_24h": prediction.recovery24_sofa,
                    "predicted_sofa_48h": prediction.recovery48_sofa,
                    "remaining_icu_hours": prediction.icu_remaining_hours,
                    "support_raw_probability": prediction.support_raw_probability,
                    "support_calibrated_probability": prediction.support_calibrated_probability,
                    "support_threshold": prediction.support_threshold,
                    "support_alert": prediction.support_alert,
                },
            )
        except Exception:  # noqa: BLE001
            logger.warning("failed to persist prediction run for %s @ %s", stay_id, prediction_time)
    return payload


def build_v2_app(root: Path) -> FastAPI:
    root = Path(root).resolve()
    runtime = V2ServingRuntime(root)
    demo_aliases = _load_demo_patient_aliases(root)
    authenticator = ClerkAuthenticator(publishable_key=os.environ.get("PUBLIC_CLERK_PUBLISHABLE_KEY"))
    persistence = MongoPersistence(uri=os.environ.get("MONGODB_URI"))
    app = FastAPI(title="Performance-V2 Retrospective Replay API")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def _authenticate(authorization: Optional[str]) -> str:
        """Verified Clerk user id, or the fixed local-dev owner id if this
        backend has no Clerk key configured at all. Raises HTTPException(401)
        for a missing/invalid/expired token when Clerk IS configured."""

        try:
            return authenticator.verify(authorization)
        except AuthError as exc:
            raise HTTPException(status_code=401, detail=str(exc)) from exc

    def _authorize_custom_stay(stay_id: str, authorization: Optional[str]) -> None:
        """If stay_id is an ephemeral custom record, require a verified
        token whose user id matches the record's owner -- otherwise raise
        UnknownDemoStayError (-> 404), the same fail-closed response used
        for a stay that never existed, so an unauthorized caller can never
        distinguish "not yours" from "doesn't exist". Demo/unknown stays are
        untouched: this is a no-op for them, exactly matching current
        product behavior (demo patients remain globally servable).

        Before checking ownership, attempts to rehydrate stay_id from
        persistent storage if it is a custom record this process has not
        loaded yet (e.g. right after a restart) -- see
        serving.v2.custom_record.rehydrate_if_needed. This never weakens
        the ownership check below: rehydration only restores a stay under
        its true recorded owner, taken from the record itself, never from
        the current caller."""

        rehydrate_if_needed(runtime, persistence, stay_id)
        owner = runtime.ephemeral_owner(stay_id)
        if owner is None:
            return
        try:
            caller = authenticator.verify(authorization)
        except AuthError as exc:
            raise UnknownDemoStayError("unknown stay_id") from exc
        if caller != owner:
            raise UnknownDemoStayError("unknown stay_id")

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "ready": True,
            "mode": "RETROSPECTIVE_SEQUENTIAL_REPLAY",
            "scope": "PERFORMANCE_V2_SYNTHETIC_BENCHMARK",
            "tasks": ["recovery24", "recovery48", "icu_stay_time", "organ_support"],
            "auth_mode": "clerk_verified" if authenticator.configured else "local_dev_no_auth",
            "persistence_mode": "mongodb" if persistence.enabled else "in_memory_only",
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
    def predict(request: PredictRequest, authorization: Optional[str] = Header(default=None)):
        _authorize_custom_stay(request.stay_id, authorization)
        return _build_prediction_payload(runtime, request.stay_id, request.prediction_time, persistence=persistence)

    @app.get("/demo-subjects")
    def demo_subjects():
        manifest = load_demo_manifest(root)
        subjects = [
            {**subject, "patient_alias": demo_aliases.get(subject["stay_id"], {}).get("patient_alias", subject["stay_id"])}
            for subject in manifest["demo_subjects"]
        ]
        return {
            "status": manifest["status"],
            "selection_criteria": manifest["selection_criteria"],
            "demo_subjects": subjects,
        }

    @app.get("/custom-records/schema")
    def custom_records_schema():
        """The canonical concepts a custom record may report -- unit and
        provenance_id are fixed by the frozen feature schema; the caller
        only ever supplies a concept name, a value, and a relative time."""

        return {"concepts": canonical_concepts(runtime.root)}

    @app.post("/custom-records")
    def create_custom_record(request: CustomRecordRequest, authorization: Optional[str] = Header(default=None)):
        owner_user_id = _authenticate(authorization)
        try:
            return build_custom_record(
                runtime,
                owner_user_id=owner_user_id,
                patient_alias=request.patient_alias,
                age_years=request.age_years,
                sex_category=request.sex_category,
                observations=[o.model_dump() for o in request.observations],
                support_intervals=[s.model_dump() for s in request.support_intervals],
                persistence=persistence,
            )
        except CustomRecordError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/custom-records")
    def list_own_custom_records(authorization: Optional[str] = Header(default=None)):
        """Every custom record owned by the caller -- durable across
        restarts when MongoDB persistence is configured (see /health's
        persistence_mode), otherwise whatever this process currently has
        loaded in memory."""

        owner_user_id = _authenticate(authorization)
        return {"records": list_custom_records(runtime, persistence, owner_user_id)}

    @app.get("/custom-records/{stay_id}")
    def read_custom_record(stay_id: str, authorization: Optional[str] = Header(default=None)):
        _authorize_custom_stay(stay_id, authorization)
        record = get_custom_record(runtime, stay_id)
        if record is None:
            raise UnknownDemoStayError("unknown custom record")
        return record

    @app.get("/custom-records/{stay_id}/predictions")
    def custom_record_prediction_history(stay_id: str, authorization: Optional[str] = Header(default=None)):
        """Persisted prediction-run snapshots for this encounter, one per
        distinct cutoff actually replayed, oldest first. Empty (with an
        explanatory note, never a fake history) when persistence is not
        configured."""

        _authorize_custom_stay(stay_id, authorization)
        owner = runtime.ephemeral_owner(stay_id)
        if owner is None:
            raise UnknownDemoStayError("unknown custom record")
        if not persistence.enabled:
            return {"stay_id": stay_id, "predictions": [], "note": "persistence is not configured; prediction history is unavailable"}
        return {"stay_id": stay_id, "predictions": persistence.list_prediction_runs(owner_user_id=owner, stay_id=stay_id)}

    @app.post("/health-record/conditions")
    def add_condition(request: ConditionIn, authorization: Optional[str] = Header(default=None)):
        owner_user_id = _authenticate(authorization)
        if not persistence.enabled:
            raise HTTPException(status_code=503, detail="persistence is not configured; conditions cannot be saved")
        return persistence.add_condition(
            owner_user_id=owner_user_id, label=request.label, diagnosed_year=request.diagnosed_year, status=request.status,
        )

    @app.get("/health-record/conditions")
    def list_conditions(authorization: Optional[str] = Header(default=None)):
        owner_user_id = _authenticate(authorization)
        if not persistence.enabled:
            return {"conditions": [], "note": "persistence is not configured"}
        return {"conditions": persistence.list_conditions(owner_user_id=owner_user_id)}

    @app.delete("/health-record/conditions/{condition_id}")
    def delete_condition(condition_id: str, authorization: Optional[str] = Header(default=None)):
        owner_user_id = _authenticate(authorization)
        if not persistence.enabled:
            raise HTTPException(status_code=503, detail="persistence is not configured")
        deleted = persistence.delete_condition(owner_user_id=owner_user_id, condition_id=condition_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="unknown condition_id")
        return {"deleted": True, "condition_id": condition_id}

    @app.get("/performance")
    def performance():
        """Reads the frozen, one-time Phase-4 fresh-test evaluation artifacts
        directly. Triggers no inference of any kind."""

        def _read(relative: str):
            return json.loads((root / relative).read_text(encoding="utf-8"))

        return {
            "final_metrics": _read("artifacts/performance_v2/phase4/metrics/final_metrics_v2.json"),
            "bootstrap": _read("artifacts/performance_v2/phase4/bootstrap/final_bootstrap_v2.json"),
            "naive_comparison": _read("artifacts/performance_v2/phase4/metrics/naive_comparison_v2.json"),
            "calibration_evidence": _read("artifacts/performance_v2/phase4/metrics/calibration_evidence_v2.json"),
            "generalization_comparison": _read("artifacts/performance_v2/phase4/metrics/generalization_comparison_v2.json"),
            "v1_historical_test": {
                "recovery24_mae": 1.121102021240419,
                "recovery48_mae": 1.5423799902108615,
                "icu_median_ae_hours": 9.479719411307386,
                "support_calibrated_auprc": 0.633735668141501,
            },
            "note": "v1 and v2 used different independent final-test cohorts; this is a historical comparison, not paired statistical testing.",
        }

    @app.get("/history")
    def history(stay_id: str, prediction_time: str, authorization: Optional[str] = Header(default=None)):
        """Raw canonical events with event_time <= prediction_time, for the
        historical-timeline panel. Reuses the same guard/legal-cutoff checks
        as /predict; returns no future data."""

        _authorize_custom_stay(stay_id, authorization)

        from datetime import datetime, timezone

        def _parse(value: str) -> datetime:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)

        cutoffs = runtime.legal_cutoffs(stay_id)  # raises UnknownDemoStayError for anything but a demo stay
        if prediction_time not in cutoffs:
            raise IllegalCutoffError(f"prediction_time {prediction_time!r} is not a legal cutoff for {stay_id}")
        cutoff_dt = _parse(prediction_time)
        events = runtime._events_by_stay.get(stay_id, ())
        visible = sorted(
            (
                {"event_time": row["event_time"], "canonical_concept": row["canonical_concept"], "value_numeric": row["value_numeric"], "unit": row.get("unit", "")}
                for row in events
                if _parse(row["event_time"]) <= cutoff_dt
            ),
            key=lambda row: row["event_time"],
        )
        return {"stay_id": stay_id, "prediction_time": prediction_time, "events": visible, "count": len(visible)}

    @app.post("/ai/recommendation")
    def ai_recommendation(request: PredictRequest, authorization: Optional[str] = Header(default=None)):
        _authorize_custom_stay(request.stay_id, authorization)
        prediction = _build_prediction_payload(runtime, request.stay_id, request.prediction_time, persistence=persistence)
        result = generate_recommendation(prediction=prediction)
        return result.as_dict()

    def _assistant_state(prediction: dict) -> dict:
        return {
            "prediction_time": prediction["prediction_time"],
            "current_sofa": prediction["current_sofa"],
            "predicted_sofa_24h": prediction["recovery"]["sofa_hat_24h"],
            "predicted_sofa_48h": prediction["recovery"]["sofa_hat_48h"],
            "remaining_icu_hours": prediction["icu_stay_time"]["remaining_hours"],
            "support_probability": prediction["organ_support"]["probability_24h"],
            "support_alert": prediction["organ_support"]["alert"],
        }

    def _assistant_context(prediction: dict) -> dict:
        alias_row = demo_aliases.get(prediction["stay_id"])
        if alias_row is None:
            custom = get_custom_record(runtime, prediction["stay_id"])
            alias_row = {
                "patient_alias": custom["patient_alias"],
                "cardiac_subtype": "custom record (direct entry)",
                "age_years": custom["age_years"],
                "sex_category": custom["sex_category"],
            } if custom else {}
        recovery, icu, support = prediction["recovery"], prediction["icu_stay_time"], prediction["organ_support"]
        explanations = prediction.get("explanations", {})

        def contributors_for(task: str):
            payload = explanations.get(task) or {}
            items = list(payload.get("top_positive_contributors", [])) + list(payload.get("top_negative_contributors", []))
            return [{"label": item["label"], "attribution": item["attribution"]} for item in items[:5]]

        return {
            "patient_alias": alias_row.get("patient_alias", prediction["stay_id"]),
            "prediction_time": prediction["prediction_time"],
            "episode": {
                "elapsed_hours": prediction["elapsed_icu_hours"],
                "cardiac_subtype": alias_row.get("cardiac_subtype", "unknown"),
                "age_years": alias_row.get("age_years"),
                "sex": alias_row.get("sex_category"),
            },
            "current_state": {"current_sofa": prediction["current_sofa"]},
            "forecasts": {
                "delta_sofa_24": recovery["delta_24h"],
                "delta_sofa_48": recovery["delta_48h"],
                "predicted_sofa_24h": recovery["sofa_hat_24h"],
                "predicted_sofa_48h": recovery["sofa_hat_48h"],
                "remaining_icu_hours": icu["remaining_hours"],
                "support_raw_probability": support["raw_probability"],
                "support_calibrated_probability": support["probability_24h"],
                "support_threshold": support["threshold"],
                "support_alert": support["alert"],
            },
            "top_contributors": {
                "recovery24": contributors_for("recovery24"),
                "recovery48": contributors_for("recovery48"),
                "icu": contributors_for("icu_stay_time"),
                "support": contributors_for("organ_support"),
            },
            "data_quality": prediction.get("data_quality", {}),
            "system_limitations": [
                "synthetic research benchmark",
                "retrospective replay, not real-time",
                "not clinically validated",
                "point forecasts for regression tasks (no per-prediction uncertainty interval)",
                "TreeSHAP attribution is descriptive/non-causal, computed on the raw model margin before calibration",
            ],
        }

    @app.post("/assistant")
    def assistant(request: AssistantRequest, authorization: Optional[str] = Header(default=None)):
        """Trajectory Copilot: a grounded interpretation layer over an
        already-computed prediction. Never predicts independently, never
        sees raw free text, never reaches the fresh-test cohort (the guard
        that protects /predict protects this identically, since it is built
        from the exact same _build_prediction_payload call)."""

        _authorize_custom_stay(request.stay_id, authorization)
        prediction = _build_prediction_payload(runtime, request.stay_id, request.prediction_time, persistence=persistence)
        context = _assistant_context(prediction)

        if request.previous_prediction_time:
            try:
                previous_prediction = _build_prediction_payload(runtime, request.stay_id, request.previous_prediction_time, persistence=persistence)
                context["previous_cutoff"] = _assistant_state(previous_prediction)
            except (IllegalCutoffError, UnknownDemoStayError):
                pass  # ignore an invalid previous cutoff rather than failing the whole request

        result = generate_assistant_response(context=context, question=request.question)
        return result.as_dict()

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
    app.state.v2_persistence = persistence
    return app


def app() -> FastAPI:
    """uvicorn --factory entrypoint: `uvicorn api.v2_app:app --factory`."""

    from dotenv import load_dotenv

    root = Path(__file__).resolve().parents[1]
    load_dotenv(root / ".env")
    return build_v2_app(root)


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
