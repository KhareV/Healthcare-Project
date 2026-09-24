"""V2 dashboard composition root. Every panel is populated through genuine
API calls (dashboard -> FastAPI -> V2PredictionPipeline); there is no local
prediction cache or lookup table. The Model Performance view is the sole
exception -- it reads already-frozen Phase-4 evaluation artifacts directly
and performs no inference of any kind.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Mapping, Optional

from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse

from dashboard import v2_charts, v2_render
from dashboard.v2_api_client import HTTPV2DashboardAPIClient, V2DashboardAPI

DEFAULT_V2_API_BASE_URL = "http://127.0.0.1:8010"


def _load_json(root: Path, relative: str):
    return json.loads((root / relative).read_text(encoding="utf-8"))


def _demo_subjects(root: Path):
    manifest = _load_json(root, "configs/performance_v2/v2_demo_manifest_v1.json")
    return manifest["demo_subjects"]


def _group_events(events, cutoff_time: str):
    groups = {"Cardiovascular": [], "Respiratory": [], "Renal": [], "Neurologic": [], "Hematologic/Liver": [], "Other": []}
    concept_group = {
        "mean_arterial_pressure": "Cardiovascular", "heart_rate": "Cardiovascular", "systolic_blood_pressure": "Cardiovascular",
        "diastolic_blood_pressure": "Cardiovascular", "vasopressor_on": "Cardiovascular", "norepinephrine_rate": "Cardiovascular",
        "epinephrine_rate": "Cardiovascular", "dopamine_rate": "Cardiovascular", "dobutamine_rate": "Cardiovascular",
        "pao2": "Respiratory", "fio2": "Respiratory", "respiratory_rate": "Respiratory", "oxygen_saturation": "Respiratory", "invasive_ventilation_on": "Respiratory",
        "creatinine": "Renal", "urine_output_volume": "Renal",
        "glasgow_coma_scale": "Neurologic",
        "platelet_count": "Hematologic/Liver", "bilirubin_total": "Hematologic/Liver",
        "temperature": "Other", "lactate": "Other",
    }
    for row in events:
        if row["event_time"] > cutoff_time:
            continue
        group = concept_group.get(row["canonical_concept"], "Other")
        groups[group].append(row)
    for group in groups.values():
        group.sort(key=lambda r: r["event_time"])
    return groups


def build_v2_dashboard_app(*, root: Path, api_client: V2DashboardAPI, raw_events_by_stay: Optional[Mapping[str, list]] = None) -> FastAPI:
    root = Path(root).resolve()
    demo_subjects = _demo_subjects(root)
    by_stay = {item["stay_id"]: item for item in demo_subjects}
    app = FastAPI(title="Personalized Patient Recovery Trajectory — V2 Dashboard")

    @app.get("/", response_class=HTMLResponse)
    def index(view: str = Query("replay"), stay_id: Optional[str] = Query(None), prediction_time: Optional[str] = Query(None)):
        if stay_id is None:
            stay_id = demo_subjects[0]["stay_id"]
        subject = by_stay.get(stay_id)
        if subject is None:
            return HTMLResponse(v2_render.render_shell(active_view=view, stay_id=None, prediction_time=None, demo_subjects=demo_subjects, body_html="", error="Unknown demo stay."), status_code=404)
        cutoffs = subject["legal_cutoffs"]
        if prediction_time is None or prediction_time not in cutoffs:
            prediction_time = cutoffs[0]
        cutoff_index = cutoffs.index(prediction_time)

        try:
            prediction = api_client.predict(stay_id, prediction_time)
        except Exception as error:  # noqa: BLE001
            body = f'<div class="section"><h2>Prediction unavailable</h2><p class="small">{type(error).__name__}</p></div>'
            return HTMLResponse(v2_render.render_shell(active_view=view, stay_id=stay_id, prediction_time=prediction_time, demo_subjects=demo_subjects, body_html=body, error="Could not obtain a prediction for this cutoff."))

        if view == "replay":
            body = _render_replay(root=root, api_client=api_client, subject=subject, stay_id=stay_id, prediction_time=prediction_time, cutoffs=cutoffs, cutoff_index=cutoff_index, prediction=prediction, raw_events_by_stay=raw_events_by_stay)
        elif view == "forecast":
            header = v2_render.render_patient_view_header(subject=subject, stay_id=stay_id, prediction_time=prediction_time, elapsed_hours=prediction["elapsed_icu_hours"], cutoffs=cutoffs, cutoff_index=cutoff_index)
            body = header + v2_render.render_forecast_details(prediction)
        elif view == "performance":
            body = _render_performance(root)
        elif view == "explainability":
            header = v2_render.render_patient_view_header(subject=subject, stay_id=stay_id, prediction_time=prediction_time, elapsed_hours=prediction["elapsed_icu_hours"], cutoffs=cutoffs, cutoff_index=cutoff_index)
            body = header + v2_render.render_explainability(prediction["explanations"])
        elif view == "dataquality":
            header = v2_render.render_patient_view_header(subject=subject, stay_id=stay_id, prediction_time=prediction_time, elapsed_hours=prediction["elapsed_icu_hours"], cutoffs=cutoffs, cutoff_index=cutoff_index)
            model_rows = [
                {"task": "recovery24", "family": "xgboost", "feature_variant": "B_MIN", "artifact_sha256": prediction["recovery"]["model_metadata"]["artifact_sha256_24h"]},
                {"task": "recovery48", "family": "xgboost", "feature_variant": "B_MIN", "artifact_sha256": prediction["recovery"]["model_metadata"]["artifact_sha256_48h"]},
                {"task": "icu_stay_time", "family": "xgboost", "feature_variant": "B_PLUS_F", "artifact_sha256": prediction["icu_stay_time"]["model_metadata"]["artifact_sha256"]},
                {"task": "organ_support", "family": "xgboost", "feature_variant": "B_FULL", "artifact_sha256": prediction["organ_support"]["model_metadata"]["artifact_sha256"]},
            ]
            body = header + v2_render.render_data_quality(data_quality=prediction["data_quality"], model_metadata_rows=model_rows)
        else:
            body = '<div class="section">Unknown view.</div>'

        return HTMLResponse(v2_render.render_shell(active_view=view, stay_id=stay_id, prediction_time=prediction_time, demo_subjects=demo_subjects, body_html=body))

    return app


def _render_replay(*, root, api_client, subject, stay_id, prediction_time, cutoffs, cutoff_index, prediction, raw_events_by_stay):
    header = v2_render.render_patient_view_header(subject=subject, stay_id=stay_id, prediction_time=prediction_time, elapsed_hours=prediction["elapsed_icu_hours"], cutoffs=cutoffs, cutoff_index=cutoff_index)
    status_cards = v2_render.render_status_cards(prediction)

    # Historical predictions up to and including the current cutoff, via
    # genuine sequential API calls -- not read from any cache.
    history_cutoffs = cutoffs[: cutoff_index + 1]
    history_predictions = [api_client.predict(stay_id, c) for c in history_cutoffs]
    history_sofa = [p["current_sofa"] for p in history_predictions]

    recovery_chart = v2_charts.recovery_trajectory_chart(
        history_times=history_cutoffs, history_sofa=history_sofa, cutoff_time=prediction_time,
        current_sofa=prediction["current_sofa"],
        forecast_24h_time="+24h", forecast_24h_sofa=prediction["recovery"]["sofa_hat_24h"],
        forecast_48h_time="+48h", forecast_48h_sofa=prediction["recovery"]["sofa_hat_48h"],
    )
    recovery_section = v2_render.render_recovery_chart_section(recovery_chart)

    icu_trend_uri = None
    support_trend_uri = None
    if len(history_predictions) >= 2:
        icu_trend_uri = v2_charts.trend_chart(x_labels=history_cutoffs, y_values=[p["icu_stay_time"]["remaining_hours"] for p in history_predictions], title="Remaining ICU-time trend", y_label="hours")
        support_trend_uri = v2_charts.trend_chart(x_labels=history_cutoffs, y_values=[p["organ_support"]["probability_24h"] for p in history_predictions], title="Support-risk trend", y_label="calibrated probability", threshold=prediction["organ_support"]["threshold"])
    icu_panel = v2_render.render_icu_panel(prediction, icu_trend_uri)
    support_panel = v2_render.render_support_panel(prediction, support_trend_uri)

    body = header + status_cards + recovery_section + f'<div class="grid-2">{icu_panel}{support_panel}</div>'

    window = prediction["temporal_window"]
    heatmap_uri = v2_charts.temporal_bin_heatmap(channel_names=window["channel_names"], observation_mask=window["observation_mask"], padding_mask=window["padding_mask"])
    body += v2_render.render_temporal_bins(heatmap_uri)
    body += _temporal_and_timeline(root=root, stay_id=stay_id, prediction_time=prediction_time, raw_events_by_stay=raw_events_by_stay, prediction=prediction)

    replay_history_section = _render_replay_history_table(history_cutoffs, history_predictions)
    body += replay_history_section
    return body


def _temporal_and_timeline(*, root, stay_id, prediction_time, raw_events_by_stay, prediction):
    if not raw_events_by_stay:
        return ""
    events = raw_events_by_stay.get(stay_id, [])
    groups = _group_events(events, prediction_time)
    timeline_html = v2_render.render_historical_timeline(group_rows=groups, cutoff_time=prediction_time)
    return timeline_html


def _render_replay_history_table(cutoffs, predictions) -> str:
    rows = "".join(
        f"""<tr><td>t{i}</td><td>{c}</td><td>{p['recovery']['delta_24h']:+.3f}</td><td>{p['recovery']['delta_48h']:+.3f}</td>
        <td>{p['icu_stay_time']['remaining_hours']:.1f}h</td><td>{p['organ_support']['probability_24h']*100:.1f}%</td>
        <td>{'ABOVE' if p['organ_support']['alert'] else 'below'}</td></tr>"""
        for i, (c, p) in enumerate(zip(cutoffs, predictions))
    )
    return f"""<div class="section"><h2>Replay history: how predictions evolve</h2>
    <p class="small">Accumulated through sequential API calls as replay advances — not sourced from a precomputed table.</p>
    <table><tr><th>Cutoff</th><th>Timestamp</th><th>Δ24</th><th>Δ48</th><th>ICU remaining</th><th>Support risk</th><th>Alert</th></tr>{rows}</table>
    </div>"""


def _load_demo_raw_events(root: Path) -> Mapping[str, list]:
    from data.synthetic.validation import load_jsonl

    demo = _load_json(root, "configs/performance_v2/v2_demo_manifest_v1.json")
    demo_stays = {item["stay_id"] for item in demo["demo_subjects"]}
    events = load_jsonl(root / "artifacts/data/synthetic/timelines/final/phase9_final_v1/canonical_timeline.jsonl")
    raw_events_by_stay: dict = {}
    for row in events:
        if row["stay_id"] in demo_stays:
            raw_events_by_stay.setdefault(row["stay_id"], []).append(row)
    return raw_events_by_stay


def app() -> FastAPI:
    """uvicorn --factory entrypoint: `uvicorn dashboard.v2_app:app --factory`.

    Calls the real V2 API over HTTP (V2_API_BASE_URL env var, default
    http://127.0.0.1:8010) -- start `api.v2_app:app` first."""

    root = Path(__file__).resolve().parents[1]
    base_url = os.environ.get("V2_API_BASE_URL", DEFAULT_V2_API_BASE_URL)
    api_client = HTTPV2DashboardAPIClient(base_url)
    return build_v2_dashboard_app(root=root, api_client=api_client, raw_events_by_stay=_load_demo_raw_events(root))


def _render_performance(root: Path) -> str:
    metrics = _load_json(root, "artifacts/performance_v2/phase4/metrics/final_metrics_v2.json")
    bootstrap = _load_json(root, "artifacts/performance_v2/phase4/bootstrap/final_bootstrap_v2.json")
    naive = _load_json(root, "artifacts/performance_v2/phase4/metrics/naive_comparison_v2.json")
    calibration = _load_json(root, "artifacts/performance_v2/phase4/metrics/calibration_evidence_v2.json")

    v1_test = {
        "recovery24": 1.121102021240419, "recovery48": 1.5423799902108615,
        "icu": 9.479719411307386, "support": 0.633735668141501,
    }
    v1_rows = [
        {"task": "Recovery24 MAE", "v1": f"{v1_test['recovery24']:.4f}", "v2": f"{metrics['recovery24']['mae']:.4f}"},
        {"task": "Recovery48 MAE", "v1": f"{v1_test['recovery48']:.4f}", "v2": f"{metrics['recovery48']['mae']:.4f}"},
        {"task": "ICU median AE (h)", "v1": f"{v1_test['icu']:.4f}", "v2": f"{metrics['icu_stay_time']['median_absolute_error']:.4f}"},
        {"task": "Support calibrated AUPRC", "v1": f"{v1_test['support']:.4f}", "v2": f"{metrics['organ_support_calibrated']['auprc']:.4f}"},
    ]

    chart_uris = {
        "recovery_naive": v2_charts.bar_comparison_chart(labels=["V2 model", "Naive"], values=[metrics["recovery24"]["mae"], naive["recovery24"]["naive_mae"]], title="Recovery+24 MAE vs naive", y_label="MAE"),
        "icu_naive": v2_charts.bar_comparison_chart(labels=["V2 model", "Naive"], values=[metrics["icu_stay_time"]["median_absolute_error"], naive["icu_stay_time"]["naive_median_ae"]], title="ICU median AE vs naive", y_label="hours"),
        "support_reliability": v2_charts.reliability_chart(table_raw=calibration["raw_reliability_table"], table_calibrated=calibration["calibrated_reliability_table"]),
    }

    return v2_render.render_model_performance(metrics=metrics, bootstrap=bootstrap, naive=naive, v1_comparison_rows=v1_rows, chart_uris=chart_uris)
