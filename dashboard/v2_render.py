"""Server-rendered HTML for the V2 dashboard. Plain HTML/CSS, no JS
framework, consistent with the existing dashboard/render.py philosophy.
Charts are pre-rendered matplotlib PNGs (dashboard/v2_charts.py) embedded as
data URIs.
"""

from __future__ import annotations

import html
from typing import Mapping, Optional, Sequence

REPLAY_BANNER = "RETROSPECTIVE SEQUENTIAL REPLAY"
SYNTHETIC_BANNER = "SYNTHETIC RESEARCH BENCHMARK"
NOT_REALTIME_BANNER = "NOT REAL-TIME CLINICAL PREDICTION"

ICU_TITLE = "Remaining ICU stay time"
ICU_DEFINITION = "Remaining time until current ICU stay ends"
SUPPORT_TITLE = "New Organ-Support Initiation Risk — Next 24 Hours"
SUPPORT_DEFINITION = "Calibrated probability of eligible OFF-to-ON initiation (qualifying vasopressor support or invasive mechanical ventilation) within the next 24 hours"
MONITORED_SUPPORTS = ("Qualifying vasopressor support", "Invasive mechanical ventilation")

_CSS = """
:root {
  --bg: #f4f6f8; --card: #ffffff; --ink: #17324d; --muted: #5b6b7c;
  --border: #dfe6ee; --accent: #0f766e; --accent-2: #2563eb; --danger: #b91c1c;
  --radius: 10px;
}
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--ink); font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif; }
.banner { background:#17324d; color:#fff; padding:10px 18px; font-size:13px; letter-spacing:.02em; display:flex; gap:18px; flex-wrap:wrap; align-items:center; }
.banner b { color:#fca5a5; }
.layout { display:flex; min-height:100vh; }
.sidebar { width:250px; flex-shrink:0; background:#0f2436; color:#e6edf3; padding:18px 14px; }
.sidebar h1 { font-size:16px; margin:0 0 4px; }
.sidebar .sub { font-size:11px; color:#9db3c4; margin-bottom:18px; }
.sidebar nav a { display:block; padding:8px 10px; border-radius:8px; color:#cfe0ec; text-decoration:none; font-size:13px; margin-bottom:4px; }
.sidebar nav a.active { background:#153650; color:#fff; font-weight:600; }
.sidebar .field { margin-top:16px; }
.sidebar label { font-size:11px; color:#9db3c4; display:block; margin-bottom:4px; }
.sidebar select { width:100%; padding:6px; border-radius:6px; border:1px solid #294b63; background:#0f2436; color:#fff; font-size:12px; }
.sidebar .status { margin-top:18px; font-size:10px; color:#7f97a8; line-height:1.5; }
.main { flex:1; padding:20px 26px; max-width:1180px; }
.cards { display:grid; grid-template-columns:repeat(auto-fit,minmax(200px,1fr)); gap:14px; margin-bottom:20px; }
.card { background:var(--card); border:1px solid var(--border); border-radius:var(--radius); padding:14px 16px; }
.card h3 { margin:0 0 6px; font-size:11px; text-transform:uppercase; letter-spacing:.04em; color:var(--muted); }
.card .value { font-size:24px; font-weight:700; }
.card .unit { font-size:12px; color:var(--muted); font-weight:400; }
.card .sub { font-size:11px; color:var(--muted); margin-top:4px; }
.section { background:var(--card); border:1px solid var(--border); border-radius:var(--radius); padding:16px 18px; margin-bottom:18px; }
.section h2 { font-size:15px; margin:0 0 10px; }
.section h3 { font-size:12px; margin: 14px 0 6px; color: var(--muted); text-transform:uppercase; letter-spacing:.03em;}
.pill { display:inline-block; padding:2px 9px; border-radius:99px; font-size:11px; font-weight:600; }
.pill.below { background:#dcfce7; color:#166534; }
.pill.above { background:#fee2e2; color:#991b1b; }
table { border-collapse:collapse; width:100%; font-size:12px; }
table th, table td { text-align:left; padding:5px 8px; border-bottom:1px solid var(--border); }
table th { color:var(--muted); font-weight:600; font-size:11px; text-transform:uppercase; }
.grid-2 { display:grid; grid-template-columns:1fr 1fr; gap:16px; }
.grid-3 { display:grid; grid-template-columns:repeat(3,1fr); gap:16px; }
img.chart { width:100%; height:auto; border-radius:8px; }
.hash { font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:11px; color:var(--muted); }
details summary { cursor:pointer; font-size:12px; color:var(--accent-2); }
.replay-controls { display:flex; gap:8px; align-items:center; margin-bottom:14px; flex-wrap:wrap; }
.replay-controls a.btn, .replay-controls select { padding:6px 12px; border-radius:8px; border:1px solid var(--border); background:#fff; font-size:12px; text-decoration:none; color:var(--ink); }
.replay-controls a.btn.disabled { color:#c1c9d2; pointer-events:none; }
.error { background:#fef2f2; border:1px solid #fecaca; color:#991b1b; padding:12px 16px; border-radius:8px; margin-bottom:16px; font-size:13px; }
.attr-pos { color:#166534; }
.attr-neg { color:#991b1b; }
.small { font-size:11px; color:var(--muted); }
.footer-note { font-size:11px; color:var(--muted); margin-top:24px; }
"""

_VIEWS = (
    ("replay", "Patient Replay"),
    ("forecast", "Forecast Details"),
    ("performance", "Model Performance"),
    ("explainability", "Explainability"),
    ("dataquality", "Data Quality & Provenance"),
)


def _e(value: object) -> str:
    return html.escape(str(value))


def _qs(**params) -> str:
    from urllib.parse import urlencode

    return urlencode({k: v for k, v in params.items() if v is not None})


def render_shell(*, active_view: str, stay_id: Optional[str], prediction_time: Optional[str], demo_subjects: Sequence[Mapping[str, object]], body_html: str, error: Optional[str] = None) -> str:
    nav_links = "".join(
        f'<a class="{"active" if key == active_view else ""}" href="/?{_qs(view=key, stay_id=stay_id, prediction_time=prediction_time)}">{label}</a>'
        for key, label in _VIEWS
    )
    options = "".join(
        f'<option value="{_e(item["stay_id"])}" {"selected" if item["stay_id"] == stay_id else ""}>{_e(item["subject_id"])} — {_e(item["cardiac_condition_group"].replace("SYNTHETIC_", "").title())}</option>'
        for item in demo_subjects
    )
    error_html = f'<div class="error">{_e(error)}</div>' if error else ""
    return f"""<!doctype html><html><head><meta charset="utf-8"/><meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Personalized Patient Recovery Trajectory — V2</title><style>{_CSS}</style></head>
<body>
<div class="banner"><b>{REPLAY_BANNER}</b> · {SYNTHETIC_BANNER} · {NOT_REALTIME_BANNER}</div>
<div class="layout">
  <div class="sidebar">
    <h1>Personalized Patient Recovery Trajectory</h1>
    <div class="sub">Time-series forecasting · V2 XGBoost models</div>
    <nav>{nav_links}</nav>
    <form method="get" class="field">
      <input type="hidden" name="view" value="{_e(active_view)}"/>
      <label>Demo patient / stay</label>
      <select name="stay_id" onchange="this.form.submit()">{options}</select>
    </form>
    <div class="status">
      RETROSPECTIVE SEQUENTIAL REPLAY over a fixed 48h lookback (eight 6h bins).<br/>
      Synthetic cardiac benchmark. Not clinically validated. Not deployed.
    </div>
  </div>
  <div class="main">
    {error_html}
    {body_html}
    <div class="footer-note">V2_DASHBOARD_INTEGRATION · models frozen in artifacts/performance_v2/phase3/selected_models_v2.json</div>
  </div>
</div>
</body></html>"""


def status_card(title: str, value: str, unit: str = "", sub: str = "") -> str:
    return f'<div class="card"><h3>{_e(title)}</h3><div class="value">{_e(value)} <span class="unit">{_e(unit)}</span></div><div class="sub">{_e(sub)}</div></div>'


def render_patient_view_header(*, subject: Mapping[str, object], stay_id: str, prediction_time: str, elapsed_hours: int, cutoffs: Sequence[str], cutoff_index: int) -> str:
    rows = "".join(
        f"<tr><td>{k}</td><td>{_e(v)}</td></tr>"
        for k, v in (
            ("Demo subject", subject["subject_id"]),
            ("Demo stay", stay_id),
            ("Age (years)", subject["age_years"]),
            ("Sex", subject["sex_category"]),
            ("Cardiac subtype", subject["cardiac_condition_group"]),
            ("ICU admission (intime)", subject["intime"]),
            ("Selected replay timestamp", prediction_time),
            ("Elapsed ICU hours at cutoff", elapsed_hours),
        )
    )
    prev_cutoff = cutoffs[cutoff_index - 1] if cutoff_index > 0 else None
    next_cutoff = cutoffs[cutoff_index + 1] if cutoff_index < len(cutoffs) - 1 else None
    first_cutoff = cutoffs[0]
    options = "".join(f'<option value="{_e(c)}" {"selected" if c == prediction_time else ""}>{_e(c)} (t{i})</option>' for i, c in enumerate(cutoffs))
    controls = f"""
    <div class="replay-controls">
      <a class="btn {"disabled" if prev_cutoff is None else ""}" href="/?{_qs(view="replay", stay_id=stay_id, prediction_time=prev_cutoff)}">&larr; Previous cutoff</a>
      <form method="get" style="display:inline">
        <input type="hidden" name="view" value="replay"/><input type="hidden" name="stay_id" value="{_e(stay_id)}"/>
        <select name="prediction_time" onchange="this.form.submit()">{options}</select>
      </form>
      <a class="btn {"disabled" if next_cutoff is None else ""}" href="/?{_qs(view="replay", stay_id=stay_id, prediction_time=next_cutoff)}">Next cutoff &rarr;</a>
      <a class="btn" href="/?{_qs(view="replay", stay_id=stay_id, prediction_time=first_cutoff)}">Reset to first cutoff</a>
      <span class="small">Cutoff t{cutoff_index} of {len(cutoffs) - 1}</span>
    </div>"""
    return f'<div class="section"><h2>Patient / Stay Overview</h2><table>{rows}</table></div>{controls}'


def render_status_cards(prediction: Mapping[str, object]) -> str:
    cards = [
        status_card("Current SOFA", f"{prediction['current_sofa']:.1f}", "points", "Observed at selected cutoff"),
        status_card("Predicted SOFA +24H", f"{prediction['recovery']['sofa_hat_24h']:.1f}", "points", f"ΔSOFA24 = {prediction['recovery']['delta_24h']:+.2f}"),
        status_card("Predicted SOFA +48H", f"{prediction['recovery']['sofa_hat_48h']:.1f}", "points", f"ΔSOFA48 = {prediction['recovery']['delta_48h']:+.2f}"),
        status_card(ICU_TITLE, f"{prediction['icu_stay_time']['remaining_hours']:.1f}", "hours", f"{prediction['icu_stay_time']['remaining_hours'] / 24:.1f} days"),
        status_card("New Organ-Support Risk (24h)", f"{prediction['organ_support']['probability_24h']*100:.1f}", "%", f"threshold {prediction['organ_support']['threshold']:.3f} — {'ABOVE' if prediction['organ_support']['alert'] else 'BELOW'}"),
    ]
    return '<div class="cards">' + "".join(cards) + "</div>"


def render_recovery_chart_section(chart_data_uri: str) -> str:
    return f'<div class="section"><h2>Recovery Trajectory</h2><img class="chart" src="{chart_data_uri}"/><div class="small">Blue = observed SOFA up to the cutoff. Red dashed = independent +24h/+48h forecast (never chained). Shaded region = forecast horizon.</div></div>'


def render_icu_panel(prediction: Mapping[str, object], trend_chart_uri: Optional[str]) -> str:
    hours = prediction["icu_stay_time"]["remaining_hours"]
    days = int(hours // 24)
    rem = hours - days * 24
    trend = f'<img class="chart" src="{trend_chart_uri}"/>' if trend_chart_uri else '<div class="small">Trend appears after at least 2 replay cutoffs.</div>'
    return f"""<div class="section"><h2>{ICU_TITLE}</h2>
    <p class="small">{ICU_DEFINITION}.</p>
    <div class="value">{hours:.2f} <span class="unit">hours</span></div>
    <div class="small">≈ {days}d {rem:.1f}h</div>
    <h3>Prediction trend across recent replay cutoffs</h3>{trend}
    </div>"""


def render_support_panel(prediction: Mapping[str, object], trend_chart_uri: Optional[str]) -> str:
    support = prediction["organ_support"]
    pill = "above" if support["alert"] else "below"
    label = "ABOVE THRESHOLD" if support["alert"] else "BELOW THRESHOLD"
    monitored = "".join(f"<li>{_e(item)}</li>" for item in MONITORED_SUPPORTS)
    trend = f'<img class="chart" src="{trend_chart_uri}"/>' if trend_chart_uri else '<div class="small">Trend appears after at least 2 replay cutoffs.</div>'
    return f"""<div class="section"><h2>{SUPPORT_TITLE}</h2>
    <p class="small">{SUPPORT_DEFINITION}. Continuation of already-active support is never counted as a new initiation.</p>
    <div class="value">{support['probability_24h']*100:.1f}<span class="unit">%</span> <span class="pill {pill}">{label}</span></div>
    <div class="small">Frozen threshold: {support['threshold']:.6f} · Raw model probability: {support['raw_probability']*100:.1f}%</div>
    <h3>Monitored support types</h3><ul class="small">{monitored}</ul>
    <h3>Calibrated risk across recent replay cutoffs</h3>{trend}
    </div>"""


def render_historical_timeline(*, group_rows: Mapping[str, Sequence[Mapping[str, object]]], cutoff_time: str) -> str:
    sections = []
    for group_name, rows in group_rows.items():
        if not rows:
            continue
        table_rows = "".join(
            f"<tr><td>{_e(r['event_time'])}</td><td>{_e(r['canonical_concept'])}</td><td>{_e(r['value_numeric'])} {_e(r.get('unit',''))}</td></tr>"
            for r in rows[-8:]
        )
        sections.append(f'<h3>{_e(group_name)}</h3><table><tr><th>Event time (&le; cutoff)</th><th>Concept</th><th>Value</th></tr>{table_rows}</table>')
    body = "".join(sections) if sections else '<p class="small">No observed events at or before this cutoff.</p>'
    return f'<div class="section"><h2>Historical Timeline (event_time &le; {_e(cutoff_time)})</h2>{body}</div>'


def render_temporal_bins(heatmap_uri: str) -> str:
    return f'<div class="section"><h2>Temporal Lookback Window</h2><p class="small">The frozen V2 models consume exactly eight 6-hour bins covering (t-48h, t].</p><img class="chart" src="{heatmap_uri}"/><div class="small">Green = observed. Red = not observed (within episode). Gray = padding (before ICU admission).</div></div>'


# --------------------------------------------------------------------------
# View 2: Forecast Details
# --------------------------------------------------------------------------

def render_forecast_details(prediction: Mapping[str, object]) -> str:
    recovery = prediction["recovery"]
    icu = prediction["icu_stay_time"]
    support = prediction["organ_support"]
    return f"""
    <div class="section"><h2>Recovery — Detail</h2>
      <table>
        <tr><th>Field</th><th>Value</th></tr>
        <tr><td>Current SOFA</td><td>{prediction['current_sofa']:.2f}</td></tr>
        <tr><td>ΔSOFA24 (raw, independent)</td><td>{recovery['delta_24h']:+.4f}</td></tr>
        <tr><td>ΔSOFA48 (raw, independent)</td><td>{recovery['delta_48h']:+.4f}</td></tr>
        <tr><td>Reconstructed SOFA +24h — clip(current+Δ24, 0, 24)</td><td>{recovery['sofa_hat_24h']:.2f}</td></tr>
        <tr><td>Reconstructed SOFA +48h — clip(current+Δ48, 0, 24)</td><td>{recovery['sofa_hat_48h']:.2f}</td></tr>
        <tr><td>Feature variant</td><td>B_MIN (both horizons)</td></tr>
        <tr><td>Model identity (24h)</td><td class="hash">{recovery['model_metadata']['artifact_sha256_24h']}</td></tr>
        <tr><td>Model identity (48h)</td><td class="hash">{recovery['model_metadata']['artifact_sha256_48h']}</td></tr>
      </table>
    </div>
    <div class="section"><h2>{ICU_TITLE} — Detail</h2>
      <table>
        <tr><th>Field</th><th>Value</th></tr>
        <tr><td>Predicted remaining hours</td><td>{icu['remaining_hours']:.2f}</td></tr>
        <tr><td>Human-readable</td><td>{int(icu['remaining_hours']//24)}d {icu['remaining_hours']%24:.1f}h</td></tr>
        <tr><td>Raw log1p prediction</td><td>{icu['raw_log_prediction']:.4f}</td></tr>
        <tr><td>Postprocess</td><td>expm1(clamp_min(raw, 0))</td></tr>
        <tr><td>Feature variant</td><td>B_PLUS_F</td></tr>
        <tr><td>Model identity</td><td class="hash">{icu['model_metadata']['artifact_sha256']}</td></tr>
      </table>
    </div>
    <div class="section"><h2>Organ Support — Detail</h2>
      <table>
        <tr><th>Field</th><th>Value</th></tr>
        <tr><td>Raw XGBoost probability (pre-calibration)</td><td>{support['raw_probability']:.4f}</td></tr>
        <tr><td>Calibrated serving probability</td><td>{support['probability_24h']:.4f}</td></tr>
        <tr><td>Frozen threshold</td><td>{support['threshold']:.6f}</td></tr>
        <tr><td>Alert state</td><td>{'ABOVE THRESHOLD' if support['alert'] else 'BELOW THRESHOLD'}</td></tr>
        <tr><td>Feature variant</td><td>B_FULL</td></tr>
        <tr><td>Model identity</td><td class="hash">{support['model_metadata']['artifact_sha256']}</td></tr>
      </table>
      <p class="small">Raw and calibrated probabilities are shown separately at all times: isotonic calibration is a non-linear, non-additive transform of the raw model score, never presented as identical to it.</p>
    </div>
    """


# --------------------------------------------------------------------------
# View 3: Model Performance (reads frozen Phase-4 artifacts only)
# --------------------------------------------------------------------------

def _metric_card(title: str, value: str, ci: Optional[Sequence[float]] = None, naive: Optional[str] = None, improvement: Optional[str] = None) -> str:
    ci_html = f'<div class="sub">95% CI [{ci[0]:.4f}, {ci[1]:.4f}]</div>' if ci else ""
    naive_html = f'<div class="sub">naive: {naive}{" · " + improvement if improvement else ""}</div>' if naive else ""
    return f'<div class="card"><h3>{_e(title)}</h3><div class="value">{_e(value)}</div>{ci_html}{naive_html}</div>'


def render_model_performance(*, metrics: Mapping[str, object], bootstrap: Mapping[str, object], naive: Mapping[str, object], v1_comparison_rows: Sequence[Mapping[str, object]], chart_uris: Mapping[str, str]) -> str:
    r24 = metrics["recovery24"]
    r48 = metrics["recovery48"]
    icu = metrics["icu_stay_time"]
    sr = metrics["organ_support_raw"]
    sc = metrics["organ_support_calibrated"]

    cards = "".join([
        _metric_card("Recovery +24 MAE", f"{r24['mae']:.4f}", (bootstrap['recovery24']['mae']['ci_lower'], bootstrap['recovery24']['mae']['ci_upper']), f"{naive['recovery24']['naive_mae']:.4f}", f"+{naive['recovery24']['relative_improvement']*100:.1f}%"),
        _metric_card("Recovery +48 MAE", f"{r48['mae']:.4f}", (bootstrap['recovery48']['mae']['ci_lower'], bootstrap['recovery48']['mae']['ci_upper']), f"{naive['recovery48']['naive_mae']:.4f}", f"+{naive['recovery48']['relative_improvement']*100:.1f}%"),
        _metric_card("ICU median AE", f"{icu['median_absolute_error']:.4f} h", (bootstrap['icu_stay_time']['median_absolute_error']['ci_lower'], bootstrap['icu_stay_time']['median_absolute_error']['ci_upper']), f"{naive['icu_stay_time']['naive_median_ae']:.2f} h", f"+{naive['icu_stay_time']['relative_improvement']*100:.1f}%"),
        _metric_card("Support calibrated AUPRC", f"{sc['auprc']:.4f}", (bootstrap['organ_support_calibrated']['auprc']['ci_lower'], bootstrap['organ_support_calibrated']['auprc']['ci_upper']), f"{naive['organ_support']['naive_auprc']:.4f}", f"+{naive['organ_support']['relative_improvement']*100:.1f}%"),
    ])

    def _row(label, value):
        return f"<tr><td>{_e(label)}</td><td>{_e(value)}</td></tr>"

    support_table = "".join([
        _row("Raw AUPRC", f"{sr['auprc']:.4f}"), _row("Raw AUROC", f"{sr['auroc']:.4f}"), _row("Raw Brier", f"{sr['brier']:.4f}"),
        _row("Calibrated AUPRC", f"{sc['auprc']:.4f}"), _row("Calibrated AUROC", f"{sc['auroc']:.4f}"), _row("Calibrated Brier", f"{sc['brier']:.4f}"),
        _row("Threshold F1", f"{sc['f1']:.4f}"), _row("Threshold precision", f"{sc['precision']:.4f}"),
        _row("Threshold recall/sensitivity", f"{sc['sensitivity']:.4f}"), _row("Threshold specificity", f"{sc['specificity']:.4f}"),
    ])

    v1_rows = "".join(
        f"<tr><td>{_e(r['task'])}</td><td>{_e(r['v1'])}</td><td>{_e(r['v2'])}</td></tr>"
        for r in v1_comparison_rows
    )

    charts_html = "".join(f'<div><img class="chart" src="{uri}"/></div>' for uri in chart_uris.values())

    return f"""
    <div class="section"><h2>Final V2 Fresh-Test Evaluation (frozen, one-time)</h2>
      <p class="small">Rendered exclusively from artifacts/performance_v2/phase4/metrics/final_metrics_v2.json, final_bootstrap_v2.json, naive_comparison_v2.json — no inference is performed on this page.</p>
      <div class="cards">{cards}</div>
    </div>
    <div class="section"><h2>Organ support — full metric table</h2><table>{support_table}</table></div>
    <div class="section"><h2>Performance charts</h2><div class="grid-3">{charts_html}</div></div>
    <div class="section"><h2>Benchmark v1 → v2 (historical, different independent cohorts)</h2>
      <table><tr><th>Task</th><th>v1 final-test (historical)</th><th>v2 fresh-test</th></tr>{v1_rows}</table>
      <p class="small">v1 and v2 used different independent final-test cohorts (different generator seed and subject namespace) — this is a historical comparison, not paired statistical testing.</p>
    </div>
    """


# --------------------------------------------------------------------------
# View 4: Explainability
# --------------------------------------------------------------------------

def _attribution_list(items: Sequence[Mapping[str, object]], css_class: str) -> str:
    return "".join(f'<li class="{css_class}">{_e(item["label"])}: {item["attribution"]:+.4f}</li>' for item in items)


def render_explainability(explanations: Mapping[str, Mapping[str, object]]) -> str:
    task_titles = {"recovery24": "Recovery +24h", "recovery48": "Recovery +48h", "icu_stay_time": ICU_TITLE, "organ_support": "Organ Support (raw margin)"}
    blocks = []
    for task, title in task_titles.items():
        payload = explanations[task]
        diag = payload["diagnostics"]
        blocks.append(f"""
        <div class="section"><h2>{_e(title)} — TreeSHAP</h2>
        <p class="small">{_e(diag['method'])}. Additivity check: {'PASSED' if diag['additivity_check_passed'] else 'FAILED'}.</p>
        <div class="grid-2">
          <div><h3>Top positive contributors</h3><ul class="small">{_attribution_list(payload['top_positive_contributors'], 'attr-pos')}</ul></div>
          <div><h3>Top negative contributors</h3><ul class="small">{_attribution_list(payload['top_negative_contributors'], 'attr-neg')}</ul></div>
        </div>
        <p class="small">Attributions <b>contributed to</b> this prediction; they do not establish causation. {"This explains the raw XGBoost margin, before isotonic calibration — calibration is not SHAP-additive." if task == "organ_support" else ""}</p>
        </div>""")
    return "".join(blocks)


# --------------------------------------------------------------------------
# View 5: Data Quality & Provenance
# --------------------------------------------------------------------------

def render_data_quality(*, data_quality: Mapping[str, object], model_metadata_rows: Sequence[Mapping[str, object]]) -> str:
    dq_rows = "".join(f"<tr><td>{_e(k.replace('_',' ').title())}</td><td>{_e(v)}</td></tr>" for k, v in data_quality.items())
    meta_rows = "".join(
        f"""<tr><td>{_e(r['task'])}</td><td>{_e(r['family'])}</td><td>{_e(r['feature_variant'])}</td>
        <td><details><summary>{_e(r['artifact_sha256'][:12])}…</summary><span class="hash">{_e(r['artifact_sha256'])}</span></details></td></tr>"""
        for r in model_metadata_rows
    )
    return f"""
    <div class="section"><h2>Data Quality — observed-bin counts</h2>
      <table>{dq_rows}</table>
      <p class="small">These are transparent observation counts, never a prediction confidence, certainty, or quality score. No such quantity is computed by this application.</p>
    </div>
    <div class="section"><h2>Model / Provenance Metadata</h2>
      <table><tr><th>Task</th><th>Family</th><th>Feature variant</th><th>Model hash</th></tr>{meta_rows}</table>
      <p class="small">selected_models_v2.json: <span class="hash">6fed05f3c71ce648658b2f864e69fa9f80f3ec8538f53a4e370fa8a4eaf176a8</span><br/>
      v2_model_freeze_v1.json: <span class="hash">f5feb29cab2e25d6e8bff860af52ac505f40951578aa73547ec420ba8fcda570</span></p>
    </div>
    """
