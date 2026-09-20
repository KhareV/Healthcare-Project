"""Small dependency-free HTML renderer for the replay dashboard."""

from html import escape
from typing import Optional, Sequence
from dashboard.view_models import (
    DashboardView,
    EXPLANATION_TITLE,
    ICU_DEFINITION,
    ICU_TITLE,
    MONITORED_SUPPORTS,
    REPLAY_BANNER,
    SUPPORT_DEFINITION,
    SUPPORT_TITLE,
    SYNTHETIC_BANNER,
)


def _e(value: object) -> str:
    return escape(str(value), quote=True)


def _options(values: Sequence[str], selected: Optional[str]) -> str:
    return "".join(
        '<option value="{}"{}>{}</option>'.format(
            _e(value), " selected" if value == selected else "", _e(value)
        )
        for value in values
    )


def _panel(title: str, body: str) -> str:
    return '<section class="panel"><h2>{}</h2>{}</section>'.format(_e(title), body)


def render_dashboard(
    *,
    stays: Sequence[str],
    selected_stay: Optional[str],
    cutoffs: Sequence[str],
    selected_cutoff: Optional[str],
    view: Optional[DashboardView],
    error: Optional[str],
    synthetic: bool,
) -> str:
    stay_form = (
        '<form method="get"><label for="stay_id">Demo-safe stay</label>'
        '<select id="stay_id" name="stay_id" onchange="this.form.submit()">'
        '<option value="">Select a stay</option>{}</select></form>'.format(
            _options(stays, selected_stay)
        )
    )
    cutoff_form = ""
    if selected_stay:
        cutoff_form = (
            '<form method="get"><input type="hidden" name="stay_id" value="{}">'
            '<label for="prediction_time">Approved prediction cutoff</label>'
            '<select id="prediction_time" name="prediction_time">'
            '<option value="">Select a cutoff</option>{}</select>'
            '<button type="submit">Replay cutoff</button></form>'
        ).format(_e(selected_stay), _options(cutoffs, selected_cutoff))
    panels = ""
    if view is not None:
        timeline_rows = "".join(
            "<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>".format(
                _e(event.prediction_available_time), _e(event.category), _e(event.label),
                _e(event.value), _e("observed" if event.observed else "not observed")
            )
            for event in view.timeline
        )
        panels += _panel(
            "Replay overview",
            "<dl><dt>Stay</dt><dd>{}</dd><dt>Prediction cutoff</dt><dd>{}</dd>"
            "<dt>ICU elapsed time</dt><dd>{:.1f} hours</dd></dl>".format(
                _e(view.stay_id), _e(view.prediction_time), view.icu_elapsed_hours
            ),
        )
        panels += _panel(
            "History available at cutoff",
            '<p>Only events available at or before the selected cutoff are shown.</p>'
            '<table><thead><tr><th>Available time</th><th>Category</th><th>Signal</th>'
            '<th>Value</th><th>Observation status</th></tr></thead><tbody>{}</tbody></table>'.format(
                timeline_rows
            ),
        )
        recovery = view.recovery
        panels += _panel(
            "Recovery trajectory",
            "<dl><dt>{}</dt><dd>{}</dd><dt>Predicted SOFA change +24h</dt><dd>{}</dd>"
            "<dt>{}</dt><dd>{}</dd><dt>Predicted SOFA change +48h</dt><dd>{}</dd>"
            "<dt>{}</dt><dd>{}</dd></dl>".format(
                _e(recovery.current_label), _e(recovery.current_sofa),
                _e(recovery.delta_24h), _e(recovery.predicted_24h_label),
                _e(recovery.predicted_sofa_24h), _e(recovery.delta_48h),
                _e(recovery.predicted_48h_label), _e(recovery.predicted_sofa_48h),
            ),
        )
        panels += _panel(
            ICU_TITLE,
            "<p>{}</p><p><strong>{} hours</strong></p>".format(
                _e(ICU_DEFINITION), _e(view.icu_stay_time_hours)
            ),
        )
        panels += _panel(
            SUPPORT_TITLE,
            "<p>{}</p><p>Calibrated probability: <strong>{}</strong></p>"
            "<p>Frozen validation threshold: {}</p><p>Monitored supports: {}</p>".format(
                _e(SUPPORT_DEFINITION), _e(view.support_probability),
                _e(view.support_threshold), _e("; ".join(MONITORED_SUPPORTS))
            ),
        )
        explanation_body = "<p>Attributions describe model behavior; they are not causal effects.</p>"
        for explanation in view.explanations:
            explanation_body += "<h3>{} — {} / {}</h3><ul>{}</ul>".format(
                _e(explanation.task), _e(explanation.family), _e(explanation.method),
                "".join(
                    "<li>{}: {}</li>".format(_e(name), _e(value))
                    for name, value in explanation.items
                ),
            )
        panels += _panel(EXPLANATION_TITLE, explanation_body)
        q = view.quality
        panels += _panel(
            "Data quality",
            "<dl><dt>Total history bins</dt><dd>{}</dd><dt>Bins with genuine observations</dt>"
            "<dd>{}</dd><dt>Pre-ICU padding bins</dt><dd>{}</dd>"
            "<dt>Total feature values</dt><dd>{}</dd><dt>Observed feature values</dt>"
            "<dd>{}</dd><dt>Missing feature values</dt><dd>{}</dd></dl>".format(
                q["total_bins"], q["observed_bins"], q["padding_bins"],
                q["total_feature_values"], q["observed_feature_values"],
                q["missing_feature_values"],
            ),
        )
        metadata_body = (
            "<p>Manifest: {}<br>Manifest SHA-256: {}<br>Split: {}</p>".format(
                _e(view.manifest_version), _e(view.manifest_sha256), _e(view.split_version)
            )
        )
        for item in view.task_metadata:
            metadata_body += (
                "<h3>{}</h3><dl><dt>Family</dt><dd>{}</dd><dt>Model version</dt><dd>{}</dd>"
                "<dt>Artifact SHA-256</dt><dd>{}</dd><dt>Preprocessor SHA-256</dt><dd>{}</dd>"
                "<dt>Feature version</dt><dd>{}</dd><dt>Label version</dt><dd>{}</dd>"
                "<dt>Split hash</dt><dd>{}</dd><dt>Explanation method</dt><dd>{}</dd></dl>"
            ).format(
                _e(item.task), _e(item.family), _e(item.model_version),
                _e(item.artifact_sha256), _e(item.preprocessor_sha256),
                _e(item.feature_version), _e(item.label_version), _e(item.split_hash),
                _e(item.explanation_method),
            )
        metadata_body += "<p>Support calibrator SHA-256: {}<br>Support threshold SHA-256: {}</p>".format(
            _e(view.calibrator_sha256), _e(view.threshold_sha256)
        )
        panels += _panel("Per-task model metadata", metadata_body)
    error_html = '<p role="alert" class="error">{}</p>'.format(_e(error)) if error else ""
    synthetic_html = '<p class="synthetic">{}</p>'.format(SYNTHETIC_BANNER) if synthetic else ""
    return """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Recovery trajectory replay</title>
<style>body{{font-family:system-ui,sans-serif;max-width:1100px;margin:auto;padding:1rem;background:#f4f6f8;color:#17212b}}
.banner,.synthetic{{font-weight:800;padding:1rem;border:3px solid #17212b;background:#fff}}.panel,form{{background:#fff;padding:1rem;margin:1rem 0;border:1px solid #b8c0c8}}
label{{font-weight:700;margin-right:.5rem}}select,button{{padding:.5rem;margin:.25rem}}table{{border-collapse:collapse;width:100%}}th,td{{text-align:left;border:1px solid #ccd2d8;padding:.4rem;vertical-align:top}}
dt{{font-weight:700}}dd{{margin-bottom:.4rem;overflow-wrap:anywhere}}.error{{padding:1rem;border:2px solid #7b1f1f;background:#fff}}</style></head>
<body><h1>Personalized patient recovery trajectory</h1><p class="banner">{}</p>{}{}{}{}{}
</body></html>""".format(
        REPLAY_BANNER, synthetic_html, error_html, stay_form, cutoff_form, panels
    )
