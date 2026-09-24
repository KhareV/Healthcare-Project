# Performance V2 — Demo Script (3–5 minutes)

Prerequisites: the V2 API and the SvelteKit dashboard running per [RUNBOOK.md](../../RUNBOOK.md#v2-final-release-demo-application):

```bash
PYTHONPATH=src:. python3 -m uvicorn api.v2_app:app --factory --host 127.0.0.1 --port 8010
cd frontend && npm install && npm run dev
```

Open `http://localhost:5173/patients` (or start from `http://localhost:5173/` for the project landing page, then "Open Dashboard"). The dependency-free Python dashboard at `http://127.0.0.1:8511/` (`dashboard/v2_app.py`) covers the same five views and can be substituted throughout this script if Node is unavailable.

## 1. Introduce the disclaimer (10s)

Point at the top banner: **RETROSPECTIVE SEQUENTIAL REPLAY · SYNTHETIC RESEARCH BENCHMARK · NOT REAL-TIME CLINICAL PREDICTION**. State plainly: this replays historical synthetic cutoffs; it does not predict in real time and is not a clinical tool.

## 2. Select a demo patient (15s)

On the sidebar, pick a demo stay from the dropdown (3+ patients, spanning distinct synthetic cardiac subtypes — arrhythmia, heart failure, ischemic). Note the patient overview card: demo-safe identifiers only, age, sex, cardiac subtype, ICU admission time.

## 3. Explain the 48h history / 6h bins (20s)

Scroll to **Temporal Lookback Window**. Explain: every prediction consumes exactly eight 6-hour bins covering the 48 hours before the selected cutoff — the heatmap shows which bins/channels were actually observed (green) vs. not observed (red) vs. before-ICU-admission padding (gray).

## 4. Show current SOFA (10s)

Point at the "Current SOFA" status card — computed at the selected cutoff from the same frozen SOFA scoring used throughout the project.

## 5. Show the +24/+48 recovery forecast (30s)

Point at "Predicted SOFA +24H" and "Predicted SOFA +48H" cards, then the Recovery Trajectory chart: blue = observed SOFA up to the cutoff, red dashed = the two independent forecasts (never chained — +48h is not "+24h plus another delta"). Mention the raw ΔSOFA24/ΔSOFA48 values are also available in Forecast Details.

## 6. Show the remaining ICU-stay forecast (15s)

Point at the "Remaining ICU stay time" card and panel — exact hours, plus a day/hour breakdown, plus its trend across replay so far.

## 7. Show organ-support risk (25s)

Point at the "New Organ-Support Initiation Risk" panel: calibrated 24h probability, the frozen threshold, and the BELOW/ABOVE THRESHOLD pill. Note the monitored support types (qualifying vasopressor, invasive mechanical ventilation) and that continuation of already-active support is never counted as a new initiation.

## 8. Show TreeSHAP factors and the AI research note (45s)

Switch to **AI + SHAP**. For one task (e.g. recovery24), point at top positive/negative contributors with readable labels (e.g. "Current SOFA", "heart rate — most recent 6h"), and the additivity-check confirmation. Note the non-causal wording ("contributed to", never "caused"). Then click "Generate summary": a Groq-hosted language model synthesizes the same numbers and SHAP drivers into a short research note, generated strictly after prediction and never feeding back into it. Point out the model tag and the explicit "not clinical advice" disclaimer under the note.

## 9. Advance through 3–5 cutoffs (40s)

Back on **Patient Replay**, click "Next cutoff" repeatedly (or use the dropdown). Call out that each click is a fresh `POST /predict` call recomputed from history truncated at the new cutoff — never a lookup table. Watch the "Replay history" table grow underneath.

## 10. Show how predictions evolve (15s)

Point at the Replay history table and the ICU/support trend charts: this is exactly how the forecasts change (or don't) as more history becomes available — a valid model may legitimately repeat an estimate; a changed number is not required for correctness.

## 11. Open Model Performance (30s)

Switch to **Model Performance**. Point at the four metric cards (Recovery+24 MAE, Recovery+48 MAE, ICU median AE, Support calibrated AUPRC), each with its 95% grouped-bootstrap CI and naive-baseline comparison. Mention this page reads only the already-frozen, one-time fresh-test evaluation artifacts — no inference happens here. Scroll to the Benchmark v1 → v2 panel and note it is a historical comparison across two different independent cohorts, not paired statistical testing.

## 12. Show the fresh-test metrics and CIs explicitly (15s)

Re-emphasize the specific numbers: e.g. Recovery+24 MAE 1.0744 [1.0303, 1.1199], Support calibrated AUPRC 0.7042 [0.6612, 0.7437] — these are the final, frozen, one-time fresh-test results; they cannot be recomputed or re-run from this application.

## 13. Close with limitations (15s)

State clearly: synthetic benchmark, no clinical validation, no external real-patient validation, no deployment-efficacy claim. This is a retrospective research prototype demonstrating the full time-series forecasting pipeline end to end.

---

Total: ~4 minutes at a comfortable pace.
