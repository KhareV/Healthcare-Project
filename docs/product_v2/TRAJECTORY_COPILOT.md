# Trajectory Copilot

## Status

Not part of the predictive model. Not part of the scientific evaluation.
Cannot change a model output. It is an interpretation layer that runs
*after* a prediction has already been computed, over the exact numbers the
dashboard already shows.

## What it is grounded in

Every Copilot response is built from a structured context packet
(`api/v2_app.py::_assistant_context`), never free text:

```
patient_alias, prediction_time, episode {elapsed_hours, cardiac_subtype/
  "custom record (direct entry)", age_years, sex}, current_state
  {current_sofa}, forecasts {delta/predicted SOFA 24h/48h, remaining ICU
  hours, support raw/calibrated probability, threshold, alert}, previous_
  cutoff {same fields, if a previous cutoff was supplied — used only to
  compute deterministic deltas, see below}, top_contributors {top 5 SHAP
  drivers per task}, data_quality, system_limitations (a fixed list:
  synthetic benchmark, retrospective replay, not clinically validated,
  point forecasts with no per-prediction interval, SHAP is non-causal)
```

For a custom record, `patient_alias`/`age_years`/`sex` come from
`serving.v2.custom_record.get_custom_record`, not the demo-manifest alias
table — so the Copilot correctly names a manually-entered patient instead
of falling back to a raw stay id.

## "What changed?" is computed, not guessed

Before calling the LLM, `serving/v2/ai_recommendation.py::_format_deltas`
computes exact before/after deltas for current SOFA, both recovery
forecasts, remaining ICU hours, and support probability, plus whether the
threshold-crossing state flipped — as plain arithmetic on the two already-
computed predictions, not something the model infers. The LLM only narrates
numbers it's handed.

## System prompt (versioned)

`configs/product_v2/trajectory_copilot_prompt_v1.txt` — hard rules: never
diagnose, never recommend a treatment/medication/ventilation/vasopressor
decision, never claim a feature "caused" an outcome (only "contributed
to"), never claim calibrated-probability decomposition from TreeSHAP (it
explains the raw margin, before calibration), never invent a fact absent
from the context, and a fixed refusal line for treatment/diagnosis
questions: *"This research prototype does not provide treatment
recommendations — it can only describe the forecasted numbers and their
statistical drivers."* Verified live in this session (see
`tests/test_v2_trajectory_copilot.py` and manual probes during
development).

## Fresh-test and ownership guards apply identically

`/assistant` is built from the exact same `_build_prediction_payload` call
`/predict` uses, so the fresh-test guard rejects a sealed-cohort stay
before any LLM call is ever made, and `_authorize_custom_stay` rejects an
unauthorized caller's request to a custom record the same way (404) —
see [`AUTH_AND_DATA_BOUNDARIES.md`](AUTH_AND_DATA_BOUNDARIES.md).

## Provider and failure mode

Groq-hosted `openai/gpt-oss-120b` (the strongest chat model this project's
Groq key has access to — verified via live model discovery, not assumed).
If the key is absent or the call fails for any reason, the endpoint returns
`{"status": "UNAVAILABLE", "error": "..."}` — the dashboard, predictions,
and TreeSHAP are entirely unaffected; the Copilot is always optional to the
core product.

## UI

A right-side drawer (`TrajectoryCopilot.svelte`), opened from a header
toggle or a per-patient "Ask Copilot" button, always grounded to whichever
patient/cutoff was last viewed (`stores/copilot.svelte.ts`). Auto-generates
a short default summary on open plus suggested prompts (e.g. "What changed
since the previous cutoff?"); free-form questions go through the same
context/guard path. Verified rendering live via Playwright
(`frontend/tests/e2e/04-explainability-and-copilot.spec.ts`).
