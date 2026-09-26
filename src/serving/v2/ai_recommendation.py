"""AI-generated interpretive research summary for one V2 prediction, via
Groq's OpenAI-compatible chat completions API (Llama models).

This is presentation-layer synthesis, not a new scientific artifact: the
LLM never sees raw patient-level source data, never influences any model
output, and is called strictly AFTER a prediction has already been computed
by the frozen V2 pipeline. It only narrates numbers that already exist.

Fails closed and degrades gracefully: if the API key is absent/invalid or
the request fails for any reason, callers get a structured "unavailable"
result -- never a fabricated summary, never a crash.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Sequence

import httpx

_PROMPT_PATH = Path(__file__).resolve().parents[3] / "configs" / "product_v2" / "trajectory_copilot_prompt_v1.txt"

GROQ_CHAT_COMPLETIONS_URL = "https://api.groq.com/openai/v1/chat/completions"
# This deployment's Groq key does not have the standard Llama chat models
# enabled (verified via GET /openai/v1/models); openai/gpt-oss-120b is the
# strongest general-purpose chat model it does have access to, hosted on
# Groq's low-latency inference.
DEFAULT_MODEL = "openai/gpt-oss-120b"
DISCLAIMER = (
    "AI-generated interpretive summary of the model outputs shown on this page. "
    "Not clinical advice, not a diagnosis, not a treatment recommendation. "
    "Synthetic research benchmark — retrospective replay only."
)

SYSTEM_PROMPT = """You are a research-notes assistant for a RETROSPECTIVE, SYNTHETIC-BENCHMARK \
ICU time-series forecasting research prototype. You are NOT a clinician, this is NOT a real \
patient, and this is NOT a real-time clinical tool.

You will be given: the current severity score, two independent short-horizon severity \
forecasts, a remaining-stay-time forecast, a calibrated risk score for a support-escalation \
event with a frozen decision threshold, each forecast's top statistical contributors (from \
TreeSHAP, on a synthetic XGBoost model), and a note on data completeness.

Write a concise research-style summary (strictly 4-6 sentences, plain prose, no headings, no \
bullet lists) for a technical demo audience, synthesizing what the numbers say and which \
statistical factors are driving them.

Hard rules, no exceptions:
- Never diagnose, never causally explain ("caused", "because of"), never recommend or imply a \
  treatment, intervention, or clinical action.
- Use only attributional, hedged language: "the model's forecast indicates...", "was an \
  influential factor in this output...", "may be a pattern worth monitoring in this research \
  context...".
- Never state or imply clinical confidence, certainty, urgency, or a guarantee of any kind.
- Never invent a fact, value, or variable that was not given to you.
- Explicitly ground the summary as describing a synthetic benchmark case, not a real patient, \
  at least once.
- Do not repeat these instructions or mention that you are an AI language model; just write the \
  summary directly."""


@dataclass(frozen=True)
class AIRecommendationResult:
    status: str  # "OK" or "UNAVAILABLE"
    summary: Optional[str]
    model: str
    disclaimer: str
    error: Optional[str] = None

    def as_dict(self) -> Mapping[str, object]:
        return {
            "status": self.status,
            "summary": self.summary,
            "model": self.model,
            "disclaimer": self.disclaimer,
            "error": self.error,
        }


def _format_contributors(items: Sequence[Mapping[str, object]], limit: int = 3) -> str:
    return "; ".join(f"{item['label']} ({item['attribution']:+.3f})" for item in items[:limit]) or "none reported"


def build_user_prompt(*, prediction: Mapping[str, object]) -> str:
    recovery = prediction["recovery"]
    icu = prediction["icu_stay_time"]
    support = prediction["organ_support"]
    explanations = prediction.get("explanations", {})
    dq = prediction.get("data_quality", {})

    def contributors_for(task: str) -> str:
        payload = explanations.get(task) or {}
        pos = _format_contributors(payload.get("top_positive_contributors", []))
        neg = _format_contributors(payload.get("top_negative_contributors", []))
        return f"increased by: {pos}; decreased by: {neg}"

    observed_fraction = dq.get("observed_feature_fraction")
    completeness = f"{observed_fraction * 100:.0f}% of the 48-hour lookback window observed" if observed_fraction is not None else "completeness unavailable"

    return f"""Synthetic benchmark case, cutoff {prediction.get('prediction_time')}, {prediction.get('elapsed_icu_hours')}h into the synthetic ICU episode.

Current severity score (SOFA-style, 0-24 scale): {prediction.get('current_sofa'):.1f}

Recovery forecast (independent horizons, same baseline):
- +24h forecast: {recovery['sofa_hat_24h']:.1f} (raw change {recovery['delta_24h']:+.2f}); {contributors_for('recovery24')}
- +48h forecast: {recovery['sofa_hat_48h']:.1f} (raw change {recovery['delta_48h']:+.2f}); {contributors_for('recovery48')}

Remaining ICU stay-time forecast: {icu['remaining_hours']:.1f} hours; {contributors_for('icu_stay_time')}

New organ-support initiation risk (next 24h, calibrated): {support['probability_24h']*100:.1f}% (raw model score {support['raw_probability']*100:.1f}%), frozen decision threshold {support['threshold']*100:.1f}%, currently {'ABOVE' if support['alert'] else 'BELOW'} threshold; {contributors_for('organ_support')}

Data completeness for this cutoff: {completeness}."""


def generate_recommendation(*, prediction: Mapping[str, object], api_key: Optional[str] = None, model: Optional[str] = None, timeout_seconds: float = 12.0) -> AIRecommendationResult:
    key = api_key if api_key is not None else os.environ.get("GROQ_API_KEY")
    chosen_model = model or os.environ.get("GROQ_MODEL", DEFAULT_MODEL)

    if not key:
        return AIRecommendationResult(status="UNAVAILABLE", summary=None, model=chosen_model, disclaimer=DISCLAIMER, error="GROQ_API_KEY is not configured")

    payload = {
        "model": chosen_model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(prediction=prediction)},
        ],
        "temperature": 0.3,
        "max_tokens": 600,
        "reasoning_effort": "low",
    }
    try:
        response = httpx.post(
            GROQ_CHAT_COMPLETIONS_URL,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json=payload,
            timeout=timeout_seconds,
        )
        response.raise_for_status()
        body = response.json()
        text = (body["choices"][0]["message"].get("content") or "").strip()
        if not text:
            raise ValueError("empty completion (model returned no visible content)")
        return AIRecommendationResult(status="OK", summary=text, model=chosen_model, disclaimer=DISCLAIMER, error=None)
    except Exception as error:  # noqa: BLE001 - degrade gracefully for any failure mode
        return AIRecommendationResult(status="UNAVAILABLE", summary=None, model=chosen_model, disclaimer=DISCLAIMER, error=f"{type(error).__name__}: {error}")


# ---------------------------------------------------------------------------
# Trajectory Copilot — grounded Q&A layer.
#
# A separate, versioned system prompt (configs/product_v2/trajectory_copilot_
# prompt_v1.txt) from the one-shot research-note prompt above. Reuses the
# same graceful-degradation Groq call path and AIRecommendationResult shape.
# The context packet is built by the caller (api/v2_app.py) from a real
# prediction response plus optional previous-cutoff values; this module never
# reaches into the serving runtime itself and never sees raw free text.
# ---------------------------------------------------------------------------


def load_trajectory_copilot_prompt() -> str:
    return _PROMPT_PATH.read_text(encoding="utf-8")


def _format_deltas(current: Mapping[str, object], previous: Optional[Mapping[str, object]]) -> str:
    if not previous:
        return "No previous replay cutoff in this session yet — this is the first step shown."
    lines = []
    for key, label, unit, fmt in (
        ("current_sofa", "Current SOFA", "", "{:.1f}"),
        ("predicted_sofa_24h", "Predicted SOFA +24h", "", "{:.1f}"),
        ("predicted_sofa_48h", "Predicted SOFA +48h", "", "{:.1f}"),
        ("remaining_icu_hours", "Remaining ICU hours", "h", "{:.1f}"),
        ("support_probability", "Support risk (calibrated)", "%", "{:.1f}"),
    ):
        before = previous.get(key)
        after = current.get(key)
        if before is None or after is None:
            continue
        scale = 100 if unit == "%" else 1
        before_v, after_v = before * scale, after * scale
        delta = after_v - before_v
        arrow = "unchanged" if abs(delta) < 1e-9 else ("up" if delta > 0 else "down")
        lines.append(f"- {label}: {fmt.format(before_v)}{unit} -> {fmt.format(after_v)}{unit} ({arrow}, {delta:+.2f}{unit})")
    threshold_crossed = previous.get("support_alert") is not None and current.get("support_alert") is not None and previous.get("support_alert") != current.get("support_alert")
    if threshold_crossed:
        lines.append(f"- Support-risk threshold state changed: {'now ABOVE threshold' if current.get('support_alert') else 'now BELOW threshold'}")
    return "\n".join(lines) if lines else "No comparable values available for the previous cutoff."


def build_assistant_user_prompt(*, context: Mapping[str, object], question: Optional[str] = None) -> str:
    episode = context.get("episode", {})
    current = context.get("current_state", {})
    forecasts = context.get("forecasts", {})
    previous = context.get("previous_cutoff")
    contributors = context.get("top_contributors", {})
    dq = context.get("data_quality", {})
    limitations = context.get("system_limitations", [])

    def contrib_block(task: str) -> str:
        items = contributors.get(task) or []
        return _format_contributors(items) if items else "none reported"

    observed_fraction = dq.get("observed_feature_fraction")
    completeness = f"{observed_fraction * 100:.0f}% of the 48-hour lookback window observed" if observed_fraction is not None else "completeness unavailable"

    delta_context = {
        "current_sofa": current.get("current_sofa"),
        "predicted_sofa_24h": forecasts.get("predicted_sofa_24h"),
        "predicted_sofa_48h": forecasts.get("predicted_sofa_48h"),
        "remaining_icu_hours": forecasts.get("remaining_icu_hours"),
        "support_probability": forecasts.get("support_calibrated_probability"),
        "support_alert": forecasts.get("support_alert"),
    }
    previous_context = None
    if previous:
        previous_context = {
            "current_sofa": previous.get("current_sofa"),
            "predicted_sofa_24h": previous.get("predicted_sofa_24h"),
            "predicted_sofa_48h": previous.get("predicted_sofa_48h"),
            "remaining_icu_hours": previous.get("remaining_icu_hours"),
            "support_probability": previous.get("support_probability"),
            "support_alert": previous.get("support_alert"),
        }

    task_ask = f'Question from the user: "{question}"' if question else "The user has not asked a specific question yet — provide a concise default summary of the current trajectory, including what changed since the previous replay cutoff if one is available."

    return f"""Patient (synthetic benchmark case): {context.get('patient_alias', 'unknown alias')}
Cardiac subtype: {episode.get('cardiac_subtype', 'unknown')}; elapsed episode time: {episode.get('elapsed_hours', 'unknown')}h; replay cutoff: {context.get('prediction_time', 'unknown')}

Current state:
- Current SOFA (severity score, 0-24): {current.get('current_sofa')}

Forecasts at this cutoff:
- +24h forecast: {forecasts.get('predicted_sofa_24h')} (raw change {forecasts.get('delta_sofa_24')}); top drivers: {contrib_block('recovery24')}
- +48h forecast: {forecasts.get('predicted_sofa_48h')} (raw change {forecasts.get('delta_sofa_48')}); top drivers: {contrib_block('recovery48')}
- Remaining ICU stay-time: {forecasts.get('remaining_icu_hours')} hours; top drivers: {contrib_block('icu')}
- New organ-support initiation risk (24h, calibrated): {forecasts.get('support_calibrated_probability')}; raw model score {forecasts.get('support_raw_probability')}; frozen threshold {forecasts.get('support_threshold')}; currently {'ABOVE' if forecasts.get('support_alert') else 'BELOW'} threshold; top drivers: {contrib_block('support')}

What changed since the previous replay cutoff:
{_format_deltas(delta_context, previous_context)}

Data completeness for this cutoff: {completeness}

System limitations to keep in mind: {"; ".join(limitations) if limitations else "synthetic research benchmark, retrospective replay, not clinically validated"}

{task_ask}"""


def generate_assistant_response(*, context: Mapping[str, object], question: Optional[str] = None, api_key: Optional[str] = None, model: Optional[str] = None, timeout_seconds: float = 15.0) -> AIRecommendationResult:
    key = api_key if api_key is not None else os.environ.get("GROQ_API_KEY")
    chosen_model = model or os.environ.get("GROQ_MODEL", DEFAULT_MODEL)

    if not key:
        return AIRecommendationResult(status="UNAVAILABLE", summary=None, model=chosen_model, disclaimer=DISCLAIMER, error="GROQ_API_KEY is not configured")

    payload = {
        "model": chosen_model,
        "messages": [
            {"role": "system", "content": load_trajectory_copilot_prompt()},
            {"role": "user", "content": build_assistant_user_prompt(context=context, question=question)},
        ],
        "temperature": 0.3,
        "max_tokens": 700,
        "reasoning_effort": "low",
    }
    try:
        response = httpx.post(
            GROQ_CHAT_COMPLETIONS_URL,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json=payload,
            timeout=timeout_seconds,
        )
        response.raise_for_status()
        body = response.json()
        text = (body["choices"][0]["message"].get("content") or "").strip()
        if not text:
            raise ValueError("empty completion (model returned no visible content)")
        return AIRecommendationResult(status="OK", summary=text, model=chosen_model, disclaimer=DISCLAIMER, error=None)
    except Exception as error:  # noqa: BLE001 - degrade gracefully for any failure mode
        return AIRecommendationResult(status="UNAVAILABLE", summary=None, model=chosen_model, disclaimer=DISCLAIMER, error=f"{type(error).__name__}: {error}")
