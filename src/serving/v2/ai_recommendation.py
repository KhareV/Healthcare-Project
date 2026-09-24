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
from typing import Mapping, Optional, Sequence

import httpx

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
