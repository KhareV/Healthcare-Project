"""Report Intelligence: turn an uploaded report's text into *candidate*
structured measurements -- never a canonical observation on its own.

Pipeline: `extract_text` (pypdf, PDF only -- image reports are not parsed in
this pass, see module docstring below) -> `extract_candidate_measurements`
(Groq, asked to map free-text lab lines onto this product's frozen
canonical concept vocabulary) -> caller stores the result as
`candidate_measurements` on the report's metadata document, each with
`confirmed: false`.

This module NEVER writes a canonical observation and NEVER touches the
serving runtime or any frozen scientific artifact. A candidate becomes a
real observation only when a human explicitly confirms it (see
`serving.v2.custom_record.append_observations_to_encounter`, called from
`POST /health-record/reports/{report_id}/confirm`) -- exactly the
human-in-the-loop boundary this product's stance on never fabricating or
silently inserting data requires.

Image reports (PNG/JPEG) are accepted for upload and storage, but are not
parsed here -- OCR is a meaningfully heavier dependency (a system Tesseract
binary) this project deliberately does not take on for a local-first demo;
an image report's `processing_status` simply stays `NOT_PARSED` if parsing
is attempted on one.
"""

from __future__ import annotations

import json
import os
import secrets
from dataclasses import dataclass
from typing import List, Mapping, Optional, Sequence

import httpx

GROQ_CHAT_COMPLETIONS_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_MODEL = "openai/gpt-oss-120b"

MAX_REPORT_TEXT_CHARS = 12_000  # keep the extraction prompt bounded and cheap


class ReportParsingUnavailable(RuntimeError):
    """Raised when parsing cannot proceed at all (no text, no API key,
    upstream failure) -- callers should record processing_status="FAILED"
    and surface the reason, never fabricate candidates."""


@dataclass(frozen=True)
class CandidateMeasurement:
    candidate_id: str
    raw_label: str
    concept: Optional[str]  # one of the frozen canonical concepts, or None if unmapped
    value: float
    unit: Optional[str]
    observed_at: Optional[str]  # best-effort real-world timestamp text from the report, if present
    confirmed: bool = False

    def as_dict(self) -> Mapping[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "raw_label": self.raw_label,
            "concept": self.concept,
            "value": self.value,
            "unit": self.unit,
            "observed_at": self.observed_at,
            "confirmed": self.confirmed,
        }


def extract_text(*, data: bytes, mime_type: str) -> str:
    """Best-effort text extraction. Returns "" for anything this pass does
    not support (e.g. an image) -- callers must treat empty text as "cannot
    parse", never as "nothing was found in an otherwise-parseable report"."""

    if mime_type != "application/pdf":
        return ""
    import io

    from pypdf import PdfReader
    from pypdf.errors import PdfReadError

    try:
        reader = PdfReader(io.BytesIO(data))
        pages = [page.extract_text() or "" for page in reader.pages]
    except (PdfReadError, ValueError):
        return ""
    return "\n".join(pages).strip()[:MAX_REPORT_TEXT_CHARS]


_EXTRACTION_SYSTEM_PROMPT = """You extract structured lab/vitals measurements from a clinical \
report's raw text for a RETROSPECTIVE, SYNTHETIC-BENCHMARK research prototype. This is NOT a \
real patient and NOT a real-time clinical tool -- you are only helping populate a candidate \
list for a human to review, never inserting anything automatically.

You will be given a fixed list of canonical concepts (with the unit each expects) and a report's \
raw extracted text. Find every distinct measurement mentioned in the text and return them as a \
JSON array. For each measurement:
- "raw_label": the label exactly as it appears in the report (e.g. "Creatinine", "Platelet Count").
- "concept": the matching canonical concept key from the provided list if you are confident it is \
  the same clinical measurement, converting units if needed to match the expected unit -- \
  otherwise null. Never guess a concept you are not confident about; null is always safe.
- "value": the numeric value, converted to the canonical concept's expected unit if you set a \
  concept, or the value exactly as reported if concept is null.
- "unit": the unit as it appears in the report (or your best interpretation), even if you also \
  converted "value" to a different unit for "concept".
- "observed_at": a timestamp or date exactly as it appears in the report if one is associated with \
  this measurement, else null. Never invent a date.

Output ONLY a JSON array (no prose, no markdown fences, no explanation). If you find no \
measurements at all, output an empty array: []."""


def _build_user_prompt(*, report_text: str, canonical_concepts: Sequence[Mapping[str, object]]) -> str:
    concept_lines = "\n".join(f"- {c['concept']}: expected unit {c['unit']!r} ({c['label']})" for c in canonical_concepts)
    return f"""Canonical concepts (use these exact keys for "concept", or null):
{concept_lines}

Report text:
---
{report_text}
---"""


def extract_candidate_measurements(
    *, report_text: str, canonical_concepts: Sequence[Mapping[str, object]],
    api_key: Optional[str] = None, model: Optional[str] = None, timeout_seconds: float = 20.0,
) -> List[CandidateMeasurement]:
    """Calls Groq once to turn `report_text` into a list of candidates.
    Raises ReportParsingUnavailable on any failure (missing key, network,
    malformed model output) -- the caller records processing_status=FAILED
    rather than ever fabricating a result."""

    if not report_text.strip():
        raise ReportParsingUnavailable("no extractable text in this report")

    key = api_key if api_key is not None else os.environ.get("GROQ_API_KEY")
    if not key:
        raise ReportParsingUnavailable("GROQ_API_KEY is not configured")
    chosen_model = model or os.environ.get("GROQ_MODEL", DEFAULT_MODEL)

    payload = {
        "model": chosen_model,
        "messages": [
            {"role": "system", "content": _EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": _build_user_prompt(report_text=report_text, canonical_concepts=canonical_concepts)},
        ],
        "temperature": 0.0,
        "max_tokens": 1500,
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
        text = (response.json()["choices"][0]["message"].get("content") or "").strip()
        # Models occasionally wrap JSON in a markdown fence despite instructions.
        if text.startswith("```"):
            text = text.strip("`")
            if text.startswith("json"):
                text = text[4:]
        rows = json.loads(text)
        if not isinstance(rows, list):
            raise ValueError("model did not return a JSON array")
    except Exception as exc:  # noqa: BLE001 - any failure mode maps to the same "unavailable" outcome
        raise ReportParsingUnavailable(f"{type(exc).__name__}: {exc}") from exc

    valid_concepts = {c["concept"] for c in canonical_concepts}
    candidates: List[CandidateMeasurement] = []
    for row in rows:
        try:
            value = float(row["value"])
        except (KeyError, TypeError, ValueError):
            continue  # a row this malformed is dropped, never guessed at
        if not (row.get("raw_label") and isinstance(row.get("raw_label"), str)):
            continue
        concept = row.get("concept")
        if concept not in valid_concepts:
            concept = None  # never trust an out-of-vocabulary concept, even if the model invented one
        candidates.append(CandidateMeasurement(
            candidate_id=f"CAND-{secrets.token_hex(6).upper()}",
            raw_label=str(row["raw_label"])[:120],
            concept=concept,
            value=value,
            unit=(str(row["unit"])[:32] if row.get("unit") else None),
            observed_at=(str(row["observed_at"])[:64] if row.get("observed_at") else None),
        ))
    return candidates
