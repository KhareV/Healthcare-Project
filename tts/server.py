"""Kokoro TTS microservice — reads the AI research note aloud.

Deliberately isolated from the main V2 API: Kokoro's ONNX runtime needs a
newer Python than the rest of this project is pinned to, so it runs from its
own virtualenv (`.venv-tts`) as a small, separate FastAPI process. It knows
nothing about predictions, SHAP, or the frozen models — it only turns text
the caller already has into a WAV file. Never touches scientific artifacts,
never mutates any project state, purely additive.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

logger = logging.getLogger("tts.kokoro")

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "kokoro" / "kokoro-v1.0.onnx"
VOICES_PATH = ROOT / "models" / "kokoro" / "voices-v1.0.bin"
MAX_CHARS = 4000
DEFAULT_VOICE = "af_heart"


class SpeakRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=MAX_CHARS)
    voice: str = Field(default=DEFAULT_VOICE, min_length=1)
    speed: float = Field(default=1.05, ge=0.5, le=1.8)

    model_config = {"extra": "forbid"}


def app() -> FastAPI:
    from kokoro_onnx import Kokoro

    if not MODEL_PATH.exists() or not VOICES_PATH.exists():
        raise RuntimeError(
            f"Kokoro model files not found at {MODEL_PATH} / {VOICES_PATH}. "
            "Download kokoro-v1.0.onnx and voices-v1.0.bin into models/kokoro/ first."
        )

    kokoro = Kokoro(str(MODEL_PATH), str(VOICES_PATH))
    voices = sorted(kokoro.get_voices())

    fastapi_app = FastAPI(title="Kokoro Narration Service", version="1.0.0")
    fastapi_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @fastapi_app.get("/health")
    def health():
        return {"status": "ok", "engine": "kokoro-onnx", "default_voice": DEFAULT_VOICE}

    @fastapi_app.get("/voices")
    def list_voices():
        return {"voices": voices, "default": DEFAULT_VOICE}

    @fastapi_app.post("/speak")
    def speak(request: SpeakRequest):
        import soundfile as sf

        voice = request.voice if request.voice in voices else DEFAULT_VOICE
        try:
            samples, sample_rate = kokoro.create(request.text, voice=voice, speed=request.speed, lang="en-us")
        except Exception as exc:  # pragma: no cover - defensive, engine-internal failures
            logger.exception("Kokoro synthesis failed")
            raise HTTPException(status_code=502, detail=f"Narration synthesis failed: {exc}") from exc

        buffer = io.BytesIO()
        sf.write(buffer, samples, sample_rate, format="WAV")
        buffer.seek(0)
        from fastapi.responses import Response

        return Response(
            content=buffer.getvalue(),
            media_type="audio/wav",
            headers={
                "X-Kokoro-Voice": voice,
                "X-Kokoro-Duration-Seconds": f"{len(samples) / sample_rate:.2f}",
            },
        )

    return fastapi_app
