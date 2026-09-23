"""Stage-4 composition root: binds the frozen real serving bundle into the
existing, unmodified FastAPI factory.

Mirrors ``api/main.py``'s own comment: production artifacts are never loaded
at import time. Call ``build_real_app`` explicitly (a script, test, or ASGI
entrypoint) to actually resolve and load the frozen Stage-3 models.
"""

from pathlib import Path
from typing import Tuple

from fastapi import FastAPI

from api.main import create_app
from serving.real.bundle import Stage4Resolution, resolve_stage4_bundle


def build_real_app(root: Path = None) -> Tuple[FastAPI, Stage4Resolution]:
    root = (root or Path(__file__).resolve().parents[1]).resolve()
    resolution = resolve_stage4_bundle(root)
    application = create_app(pipeline=resolution.pipeline)
    return application, resolution
