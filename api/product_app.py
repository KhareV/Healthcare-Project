"""Product FastAPI composition root: the corrected, frozen V3 bundle is the serving authority.

`uvicorn api.product_app:app --factory`

Startup FAILS CLOSED if the V3 bundle cannot be loaded and verified; there is no V2 fallback.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI

from api.v2_app import build_v2_app
from serving.v3.backend import V3Backend


def build_product_app(root: Path) -> FastAPI:
    backend = V3Backend(root)  # raises V3ServingError on any hash/load problem
    return build_v2_app(root, backend=backend)


def app() -> FastAPI:
    from dotenv import load_dotenv

    root = Path(__file__).resolve().parents[1]
    load_dotenv(root / ".env")
    return build_product_app(root)
