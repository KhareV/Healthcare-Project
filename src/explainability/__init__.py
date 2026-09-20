"""Serving explanation routing contracts and fail-closed adapters.

Concrete adapters are configured and injected by the serving composition root;
the router deliberately contains no attribution algorithm or hidden defaults.
"""

from explainability.router import (
    AdapterExplanation,
    ExplanationContext,
    ExplanationResult,
    ExplanationRouter,
)

__all__ = (
    "AdapterExplanation",
    "ExplanationContext",
    "ExplanationResult",
    "ExplanationRouter",
)
