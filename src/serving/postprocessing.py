"""Serving postprocessing adapters that reuse frozen model-domain transforms."""

from typing import Mapping, Protocol

import torch

from models.icu_time_postprocess import remaining_icu_hours_from_log_prediction


class ServingPostprocessError(RuntimeError):
    """Raised when a model-domain output cannot be safely served."""


class RecoveryResponseProvider(Protocol):
    def recovery(
        self,
        raw_output: object,
        prepared_input: object,
        **context: object,
    ) -> Mapping[str, float]:
        """Return an already implemented Phase-5-compatible recovery payload."""


class CanonicalICUTimeServingPostprocessor:
    """Delegate recovery while applying the one frozen ICU-time transform.

    Recovery reconstruction remains outside this class and is deliberately
    delegated to a supplied provider pending Pulkit Phase 12.
    """

    def __init__(self, recovery_provider: RecoveryResponseProvider) -> None:
        self._recovery_provider = recovery_provider

    def recovery(
        self,
        raw_output: object,
        prepared_input: object,
        **context: object,
    ) -> Mapping[str, float]:
        return self._recovery_provider.recovery(
            raw_output, prepared_input, **context
        )

    def icu_stay_time_hours(self, raw_output: object) -> float:
        if isinstance(raw_output, bool):
            raise ServingPostprocessError("ICU-time model output must be numeric")
        try:
            tensor = torch.as_tensor(raw_output, dtype=torch.float64)
        except (TypeError, ValueError, RuntimeError) as error:
            raise ServingPostprocessError(
                "ICU-time model output must be numeric"
            ) from error
        if tensor.numel() != 1:
            raise ServingPostprocessError(
                "ICU-time serving requires exactly one model output"
            )
        try:
            hours = remaining_icu_hours_from_log_prediction(tensor.reshape(()))
        except (TypeError, ValueError, RuntimeError) as error:
            raise ServingPostprocessError(
                "ICU-time model output cannot be safely postprocessed"
            ) from error
        value = float(hours.item())
        if value < 0:
            raise ServingPostprocessError("remaining ICU hours cannot be negative")
        return value
