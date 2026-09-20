"""Recovery serving orchestration around the canonical display transform.

This module never computes SOFA components and never accesses future SOFA.
The current state and original-unit deltas must arrive through explicit,
versioned handoff interfaces.
"""

import math
from dataclasses import dataclass
from typing import Optional, Protocol, Tuple

import torch

from models.gru_recovery import RECOVERY_HORIZON_ORDER
from models.recovery_output import reconstruct_absolute_sofa
from serving.interfaces import ModelIdentity
from vedant_infra.hashing import is_sha256


ORIGINAL_DELTA_DOMAIN = "original_sofa_delta_points"
ALLOWED_TRANSFORM_PROVENANCE = (
    "model_native_original_units",
    "frozen_inverse_transform_applied_once",
)


class RecoveryServingError(RuntimeError):
    """Recovery output cannot be safely reconstructed for serving."""


@dataclass(frozen=True)
class CurrentSOFAState:
    stay_id: object
    prediction_time: str
    value: float
    sofa_version: str
    source_version: str
    source_sha256: str
    component_observed: Optional[Tuple[bool, ...]] = None

    def validate(self) -> None:
        if isinstance(self.value, bool) or not isinstance(self.value, (int, float)):
            raise RecoveryServingError("current SOFA must be numeric")
        if not math.isfinite(float(self.value)) or not 0 <= float(self.value) <= 24:
            raise RecoveryServingError("current SOFA must be finite within [0,24]")
        if not self.prediction_time or not self.sofa_version or not self.source_version:
            raise RecoveryServingError("current SOFA provenance is incomplete")
        if not is_sha256(self.source_sha256):
            raise RecoveryServingError("current SOFA source hash is invalid")
        if self.component_observed is not None and (
            len(self.component_observed) != 6
            or any(not isinstance(value, bool) for value in self.component_observed)
        ):
            raise RecoveryServingError(
                "current SOFA component-observed flags must contain six booleans"
            )


@dataclass(frozen=True)
class OriginalUnitRecoveryDeltas:
    delta_24h: float
    delta_48h: float
    output_domain: str
    horizon_order: Tuple[str, str]
    transform_provenance: str

    def validate(self) -> None:
        if self.output_domain != ORIGINAL_DELTA_DOMAIN:
            raise RecoveryServingError(
                "recovery output must be in original SOFA-delta units"
            )
        if self.horizon_order != RECOVERY_HORIZON_ORDER:
            raise RecoveryServingError("recovery horizon order is incompatible")
        if self.transform_provenance not in ALLOWED_TRANSFORM_PROVENANCE:
            raise RecoveryServingError(
                "recovery inverse-transform provenance is incompatible"
            )
        for value in (self.delta_24h, self.delta_48h):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise RecoveryServingError("recovery deltas must be numeric")
            if not math.isfinite(float(value)):
                raise RecoveryServingError("recovery deltas must be finite")


class CurrentSOFAProvider(Protocol):
    def current_sofa(
        self, *, stay_id: object, prediction_time: str
    ) -> CurrentSOFAState:
        """Return verified SOFA(t) for exactly the requested cutoff."""


class RecoveryDeltaAdapter(Protocol):
    def to_original_units(
        self,
        raw_output: object,
        prepared_input: object,
        model_identity: ModelIdentity,
    ) -> OriginalUnitRecoveryDeltas:
        """Normalize one selected model output exactly once into SOFA deltas."""


class ExplicitOriginalUnitDeltaAdapter:
    """Accept only outputs already carrying explicit original-unit provenance."""

    def to_original_units(
        self,
        raw_output: object,
        prepared_input: object,
        model_identity: ModelIdentity,
    ) -> OriginalUnitRecoveryDeltas:
        del prepared_input
        if model_identity.task != "recovery":
            raise RecoveryServingError("recovery adapter received another task")
        if not isinstance(raw_output, OriginalUnitRecoveryDeltas):
            raise RecoveryServingError(
                "untyped or standardized recovery output cannot be reconstructed"
            )
        raw_output.validate()
        return raw_output


class RecoveryServingPostprocessor:
    """Combine verified SOFA(t) with independent original-unit horizons."""

    def __init__(
        self,
        *,
        current_sofa_provider: CurrentSOFAProvider,
        delta_adapter: RecoveryDeltaAdapter,
        expected_sofa_version: str,
    ) -> None:
        if not expected_sofa_version:
            raise RecoveryServingError("expected SOFA version is required")
        self._current_sofa_provider = current_sofa_provider
        self._delta_adapter = delta_adapter
        self._expected_sofa_version = expected_sofa_version

    def recovery(
        self,
        raw_output: object,
        prepared_input: object,
        *,
        stay_id: object,
        prediction_time: str,
        model_identity: ModelIdentity,
    ):
        deltas = self._delta_adapter.to_original_units(
            raw_output, prepared_input, model_identity
        )
        if not isinstance(deltas, OriginalUnitRecoveryDeltas):
            raise RecoveryServingError(
                "recovery delta adapter returned an incompatible contract"
            )
        deltas.validate()
        state = self._current_sofa_provider.current_sofa(
            stay_id=stay_id, prediction_time=prediction_time
        )
        if not isinstance(state, CurrentSOFAState):
            raise RecoveryServingError(
                "current SOFA provider returned an incompatible contract"
            )
        state.validate()
        if state.stay_id != stay_id or state.prediction_time != prediction_time:
            raise RecoveryServingError("current SOFA is bound to another stay or cutoff")
        if state.sofa_version != self._expected_sofa_version:
            raise RecoveryServingError("current SOFA version is incompatible")

        raw = torch.tensor(
            [[float(deltas.delta_24h), float(deltas.delta_48h)]],
            dtype=torch.float64,
        )
        display = reconstruct_absolute_sofa(
            torch.tensor([float(state.value)], dtype=torch.float64),
            raw,
            clip_for_display=True,
        )[0]
        return {
            "delta_24h": float(deltas.delta_24h),
            "delta_48h": float(deltas.delta_48h),
            "reconstructed_sofa_24h": float(display[0].item()),
            "reconstructed_sofa_48h": float(display[1].item()),
        }
