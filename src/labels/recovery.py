"""Pure Phase-8 recovery labels orchestrating the single Phase-6 SOFA engine."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
import math
from typing import Callable, Optional, Tuple

from data.synthetic.sofa import SOFAResult
from data.timestamps import calculate_horizon_availability


class RecoveryLabelError(ValueError):
    pass


@dataclass(frozen=True)
class RecoveryLabel:
    recovery24_eligible: bool
    recovery48_eligible: bool
    delta_sofa_24: Optional[float]
    delta_sofa_48: Optional[float]
    baseline_sofa: Optional[int]
    future_sofa_24: Optional[int]
    future_sofa_48: Optional[int]
    baseline_cutoff: str
    horizon24_cutoff: str
    horizon48_cutoff: str
    component_observed_baseline: Optional[Tuple[bool, ...]]
    component_observed_24: Optional[Tuple[bool, ...]]
    component_observed_48: Optional[Tuple[bool, ...]]
    sofa_spec_version: Optional[str]
    sofa_spec_sha256: Optional[str]


def _instant(value: datetime) -> str:
    if not isinstance(value, datetime) or value.utcoffset() is None:
        raise RecoveryLabelError("recovery cutoffs must be timezone-aware datetimes")
    return value.isoformat()


def _checked_score(result: SOFAResult, expected_cutoff: datetime, stay_id: object) -> SOFAResult:
    if not isinstance(result, SOFAResult):
        raise RecoveryLabelError("SOFA provider must return the Phase-6 SOFAResult")
    result.validate()
    if result.stay_id != stay_id or datetime.fromisoformat(result.cutoff_time.replace("Z", "+00:00")) != expected_cutoff:
        raise RecoveryLabelError("SOFA result identity/cutoff mismatch")
    return result


def build_recovery_label(*, stay_id: object, prediction_time: datetime, outtime: datetime,
                         recovery24_followup_available: bool, recovery48_followup_available: bool,
                         score_at: Callable[[datetime], SOFAResult]) -> RecoveryLabel:
    """Build independent raw deltas; the same injected Phase-6 function supplies all scores."""
    availability = calculate_horizon_availability(prediction_time, outtime)
    if (recovery24_followup_available, recovery48_followup_available) != (
        availability.recovery24_followup_available, availability.recovery48_followup_available
    ):
        raise RecoveryLabelError("structural recovery flags disagree with timestamp contract")
    t24, t48 = prediction_time + timedelta(hours=24), prediction_time + timedelta(hours=48)
    if not (recovery24_followup_available or recovery48_followup_available):
        return RecoveryLabel(False, False, None, None, None, None, None, _instant(prediction_time), _instant(t24), _instant(t48), None, None, None, None, None)
    baseline = _checked_score(score_at(prediction_time), prediction_time, stay_id)
    score24 = _checked_score(score_at(t24), t24, stay_id) if recovery24_followup_available else None
    score48 = _checked_score(score_at(t48), t48, stay_id) if recovery48_followup_available else None
    scores = tuple(x for x in (baseline, score24, score48) if x is not None)
    if any((x.sofa_spec_version, x.sofa_spec_sha256) != (baseline.sofa_spec_version, baseline.sofa_spec_sha256) for x in scores):
        raise RecoveryLabelError("SOFA specification identity differs across horizons")
    d24 = None if score24 is None else float(score24.total_score - baseline.total_score)
    d48 = None if score48 is None else float(score48.total_score - baseline.total_score)
    if any(not math.isfinite(x) or not -24 <= x <= 24 for x in (d24, d48) if x is not None):
        raise RecoveryLabelError("recovery delta is outside the exact SOFA domain")
    return RecoveryLabel(recovery24_followup_available, recovery48_followup_available, d24, d48,
        baseline.total_score, None if score24 is None else score24.total_score, None if score48 is None else score48.total_score,
        _instant(prediction_time), _instant(t24), _instant(t48), baseline.component_observed,
        None if score24 is None else score24.component_observed, None if score48 is None else score48.component_observed,
        baseline.sofa_spec_version, baseline.sofa_spec_sha256)


def validate_recovery_label(label: RecoveryLabel) -> None:
    for eligible, value in ((label.recovery24_eligible, label.delta_sofa_24), (label.recovery48_eligible, label.delta_sofa_48)):
        if eligible != (value is not None):
            raise RecoveryLabelError("recovery eligibility and canonical target nullness disagree")
        if value is not None and (not math.isfinite(value) or not -24 <= value <= 24):
            raise RecoveryLabelError("recovery target is outside the finite SOFA-delta domain")
    if label.delta_sofa_24 is not None and label.delta_sofa_24 != label.future_sofa_24 - label.baseline_sofa:  # type: ignore[operator]
        raise RecoveryLabelError("24h recovery formula mismatch")
    if label.delta_sofa_48 is not None and label.delta_sofa_48 != label.future_sofa_48 - label.baseline_sofa:  # type: ignore[operator]
        raise RecoveryLabelError("48h recovery formula mismatch")
