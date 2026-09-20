"""Pure Phase-8 remaining-current-episode target."""
from dataclasses import dataclass
from datetime import datetime, timezone
import math


class ICUTimeLabelError(ValueError):
    pass


@dataclass(frozen=True)
class ICUTimeLabel:
    eligible: bool
    remaining_hours: float
    log1p_remaining_hours: float


def _utc(value: datetime) -> datetime:
    if not isinstance(value, datetime) or value.utcoffset() is None:
        raise ICUTimeLabelError("episode timestamps must be timezone-aware")
    return value.astimezone(timezone.utc)


def remaining_episode_time_label(*, prediction_time: datetime, outtime: datetime,
                                 icu_time_temporally_eligible: bool) -> ICUTimeLabel:
    if icu_time_temporally_eligible is not True:
        raise ICUTimeLabelError("canonical emitted row must be ICU-time temporally eligible")
    hours = (_utc(outtime) - _utc(prediction_time)).total_seconds() / 3600.0
    if not math.isfinite(hours) or hours <= 0:
        raise ICUTimeLabelError("legal row must have positive finite remaining episode time")
    target = math.log1p(hours)
    if not math.isfinite(target) or target <= 0:
        raise ICUTimeLabelError("ICU-time log1p target must be positive and finite")
    return ICUTimeLabel(True, hours, target)
