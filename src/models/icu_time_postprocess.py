"""One frozen ICU-time transform used by evaluation and serving consumers."""

import torch


ICU_TIME_POSTPROCESS_VERSION = "icu_time_nonnegative_log_expm1_v1"


def remaining_icu_hours_from_log_prediction(raw_log_prediction: torch.Tensor) -> torch.Tensor:
    """Compute expm1(max(0, y_hat)) exactly in the frozen order."""

    if not torch.is_floating_point(raw_log_prediction):
        raise ValueError("raw ICU-time prediction must be floating point")
    if not torch.isfinite(raw_log_prediction).all():
        raise ValueError("raw ICU-time prediction must be finite")
    nonnegative_log = torch.clamp_min(raw_log_prediction, 0.0)
    hours = torch.expm1(nonnegative_log)
    if not torch.isfinite(hours).all():
        raise ValueError("postprocessed remaining ICU hours must be finite")
    return hours
