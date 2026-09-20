"""Pure independent-horizon recovery output transformations."""

import torch


def reconstruct_absolute_sofa(
    baseline_sofa: torch.Tensor,
    raw_delta_predictions: torch.Tensor,
    *,
    clip_for_display: bool = False,
) -> torch.Tensor:
    """Add each raw delta to the same baseline; clipping is display-only."""

    if baseline_sofa.ndim != 1:
        raise ValueError("baseline_sofa must have shape [B]")
    if raw_delta_predictions.shape != (baseline_sofa.shape[0], 2):
        raise ValueError("raw_delta_predictions must have shape [B,2]")
    if not torch.isfinite(baseline_sofa).all():
        raise ValueError("baseline_sofa must be finite")
    if not torch.isfinite(raw_delta_predictions).all():
        raise ValueError("raw_delta_predictions must be finite")
    if ((baseline_sofa < 0) | (baseline_sofa > 24)).any():
        raise ValueError("baseline_sofa must be within [0,24]")
    if not isinstance(clip_for_display, bool):
        raise ValueError("clip_for_display must be boolean")
    reconstructed = baseline_sofa.unsqueeze(1) + raw_delta_predictions
    return reconstructed.clamp(0.0, 24.0) if clip_for_display else reconstructed
