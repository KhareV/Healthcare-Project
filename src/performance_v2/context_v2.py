"""Phase 2: the V2 cutoff-context vector and its three compact variants.

Reuses Phase 1's exact, already-validated Group B / Group F feature builders
(src/performance_v2/diagnostic_features.py) verbatim -- no new feature is
invented here, and no Phase-1 feature is silently renamed or reinterpreted.

Verified semantic contract (src/labels/recovery.py:build_recovery_label):
``baseline_sofa = SOFA scored exactly at prediction_time`` (the label
builder's own ``baseline = _checked_score(score_at(prediction_time), ...)``
call). Group B's ``B__current_sofa`` is therefore exactly SOFA(t), the
"current SOFA at prediction time" the Phase-1/Phase-2 spec requires -- not a
different quantity, and not recomputed by a second SOFA implementation.

Variants (all DEVELOPMENT_VALIDATION_ONLY until Phase 3 freezes one):

  V0:      no V2 context at all (frozen Benchmark-v1 information only).
  B_MIN:   elapsed_episode_hours_at_t, current_sofa_at_t (2 features).
           Drops cutoff_index/hours_since_first_eligible_cutoff, which are
           exact affine functions of elapsed_episode_hours_at_t
           (elapsed = 24 + 6*cutoff_index = 24 + hours_since_first_eligible),
           removing the linear redundancy called out in the Phase-2 spec.
  B_FULL:  all four Group-B features.
  B_PLUS_F: B_FULL plus the exact eight Group-F features (support duration/
            transition state) -- nothing else.
"""

from __future__ import annotations

from typing import Dict, Mapping, Sequence, Tuple

from performance_v2.diagnostic_features import build_group_b, build_group_f

VARIANT_NAMES = ("V0", "B_MIN", "B_FULL", "B_PLUS_F")

_B_MIN_KEYS = ("B__elapsed_episode_hours_at_t", "B__current_sofa")
_B_FULL_KEYS = (
    "B__elapsed_episode_hours_at_t", "B__cutoff_index",
    "B__hours_since_first_eligible_cutoff", "B__current_sofa",
)
_F_KEYS = (
    "F__vasopressor__current_state", "F__vasopressor__duration_in_state_bins",
    "F__vasopressor__transitions_in_window", "F__vasopressor__bins_since_last_transition",
    "F__invasive_ventilation__current_state", "F__invasive_ventilation__duration_in_state_bins",
    "F__invasive_ventilation__transitions_in_window", "F__invasive_ventilation__bins_since_last_transition",
)

VARIANT_KEYS: Mapping[str, Tuple[str, ...]] = {
    "V0": (),
    "B_MIN": _B_MIN_KEYS,
    "B_FULL": _B_FULL_KEYS,
    "B_PLUS_F": _B_FULL_KEYS + _F_KEYS,
}


def raw_context_features(row: Mapping[str, object]) -> Dict[str, float]:
    """All context features Phase 2 may draw from, keyed by their exact
    Phase-1 name. Superset of every variant; a variant simply selects keys."""

    features = dict(build_group_b(row, baseline_sofa=row.get("baseline_sofa")))
    features.update(build_group_f(row))
    return features


def context_vector(row: Mapping[str, object], variant: str) -> Tuple[float, ...]:
    if variant not in VARIANT_KEYS:
        raise ValueError("unknown feature variant: " + repr(variant))
    keys = VARIANT_KEYS[variant]
    if not keys:
        return ()
    features = raw_context_features(row)
    return tuple(features[key] for key in keys)


def context_feature_names(variant: str) -> Tuple[str, ...]:
    if variant not in VARIANT_KEYS:
        raise ValueError("unknown feature variant: " + repr(variant))
    return VARIANT_KEYS[variant]
