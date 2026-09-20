"""In-memory continuous latent dynamics; this module exports no patient rows."""

from __future__ import annotations

import numpy as np

from .rng import subject_rng


def initial_state(config, subject_ordinal: int, group_index: int) -> tuple[np.ndarray, np.ndarray, float]:
    p = config.values["latent"]
    rng = subject_rng(config.seed, subject_ordinal, "latent")
    mean = np.asarray(p["group_means"][group_index], dtype=float)
    covariance = np.asarray(p["group_covariance"], dtype=float)
    z0 = rng.multivariate_normal(mean, covariance)
    equilibrium = z0 + rng.normal(0.0, float(p["patient_effect_scale"]), 5)
    support_propensity = float(rng.normal())
    return z0, equilibrium, support_propensity


def simulate(config, subject_ordinal: int, z0: np.ndarray, equilibrium: np.ndarray, hours: int) -> np.ndarray:
    p = config.values["latent"]
    rng = subject_rng(config.seed, subject_ordinal, "latent", child=1)
    states = np.empty((hours + 1, 5), dtype=float)
    states[0] = z0
    k = np.asarray(p["mean_reversion"], dtype=float)
    coupling = np.asarray(p["coupling_matrix"], dtype=float)
    innovation = np.asarray(p["innovation_loading"], dtype=float)
    drift = rng.normal(0.0, float(p["drift_scale"]), 5)
    shock_remaining = 0
    shock = np.zeros(5)
    for step in range(hours):
        if shock_remaining <= 0 and rng.random() < float(p["shock_probability_per_hour"]):
            shock_remaining = int(rng.integers(2, 9))
            shock = rng.normal(0.0, float(p["shock_scale"]), 5)
        pulse = shock / max(shock_remaining, 1) if shock_remaining > 0 else 0.0
        states[step + 1] = (
            states[step] + k * (equilibrium - states[step]) + drift
            + coupling @ np.tanh(states[step]) + pulse + innovation * rng.normal(size=5)
        )
        shock_remaining -= 1
    return states
