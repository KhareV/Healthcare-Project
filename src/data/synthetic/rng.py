"""Local deterministic RNG substreams defined by the Phase-2 policy."""

from __future__ import annotations

import numpy as np

COMPONENT_CODES = {"identity": 1, "episode": 2, "latent": 3, "physiology": 4, "observation": 5, "measurement": 6, "support": 7}


def subject_rng(primary_seed: int, subject_ordinal: int, component: str, child: int = 0) -> np.random.Generator:
    if component not in COMPONENT_CODES:
        raise ValueError(f"unknown RNG component: {component}")
    seed = np.random.SeedSequence([primary_seed, subject_ordinal, COMPONENT_CODES[component], child])
    return np.random.Generator(np.random.PCG64(seed))


def stream_id(primary_seed: int, subject_ordinal: int) -> str:
    # A digest identifies the stream without exposing an outcome or mutable RNG state.
    import hashlib
    return "SS-" + hashlib.sha256(f"synthetic_rng_policy_v1:{primary_seed}:{subject_ordinal}".encode()).hexdigest()[:16]
