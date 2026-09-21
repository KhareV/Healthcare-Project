"""True physiology and irregular observed-event generation."""

from __future__ import annotations

from datetime import datetime, timedelta
import math
from typing import Mapping

import numpy as np

from .records import RawEvent
from .rng import subject_rng

STATE_INDEX = {"systemic_severity": 0, "cardiac_stress": 1, "respiratory_burden": 2, "renal_burden": 3, "neurologic_burden": 4}


def _bounded_measurement(baseline, effect, noise, driver, low, high, rng, integer=False):
    """Logit-link generation guarantees safety bounds without post-hoc clipping."""
    fraction = (float(baseline) - low) / (high - low)
    if not 0.0 < fraction < 1.0:
        raise ValueError(f"baseline {baseline} must be strictly inside [{low}, {high}]")
    eta0 = math.log(fraction / (1.0 - fraction))
    scale = (high - low) / 4.0
    eta = eta0 + float(effect) * driver / scale + float(noise) * rng.normal() / scale
    value = low + (high - low) / (1.0 + math.exp(-max(-40.0, min(40.0, eta))))
    return int(round(value)) if integer else round(float(value), 6)


def generate_events(config, subject, episode, intime: datetime, duration_hours: float, states: np.ndarray, inventory: Mapping[str, object], rng_subject_ordinal=None):
    rows = []
    observation = config.values["observation"]
    rng_ordinal = int(subject.subject_id[-8:]) if rng_subject_ordinal is None else int(rng_subject_ordinal)
    patient_rng = subject_rng(config.seed, rng_ordinal, "observation")
    patient_effect = float(patient_rng.normal(0.0, observation["patient_effect_scale"]))
    for concept_index, (concept, spec) in enumerate(inventory.items(), start=1):
        baseline, effect, noise, base_rate = config.values["variables"][concept]
        obs_rng = subject_rng(config.seed, rng_ordinal, "observation", concept_index)
        measure_rng = subject_rng(config.seed, rng_ordinal, "measurement", concept_index)
        dependencies = [STATE_INDEX[name] for name in spec["latent_dependencies"]]
        elapsed = float(obs_rng.exponential(1.0 / float(base_rate)))
        ordinal = 0
        while elapsed <= duration_hours:
            state = states[min(int(elapsed), len(states) - 1)]
            driver = float(np.mean(state[dependencies]))
            low, high = map(float, spec["synthetic_generation_domain"])
            value = _bounded_measurement(baseline, effect, noise, driver, low, high, measure_rng, spec["value_type"] == "integer")
            event_time = intime + timedelta(hours=elapsed)
            interval_start = interval_end = None
            if spec.get("interval_start_required"):
                start = max(intime, event_time - timedelta(hours=6))
                interval_start, interval_end = _iso(start), _iso(event_time)
            ordinal += 1
            rows.append(RawEvent(
                event_id=f"SYN-O-{int(subject.subject_id[-8:]):08d}-{concept_index:02d}-{ordinal:05d}",
                subject_id=subject.subject_id, stay_id=episode.stay_id, event_time=_iso(event_time),
                interval_start=interval_start, interval_end=interval_end, concept_code=concept,
                value_numeric=value, value_text=None, unit=spec["unit"],
            ))
            rate = float(base_rate) * math.exp(float(observation["severity_loading"]) * driver + patient_effect)
            rate = max(float(observation["minimum_rate_per_hour"]), min(float(observation["maximum_rate_per_hour"]), rate))
            elapsed += float(obs_rng.exponential(1.0 / rate))
    return rows


def _iso(value: datetime) -> str:
    return value.isoformat(timespec="microseconds").replace("+00:00", "Z")
