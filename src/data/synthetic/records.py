"""Strict exported records; latent state has no serialization field."""

from dataclasses import asdict, dataclass
from typing import Optional


@dataclass(frozen=True)
class Subject:
    subject_id: str
    age_years: int
    sex_category: str
    cardiac_condition_group: str

    def export(self): return asdict(self)


@dataclass(frozen=True)
class Episode:
    subject_id: str
    stay_id: str
    intime: str
    outtime: str
    episode_sequence: int
    generator_subject_stream_id: str

    def export(self): return asdict(self)


@dataclass(frozen=True)
class RawEvent:
    event_id: str
    subject_id: str
    stay_id: str
    event_time: str
    interval_start: Optional[str]
    interval_end: Optional[str]
    concept_code: str
    value_numeric: float
    value_text: Optional[str]
    unit: str
    generator_component: str = "phase3_observation_v1"

    def export(self): return asdict(self)
