"""Future-facing adapter boundary for external record sources.

Today there is exactly one way to get a patient record into this product's
serving pipeline: manual direct entry (serving.v2.custom_record — a person
types vitals/labs and, optionally, organ-support intervals into the "Enter
My Own Record" form). That module produces one thing: the argument list for
serving.v2.custom_record.build_custom_record (patient_alias, age_years,
sex_category, observations, support_intervals).

This module defines the shape any FUTURE external source would need to
produce to plug into the exact same pipeline, with zero changes to
custom_record.py, runtime.py, or any frozen scientific code. It is an
interface only -- there is no FHIR parser, no SMART-on-FHIR client, no
network code here. Implementing one is future work; this just names the
seam so that work has an obvious, narrow place to land.

Status:
  - Manual custom entry:     IMPLEMENTED (serving.v2.custom_record)
  - FHIR R4 adapter:         FUTURE EXTENSION (not implemented)
  - SMART-on-FHIR connector: FUTURE EXTENSION (not implemented)

Do not claim either future extension works in any user-facing copy until an
implementation actually exists and passes the same round-trip verification
standard as direct entry (a real prediction through the real frozen models).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Mapping, Protocol, Sequence


@dataclass(frozen=True)
class ExternalObservation:
    """One vitals/labs reading, already resolved to a canonical concept.

    Matches the shape serving.v2.custom_record.build_custom_record expects
    for its `observations` argument. `concept` must be one of the concepts
    serving.v2.custom_record.canonical_concepts() reports as supported;
    anything else is rejected by build_custom_record itself (fail closed),
    not silently dropped by an adapter.
    """

    concept: str
    hours_since_admission: float
    value: float


@dataclass(frozen=True)
class ExternalSupportInterval:
    """One organ-support interval, already resolved to this project's
    frozen support vocabulary (configs/event_dict_v2.yaml). Matches the
    shape build_custom_record expects for `support_intervals`."""

    kind: str  # "vasopressor" | "ventilation"
    start_hour: float
    end_hour: float | None  # None = currently active
    agent: str | None = None  # required for vasopressor: see custom_record.VASOPRESSOR_AGENTS
    rate: float | None = None  # ug/kg/min, required for vasopressor


@dataclass(frozen=True)
class ExternalRecordPayload:
    """The exact argument set for serving.v2.custom_record.build_custom_record,
    minus `owner_user_id` and `runtime` (those are supplied by the API layer,
    from the verified caller's identity and the running service -- an
    adapter never determines who owns a record)."""

    patient_alias: str
    age_years: int
    sex_category: str
    observations: List[ExternalObservation]
    support_intervals: List[ExternalSupportInterval]
    # Free-form, adapter-specific facts worth surfacing to the caller for
    # transparency (e.g. "12 of 40 source observations mapped; 28 used
    # codes with no canonical-concept mapping and were dropped, not
    # guessed"). Never fed to the model -- informational only.
    mapping_warnings: List[str]


class ExternalRecordAdapter(Protocol):
    """What any future external record source must implement.

    An adapter's only job is translation: source format -> the canonical
    concepts and support vocabulary this project's frozen feature builder
    and SOFA computation already understand (see
    serving.v2.custom_record.canonical_concepts() and
    configs/event_dict_v2.yaml). An adapter must never:
      - invent a value for a concept the source didn't actually report;
      - silently remap an unrecognized code to a canonical concept (report
        it in mapping_warnings and drop it instead);
      - bypass build_custom_record's own validation (grid bounds, GCS
        integer requirement, positive vasopressor rate, etc.) -- those
        checks stay authoritative regardless of the source.
    """

    def parse(self, raw_source: bytes | Mapping[str, object]) -> ExternalRecordPayload:
        """Validate and translate one source record. Raise a clear,
        specific error for anything it cannot honestly translate; never
        return a partially-fabricated payload."""
        ...

    def supported_concepts(self) -> Sequence[str]:
        """Which canonical concepts (a subset of
        serving.v2.custom_record.canonical_concepts()) this adapter can
        actually populate from its source format. Used by a future review
        screen to tell a caller up front what will and won't be mapped."""
        ...
