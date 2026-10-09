"""Cohort row loading for V3 development data (fail-closed against sealed/exposed test cohorts).

Mirrors performance_v2.data_loading.load_dev_rows enrichment for the V3 cohort
directory layout (raw/cohort/timeline/features/pre_split). Sealed-test access
belongs to a separate guarded module; nothing here can open it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple, Union

from performance_v2.data_loading import _utc, load_dev_rows, load_jsonl

ALLOWED_ROLES = ("dev", "lce")
FORBIDDEN_PATH_TOKENS = ("final_test", "fresh_test", "sealed")
FORBIDDEN_COHORT_IDS = ("SYN-V3-T", "SYN-V3-TEST", "SYN-V2")
FORBIDDEN_SUBJECT_PREFIXES = ("SYN-V3-T-", "SYN-V2-")
HEAVY_FIELDS = ("history_values", "observation_mask", "tslo_hours", "padding_mask")

PRE_SPLIT_FILE = "pre_split/pre_split_scientific_package.jsonl"
STATICS_FILE = "timeline/canonical_statics.jsonl"


class CohortAccessError(RuntimeError):
    """Raised on any attempt to load a non-development role or a sealed/exposed cohort."""


@dataclass(frozen=True)
class CohortDir:
    path: Path

    @property
    def raw(self) -> Path:
        return self.path / "raw"

    @property
    def cohort(self) -> Path:
        return self.path / "cohort"

    @property
    def timeline(self) -> Path:
        return self.path / "timeline"

    @property
    def features(self) -> Path:
        return self.path / "features"

    @property
    def pre_split(self) -> Path:
        return self.path / "pre_split"

    @property
    def pre_split_path(self) -> Path:
        return self.path / PRE_SPLIT_FILE

    @property
    def statics_path(self) -> Path:
        return self.path / STATICS_FILE


def _as_cohort_dir(root: Path, cohort_dir: Union[CohortDir, Path, str]) -> CohortDir:
    path = cohort_dir.path if isinstance(cohort_dir, CohortDir) else Path(cohort_dir)
    if not path.is_absolute():
        path = root / path
    return CohortDir(path)


def _guard(root: Path, cohort: CohortDir, cohort_id: str, role: str) -> None:
    if role not in ALLOWED_ROLES:
        raise CohortAccessError(f"role {role!r} refused; only {ALLOWED_ROLES} may be loaded here")
    if cohort_id in FORBIDDEN_COHORT_IDS or any(cohort_id.startswith(p) for p in FORBIDDEN_COHORT_IDS):
        raise CohortAccessError(f"cohort {cohort_id!r} is a sealed/exposed test cohort")
    try:
        visible = str(cohort.path.resolve().relative_to(root.resolve()))
    except ValueError:
        visible = cohort.path.name
    lowered = visible.lower()
    for token in FORBIDDEN_PATH_TOKENS:
        if token in lowered:
            raise CohortAccessError(f"cohort path {visible!r} contains forbidden token {token!r}")
    if cohort.path.name.upper() in FORBIDDEN_COHORT_IDS:
        raise CohortAccessError(f"cohort directory {cohort.path.name!r} is a sealed/exposed test cohort")


def _enrich(row: Mapping[str, object], static: Mapping[str, object]) -> Dict[str, object]:
    intime = _utc(static["intime"])
    prediction_time = _utc(row["prediction_time"])
    elapsed_hours = (prediction_time - intime).total_seconds() / 3600.0
    enriched = dict(row)
    enriched["elapsed_episode_hours_at_t"] = elapsed_hours
    enriched["cutoff_index"] = int(row["grid_index"])
    enriched["hours_since_first_eligible_cutoff"] = elapsed_hours - 24.0
    enriched["age_years"] = static["age_years"]
    enriched["sex_category"] = static["sex_category"]
    enriched["cardiac_condition_group"] = static["cardiac_condition_group"]
    enriched["intime"] = static["intime"]
    return enriched


def compact_row(row: Mapping[str, object]) -> Dict[str, object]:
    """Drop the heavy 8x21 bin arrays; only valid once the feature cache has been built."""

    return {k: v for k, v in row.items() if k not in HEAVY_FIELDS}


def iter_cohort_rows(
    root: Path,
    cohort_dir: Union[CohortDir, Path, str],
    *,
    cohort_id: str,
    role: str,
    compact: bool = False,
) -> Iterator[Dict[str, object]]:
    """Lazily yield enriched rows for one development/learning-curve cohort (JSONL line streaming)."""

    root = Path(root)
    cohort = _as_cohort_dir(root, cohort_dir)
    _guard(root, cohort, cohort_id, role)
    statics = {row["stay_id"]: row for row in load_jsonl(cohort.statics_path)}
    for row in load_jsonl(cohort.pre_split_path):
        subject_id = str(row["subject_id"])
        if subject_id.startswith(FORBIDDEN_SUBJECT_PREFIXES):
            raise CohortAccessError(f"subject {subject_id!r} belongs to a sealed/exposed test namespace")
        enriched = _enrich(row, statics[row["stay_id"]])
        enriched["split"] = role
        enriched["role"] = role
        enriched["cohort_id"] = cohort_id
        yield compact_row(enriched) if compact else enriched


def load_cohort_rows(
    root: Path,
    cohort_dir: Union[CohortDir, Path, str],
    *,
    cohort_id: str,
    role: str,
    compact: bool = False,
) -> Tuple[Dict[str, object], ...]:
    return tuple(iter_cohort_rows(root, cohort_dir, cohort_id=cohort_id, role=role, compact=compact))


def iter_v1_dev_rows(root: Path, *, compact: bool = False) -> Iterator[Dict[str, object]]:
    """V1 train+validation rows only (load_dev_rows itself refuses the V1 test split)."""

    for row in load_dev_rows(Path(root)):
        tagged = dict(row)
        tagged["cohort_id"] = "SYN"
        tagged["role"] = "dev"
        yield compact_row(tagged) if compact else tagged


def load_v1_dev_rows(root: Path, *, compact: bool = False) -> Tuple[Dict[str, object], ...]:
    return tuple(iter_v1_dev_rows(root, compact=compact))


def _chunk_items(chunk_dirs: Union[Mapping[str, object], Sequence[Tuple[str, object]]]) -> List[Tuple[str, object]]:
    items = list(chunk_dirs.items()) if isinstance(chunk_dirs, Mapping) else list(chunk_dirs)
    ids = [cohort_id for cohort_id, _ in items]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate cohort ids in chunk_dirs")
    return items


def iter_dev_pool(
    root: Path,
    chunk_dirs: Union[Mapping[str, object], Sequence[Tuple[str, object]]],
    *,
    compact: bool = False,
    include_v1: bool = True,
) -> Iterator[Dict[str, object]]:
    """V1 dev rows first, then each chunk in the given order; asserts subject-id uniqueness across cohorts.

    A subject may contribute many rows, but may belong to exactly one cohort.
    """

    owner: Dict[str, str] = {}

    def checked(rows: Iterable[Dict[str, object]], cohort_id: str) -> Iterator[Dict[str, object]]:
        for row in rows:
            subject_id = str(row["subject_id"])
            previous = owner.setdefault(subject_id, cohort_id)
            if previous != cohort_id:
                raise ValueError(f"subject {subject_id} appears in cohorts {previous} and {cohort_id}")
            yield row

    if include_v1:
        yield from checked(iter_v1_dev_rows(root, compact=compact), "SYN")
    for cohort_id, cohort_dir in _chunk_items(chunk_dirs):
        yield from checked(iter_cohort_rows(root, cohort_dir, cohort_id=cohort_id, role="dev", compact=compact), cohort_id)


def assemble_dev_pool(
    root: Path,
    chunk_dirs: Union[Mapping[str, object], Sequence[Tuple[str, object]]],
    *,
    compact: bool = False,
    include_v1: bool = True,
) -> Tuple[Dict[str, object], ...]:
    return tuple(iter_dev_pool(root, chunk_dirs, compact=compact, include_v1=include_v1))
