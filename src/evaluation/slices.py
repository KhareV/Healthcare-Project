"""Versioned, deterministic, prespecified slice assignment for Phase 15."""

from dataclasses import asdict, dataclass
from typing import Mapping, Optional, Sequence, Tuple

from evaluation.sensitivity import CompleteComponentRecord, complete_component_included
from experiments.search_governance import canonical_sha256


SLICE_IMPLEMENTATION_VERSION = "prespecified_error_slices_v1"
FORBIDDEN_SLICE_SOURCES = frozenset(
    {
        "absolute_error",
        "error",
        "residual",
        "correct_prediction",
        "prediction",
        "predicted_probability",
        "confidence",
        "target",
        "future_sofa",
        "sofa_t_plus_24",
        "sofa_t_plus_48",
    }
)


class SliceValidationError(ValueError):
    """Raised when a slice could be post-hoc, ambiguous, or non-deterministic."""


@dataclass(frozen=True)
class SliceGroup:
    name: str
    lower: Optional[float] = None
    upper: Optional[float] = None
    categories: Tuple[str, ...] = ()


@dataclass(frozen=True)
class SliceDefinition:
    slice_name: str
    source_variable: str
    allowed_tasks: Tuple[str, ...]
    groups: Tuple[SliceGroup, ...]
    missing_value_policy: str
    missing_group: Optional[str]
    mapping_version: str
    provenance: str
    status: str = "FROZEN"
    assignment_kind: str = "numeric"


@dataclass(frozen=True)
class SliceSpecification:
    spec_version: str
    slices: Tuple[SliceDefinition, ...]
    final_test_authorized: bool = False
    minimum_sample_size: Optional[int] = None
    mode: str = "synthetic"

    @property
    def content_hash(self) -> str:
        return canonical_sha256(asdict(self))


def validate_slice_specification(spec: SliceSpecification, *, require_frozen: bool) -> None:
    if spec.mode not in ("synthetic", "real"):
        raise SliceValidationError("slice specification mode must be synthetic or real")
    if not spec.spec_version or not spec.slices:
        raise SliceValidationError("slice specification requires a version and slices")
    names = [item.slice_name for item in spec.slices]
    if len(set(names)) != len(names):
        raise SliceValidationError("slice names must be unique")
    for item in spec.slices:
        source = item.source_variable.strip().lower()
        if source in FORBIDDEN_SLICE_SOURCES or any(
            token in source for token in ("error", "residual", "correct_prediction")
        ):
            raise SliceValidationError("prediction/error-derived slice sources are forbidden")
        if require_frozen and item.status != "FROZEN":
            raise SliceValidationError(item.status)
        if not item.allowed_tasks or any(
            task not in ("recovery", "icu_stay_time", "organ_support")
            for task in item.allowed_tasks
        ):
            raise SliceValidationError("slice allowed_tasks are invalid")
        if item.missing_value_policy not in ("group", "exclude"):
            raise SliceValidationError("slice missing-value policy must be group or exclude")
        if item.missing_value_policy == "group" and not item.missing_group:
            raise SliceValidationError("group missing-value policy requires missing_group")
        if item.assignment_kind not in ("numeric", "categorical", "complete_component"):
            raise SliceValidationError("unknown slice assignment kind")
        if not item.groups or len({group.name for group in item.groups}) != len(item.groups):
            raise SliceValidationError("slice groups must be nonempty and unique")
        if item.missing_group and item.missing_group not in {g.name for g in item.groups}:
            raise SliceValidationError("missing_group must be an explicitly declared group")
        if item.assignment_kind == "numeric":
            numeric = [g for g in item.groups if g.name != item.missing_group]
            previous_upper = None
            for index, group in enumerate(numeric):
                if group.categories or (group.lower is None and group.upper is None):
                    raise SliceValidationError("numeric groups require numeric bounds")
                if group.lower is not None and group.upper is not None and group.lower >= group.upper:
                    raise SliceValidationError("numeric group lower bound must precede upper")
                if index and group.lower != previous_upper:
                    raise SliceValidationError("numeric slice groups must be contiguous and ordered")
                previous_upper = group.upper
        elif item.assignment_kind == "categorical":
            categories = [value for group in item.groups for value in group.categories]
            if len(set(categories)) != len(categories):
                raise SliceValidationError("categorical values cannot map to multiple groups")


def _missing_group(definition: SliceDefinition) -> Optional[str]:
    return definition.missing_group if definition.missing_value_policy == "group" else None


def assign_slice_group(
    definition: SliceDefinition,
    metadata: Mapping[str, object],
    *,
    complete_component_record: Optional[CompleteComponentRecord] = None,
) -> Optional[str]:
    """Assign from prespecified metadata only; None means explicit exclusion."""

    if definition.assignment_kind == "complete_component":
        if complete_component_record is None:
            return _missing_group(definition)
        value = "complete" if complete_component_included(complete_component_record) else "incomplete"
        if value not in {group.name for group in definition.groups}:
            raise SliceValidationError("complete-component groups must include assignment")
        return value
    value = metadata.get(definition.source_variable)
    if value is None:
        return _missing_group(definition)
    if definition.assignment_kind == "categorical":
        rendered = str(value)
        for group in definition.groups:
            if rendered in group.categories:
                return group.name
        return _missing_group(definition)
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise SliceValidationError("numeric slice metadata must be numeric") from error
    for group in definition.groups:
        if group.name == definition.missing_group:
            continue
        lower_ok = group.lower is None or number >= group.lower
        upper_ok = group.upper is None or number < group.upper
        if lower_ok and upper_ok:
            return group.name
    raise SliceValidationError("numeric value is outside the frozen slice groups")


def task_slice_definitions(
    spec: SliceSpecification, task: str
) -> Sequence[SliceDefinition]:
    return tuple(item for item in spec.slices if task in item.allowed_tasks)
