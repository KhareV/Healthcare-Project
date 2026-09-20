"""Governed Phase-2 specification and Phase-3 raw synthetic generation."""

from .spec import SyntheticSpecError, validate_phase2_specification
from .config import GENERATOR_VERSION, SyntheticConfigError, load_runtime_config
from .generator import generate, generate_to_directory
from .manifest import validate_manifest
from .cohort import CohortBuild, SyntheticCohortError, build_cohort
from .cohort_manifest import build_to_directory as build_cohort_to_directory
from .cohort_manifest import validate_cohort_manifest

__all__ = ["CohortBuild", "GENERATOR_VERSION", "SyntheticCohortError", "SyntheticConfigError", "SyntheticSpecError", "build_cohort", "build_cohort_to_directory", "generate", "generate_to_directory", "load_runtime_config", "validate_cohort_manifest", "validate_manifest", "validate_phase2_specification"]
