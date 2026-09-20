"""Governed Phase-2 specification and Phase-3 raw synthetic generation."""

from .spec import SyntheticSpecError, validate_phase2_specification
from .config import GENERATOR_VERSION, SyntheticConfigError, load_runtime_config
from .generator import generate, generate_to_directory
from .manifest import validate_manifest

__all__ = ["GENERATOR_VERSION", "SyntheticConfigError", "SyntheticSpecError", "generate", "generate_to_directory", "load_runtime_config", "validate_manifest", "validate_phase2_specification"]
