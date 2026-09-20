"""Specification-only contracts for future synthetic cardiac data generation.

This package deliberately contains no generator implementation and emits no data.
"""

from .spec import SyntheticSpecError, validate_phase2_specification

__all__ = ["SyntheticSpecError", "validate_phase2_specification"]
