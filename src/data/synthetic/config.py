"""Fail-closed Phase-3 runtime configuration loading."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping

import yaml


GENERATOR_VERSION = "synthetic_generator_v1"
RUNTIME_SCOPE = "ENGINEERING_SYNTHETIC_FIXTURE_NON_SCIENTIFIC_NOT_FINAL_DATASET"


class SyntheticConfigError(ValueError):
    """Raised for incomplete, unreviewed, or inconsistent generator parameters."""


@dataclass(frozen=True)
class RuntimeConfig:
    path: Path
    values: Mapping[str, Any]
    sha256: str

    @property
    def mode(self) -> str:
        return str(self.values["mode"])

    @property
    def seed(self) -> int:
        return int(self.values["primary_seed"])

    @property
    def n_subjects(self) -> int:
        return int(self.values["n_subjects"])


def canonical_json_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode()


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SyntheticConfigError(f"runtime config is unreadable: {path}: {error}") from error
    if not isinstance(value, dict):
        raise SyntheticConfigError("runtime config must be a JSON object")
    return value


def load_runtime_config(path: Path, repo_root: Path) -> RuntimeConfig:
    path = path.resolve()
    values = _load_json(path)
    inherited = values.get("inherits")
    if inherited:
        base_path = (repo_root / str(inherited)).resolve()
        base = _load_json(base_path)
        merged = copy.deepcopy(base)
        merged.update({k: v for k, v in values.items() if k != "inherits"})
        values = merged
    mode = values.get("mode")
    if mode not in {"fixture", "smoke", "final"}:
        raise SyntheticConfigError(f"mode expected fixture|smoke|final; observed {mode!r}")
    if mode == "final":
        spec = yaml.safe_load((repo_root / "configs/synthetic/synthetic_generator_v1.yaml").read_text())
        unresolved = spec.get("unresolved_parameters", [])
        raise SyntheticConfigError(
            "FINAL mode refused: Phase-2 required parameters remain unresolved: " + ", ".join(unresolved)
        )
    if values.get("scope") != RUNTIME_SCOPE:
        raise SyntheticConfigError("fixture/smoke scope must explicitly be non-scientific and not final")
    if not isinstance(values.get("primary_seed"), int) or values["primary_seed"] < 0:
        raise SyntheticConfigError("primary_seed expected a nonnegative integer fixture/smoke seed")
    if not isinstance(values.get("n_subjects"), int) or values["n_subjects"] < 1:
        raise SyntheticConfigError("n_subjects expected a positive integer")
    required = {"population", "calendar", "duration", "latent", "observation", "variables"}
    missing = sorted(required - values.keys())
    if missing:
        raise SyntheticConfigError("runtime config missing parameter groups: " + ", ".join(missing))
    digest = hashlib.sha256(canonical_json_bytes(values)).hexdigest()
    return RuntimeConfig(path=path, values=values, sha256=digest)
