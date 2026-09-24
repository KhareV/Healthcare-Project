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
FINAL_SCOPE = "AUTHORIZED_FINAL_SYNTHETIC_DATA"

# Frozen, explicitly pre-registered (primary_seed, n_subjects) identities that
# may use mode="final". Each entry is bound one-to-one to a required
# subject_id_namespace so no config can claim an authorized final identity
# under the wrong namespace. Adding a new entry here is the ONLY way to
# authorize another final-scope generation; it never relaxes the check for an
# already-registered identity, and the original v1 pair is unchanged.
FINAL_IDENTITIES = {
    (20260921, 2000): "SYN",
    (402995653, 1000): "SYN-V2",
}


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

    @property
    def subject_id_namespace(self) -> str:
        return str(self.values.get("subject_id_namespace", "SYN"))


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
        if values.get("scope") != FINAL_SCOPE:
            raise SyntheticConfigError("FINAL mode refused: final scope must be AUTHORIZED_FINAL_SYNTHETIC_DATA")
        identity_key = (values.get("primary_seed"), values.get("n_subjects"))
        if identity_key not in FINAL_IDENTITIES:
            raise SyntheticConfigError(
                "final generation identity must be one of the frozen authorized (seed, n_subjects) pairs: "
                + repr(sorted(FINAL_IDENTITIES))
            )
        required_namespace = FINAL_IDENTITIES[identity_key]
        if str(values.get("subject_id_namespace", "SYN")) != required_namespace:
            raise SyntheticConfigError(
                f"final identity {identity_key} is registered to subject_id_namespace {required_namespace!r}; "
                f"observed {values.get('subject_id_namespace', 'SYN')!r}"
            )
        expected_policy = f"DETERMINISTIC_REPLACEMENT_UNTIL_{values.get('n_subjects')}_WITH_AT_LEAST_ONE_LEGAL_CUTOFF"
        if values.get("accepted_subject_policy") != expected_policy or values.get("minimum_accepted_episode_hours") != 30.0:
            raise SyntheticConfigError("final accepted-cohort replacement policy is not frozen")
        for field in ("latent_process_path", "support_process_path", "event_dictionary_path"):
            target = repo_root / str(values.get(field, ""))
            if not target.is_file():
                raise SyntheticConfigError("final config missing frozen dependency: " + field)
        support = yaml.safe_load((repo_root / values["support_process_path"]).read_text())
        latent_path = repo_root / values["latent_process_path"]
        if support.get("status") != "FROZEN_SYNTHETIC_AUTHORIZED":
            raise SyntheticConfigError("support process is not frozen and authorized")
        if support.get("latent_process_sha256") != hashlib.sha256(latent_path.read_bytes()).hexdigest():
            raise SyntheticConfigError("support/latent atomic hash binding mismatch")
    elif values.get("scope") != RUNTIME_SCOPE:
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
