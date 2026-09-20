"""Versioned, hash-verified PyTorch checkpoint helpers."""

import json
import math
from pathlib import Path
from typing import Mapping, Optional, Tuple, Union

import torch

from vedant_infra.hashing import is_sha256, sha256_file


PathLike = Union[str, Path]
REQUIRED_METADATA_FIELDS = (
    "run_id",
    "task",
    "model_family",
    "seed",
    "epoch",
    "config_hash",
    "tensor_contract_version",
    "feature_schema_version",
    "split_hash",
    "code_commit",
    "validation_value",
    "synthetic_smoke_test",
)


class CheckpointError(RuntimeError):
    """Raised for missing provenance, corruption, or incompatibility."""


def _validate_metadata(metadata: Mapping[str, object]) -> None:
    missing = [field for field in REQUIRED_METADATA_FIELDS if field not in metadata]
    if missing:
        raise CheckpointError("checkpoint metadata missing: " + ", ".join(missing))
    for field in (
        "run_id",
        "task",
        "model_family",
        "tensor_contract_version",
        "feature_schema_version",
        "code_commit",
    ):
        if not isinstance(metadata[field], str) or not metadata[field].strip():
            raise CheckpointError("checkpoint {} must be a non-empty string".format(field))
    for field in ("config_hash", "split_hash"):
        if not isinstance(metadata[field], str) or not is_sha256(metadata[field]):
            raise CheckpointError("checkpoint {} is not SHA-256".format(field))
    if metadata["synthetic_smoke_test"] is not True:
        raise CheckpointError("Phase 4 checkpoint must be marked synthetic_smoke_test")
    if isinstance(metadata["seed"], bool) or not isinstance(metadata["seed"], int):
        raise CheckpointError("checkpoint seed must be an integer")
    if isinstance(metadata["epoch"], bool) or not isinstance(metadata["epoch"], int):
        raise CheckpointError("checkpoint epoch must be an integer")
    validation_value = metadata["validation_value"]
    if isinstance(validation_value, bool) or not isinstance(validation_value, (int, float)):
        raise CheckpointError("checkpoint validation_value must be numeric")
    if not math.isfinite(float(validation_value)):
        raise CheckpointError("checkpoint validation_value must be finite")


def metadata_sidecar_path(checkpoint_path: PathLike) -> Path:
    path = Path(checkpoint_path)
    return path.with_name(path.name + ".metadata.json")


def save_checkpoint(
    checkpoint_path: PathLike,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    metadata: Mapping[str, object],
) -> Tuple[Path, str]:
    """Save state, hash it, and write a deterministic metadata sidecar."""

    _validate_metadata(metadata)
    path = Path(checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metadata": dict(metadata),
        },
        path,
    )
    digest = sha256_file(path)
    sidecar = dict(metadata)
    sidecar["checkpoint_file"] = path.name
    sidecar["checkpoint_sha256"] = digest
    metadata_sidecar_path(path).write_text(
        json.dumps(sidecar, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path, digest


def load_checkpoint(
    checkpoint_path: PathLike,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    *,
    expected_tensor_contract_version: str,
    expected_feature_schema_version: str,
) -> Mapping[str, object]:
    """Verify hash/versions before restoring model and optimizer state."""

    path = Path(checkpoint_path)
    sidecar_path = metadata_sidecar_path(path)
    try:
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CheckpointError("checkpoint metadata sidecar is unreadable") from error
    if not isinstance(sidecar, dict):
        raise CheckpointError("checkpoint metadata sidecar must be an object")
    expected_hash = sidecar.get("checkpoint_sha256")
    if not isinstance(expected_hash, str) or not is_sha256(expected_hash):
        raise CheckpointError("checkpoint sidecar hash is missing or malformed")
    if sha256_file(path) != expected_hash:
        raise CheckpointError("checkpoint SHA-256 mismatch")
    if sidecar.get("checkpoint_file") != path.name:
        raise CheckpointError("checkpoint filename mismatch")
    if sidecar.get("tensor_contract_version") != expected_tensor_contract_version:
        raise CheckpointError("tensor contract version mismatch")
    if sidecar.get("feature_schema_version") != expected_feature_schema_version:
        raise CheckpointError("feature schema version mismatch")

    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise CheckpointError("checkpoint payload cannot be loaded safely") from error
    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        raise CheckpointError("checkpoint payload structure is invalid")
    payload_metadata = payload.get("metadata")
    if not isinstance(payload_metadata, Mapping):
        raise CheckpointError("checkpoint payload metadata is invalid")
    _validate_metadata(payload_metadata)
    expected_payload_metadata = {
        key: value
        for key, value in sidecar.items()
        if key not in ("checkpoint_file", "checkpoint_sha256")
    }
    if dict(payload_metadata) != expected_payload_metadata:
        raise CheckpointError("checkpoint payload metadata differs from sidecar")
    try:
        model.load_state_dict(payload["model_state_dict"], strict=True)
        if optimizer is not None:
            optimizer.load_state_dict(payload["optimizer_state_dict"])
    except (KeyError, TypeError, ValueError, RuntimeError) as error:
        raise CheckpointError("checkpoint state is incompatible") from error
    return sidecar
