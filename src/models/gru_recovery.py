"""Two-output recovery GRU with independent 24h and 48h horizons."""

import json
from pathlib import Path
from typing import Union

import torch

from data.collate import CanonicalBatch
from models.gru import GRUEncoder, GRUEncoderConfig
from preprocess.target_scaler import RecoveryTargetScaler
from training.checkpoint import load_checkpoint, metadata_sidecar_path
from vedant_infra.hashing import sha256_file


RECOVERY_HORIZON_ORDER = ("delta_sofa_24h", "delta_sofa_48h")
PathLike = Union[str, Path]


class RecoveryArtifactError(RuntimeError):
    """Raised when a recovery checkpoint/scaler bundle is incompatible."""


class RecoveryGRU(torch.nn.Module):
    """Predict separately standardized DeltaSOFA24 and DeltaSOFA48."""

    def __init__(self, config: GRUEncoderConfig) -> None:
        super().__init__()
        self.encoder = GRUEncoder(config)
        self.head = torch.nn.Linear(config.hidden_dim, len(RECOVERY_HORIZON_ORDER))

    @property
    def horizon_order(self):
        return RECOVERY_HORIZON_ORDER

    @property
    def model_config(self):
        return self.encoder.config.to_dict()

    def forward(self, batch: CanonicalBatch) -> torch.Tensor:
        output = self.head(self.encoder(batch))
        if output.ndim != 2 or output.shape[1] != 2:
            raise RuntimeError("recovery output contract must be [B,2]")
        return output


def load_recovery_bundle(
    checkpoint_path: PathLike,
    *,
    repository_root: PathLike,
    expected_tensor_contract_version: str,
    expected_feature_schema_version: str,
):
    """Construct and verify a recovery model plus its target scaler artifact."""

    checkpoint = Path(checkpoint_path)
    try:
        preliminary = json.loads(
            metadata_sidecar_path(checkpoint).read_text(encoding="utf-8")
        )
        config = GRUEncoderConfig(**preliminary["model_config"])
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise RecoveryArtifactError("recovery checkpoint model config is invalid") from error
    model = RecoveryGRU(config)
    metadata = load_checkpoint(
        checkpoint,
        model,
        expected_tensor_contract_version=expected_tensor_contract_version,
        expected_feature_schema_version=expected_feature_schema_version,
    )
    if metadata.get("task") != "recovery" or metadata.get("model_family") != "gru":
        raise RecoveryArtifactError("checkpoint is not a recovery GRU")
    if metadata.get("horizon_order") != list(RECOVERY_HORIZON_ORDER):
        raise RecoveryArtifactError("recovery checkpoint horizon order is incompatible")
    scaler_reference = metadata.get("target_scaler_artifact")
    scaler_hash = metadata.get("target_scaler_sha256")
    if not isinstance(scaler_reference, str) or Path(scaler_reference).is_absolute():
        raise RecoveryArtifactError("target scaler reference must be repository-relative")
    scaler_path = Path(repository_root) / scaler_reference
    if not isinstance(scaler_hash, str) or sha256_file(scaler_path) != scaler_hash:
        raise RecoveryArtifactError("target scaler SHA-256 mismatch")
    scaler = RecoveryTargetScaler.load(scaler_path)
    if scaler.tensor_contract_version != expected_tensor_contract_version:
        raise RecoveryArtifactError("target scaler tensor contract mismatch")
    if scaler.feature_schema_version != expected_feature_schema_version:
        raise RecoveryArtifactError("target scaler feature schema mismatch")
    return model, scaler, metadata
