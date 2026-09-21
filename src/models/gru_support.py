"""One-logit GRU classifier for Pulkit-supplied organ-support labels."""

import json
from pathlib import Path
from typing import Union

import torch

from data.collate import CanonicalBatch
from models.gru import GRUEncoder, GRUEncoderConfig
from training.checkpoint import load_checkpoint, metadata_sidecar_path
from training.class_weights import SupportClassWeight
from vedant_infra.hashing import sha256_file


SUPPORT_OUTPUT_TYPE = "logit"
SUPPORT_PROBABILITY_TRANSFORM = "sigmoid_uncalibrated_v1"
PathLike = Union[str, Path]


class SupportArtifactError(RuntimeError):
    """Raised when a support checkpoint/class-weight bundle is incompatible."""


class OrganSupportGRU(torch.nn.Module):
    """Return one raw logit; sigmoid is intentionally external."""

    def __init__(self, config: GRUEncoderConfig) -> None:
        super().__init__()
        self.encoder = GRUEncoder(config)
        self.head = torch.nn.Linear(self.encoder.output_dim, 1)

    @property
    def model_config(self):
        return self.encoder.config.to_dict()

    def forward(self, batch: CanonicalBatch) -> torch.Tensor:
        logits = self.head(self.encoder(batch))
        if logits.ndim != 2 or logits.shape[1] != 1:
            raise RuntimeError("organ-support output contract must be [B,1]")
        return logits


def uncalibrated_support_probability(logits: torch.Tensor) -> torch.Tensor:
    """Convert finite raw logits to explicitly uncalibrated probabilities."""

    if not torch.is_floating_point(logits) or not torch.isfinite(logits).all():
        raise ValueError("support logits must be finite floating point")
    return torch.sigmoid(logits)


def load_support_bundle(
    checkpoint_path: PathLike,
    *,
    repository_root: PathLike,
    expected_tensor_contract_version: str,
    expected_feature_schema_version: str,
):
    """Verify model identity and its training-only class-weight provenance."""

    checkpoint = Path(checkpoint_path)
    try:
        preliminary = json.loads(
            metadata_sidecar_path(checkpoint).read_text(encoding="utf-8")
        )
        config = GRUEncoderConfig(**preliminary["model_config"])
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise SupportArtifactError("support checkpoint model config is invalid") from error
    model = OrganSupportGRU(config)
    metadata = load_checkpoint(
        checkpoint,
        model,
        expected_tensor_contract_version=expected_tensor_contract_version,
        expected_feature_schema_version=expected_feature_schema_version,
    )
    if metadata.get("task") != "organ_support" or metadata.get("model_family") != "gru":
        raise SupportArtifactError("checkpoint is not an organ-support GRU")
    if metadata.get("output_type") != SUPPORT_OUTPUT_TYPE:
        raise SupportArtifactError("support output type must be logit")
    if metadata.get("probability_transform") != SUPPORT_PROBABILITY_TRANSFORM:
        raise SupportArtifactError("support probability transform is incompatible")
    reference = metadata.get("class_weight_artifact")
    expected_hash = metadata.get("class_weight_sha256")
    if not isinstance(reference, str) or Path(reference).is_absolute():
        raise SupportArtifactError("class-weight reference must be repository-relative")
    artifact_path = Path(repository_root) / reference
    if not isinstance(expected_hash, str) or sha256_file(artifact_path) != expected_hash:
        raise SupportArtifactError("class-weight SHA-256 mismatch")
    class_weight = SupportClassWeight.load(artifact_path)
    if class_weight.label_contract_version != metadata.get("label_contract_version"):
        raise SupportArtifactError("support label contract version mismatch")
    if class_weight.event_dictionary_version != metadata.get("event_dictionary_version"):
        raise SupportArtifactError("support event-dictionary version mismatch")
    return model, class_weight, metadata
