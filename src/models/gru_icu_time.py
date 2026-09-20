"""Scalar GRU model for remaining time until the current ICU stay ends."""

import json
from pathlib import Path
from typing import Union

import torch

from data.collate import CanonicalBatch
from models.gru import GRUEncoder, GRUEncoderConfig
from models.icu_time_postprocess import ICU_TIME_POSTPROCESS_VERSION
from training.checkpoint import load_checkpoint, metadata_sidecar_path


ICU_TIME_RAW_OUTPUT = "log1p_remaining_current_icu_hours"
PathLike = Union[str, Path]


class ICUTimeArtifactError(RuntimeError):
    """Raised when an ICU-time checkpoint is incompatible."""


class ICUTimeGRU(torch.nn.Module):
    """Predict one raw log1p remaining-current-ICU-time value per row."""

    def __init__(self, config: GRUEncoderConfig) -> None:
        super().__init__()
        self.encoder = GRUEncoder(config)
        self.head = torch.nn.Linear(config.hidden_dim, 1)

    @property
    def model_config(self):
        return self.encoder.config.to_dict()

    @property
    def raw_output_meaning(self) -> str:
        return ICU_TIME_RAW_OUTPUT

    def forward(self, batch: CanonicalBatch) -> torch.Tensor:
        output = self.head(self.encoder(batch))
        if output.ndim != 2 or output.shape[1] != 1:
            raise RuntimeError("ICU-time output contract must be [B,1]")
        return output


def load_icu_time_bundle(
    checkpoint_path: PathLike,
    *,
    expected_tensor_contract_version: str,
    expected_feature_schema_version: str,
):
    """Construct and hash/version-verify one ICU-time model checkpoint."""

    checkpoint = Path(checkpoint_path)
    try:
        preliminary = json.loads(
            metadata_sidecar_path(checkpoint).read_text(encoding="utf-8")
        )
        config = GRUEncoderConfig(**preliminary["model_config"])
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise ICUTimeArtifactError("ICU-time checkpoint model config is invalid") from error
    model = ICUTimeGRU(config)
    metadata = load_checkpoint(
        checkpoint,
        model,
        expected_tensor_contract_version=expected_tensor_contract_version,
        expected_feature_schema_version=expected_feature_schema_version,
    )
    if metadata.get("task") != "icu_stay_time" or metadata.get("model_family") != "gru":
        raise ICUTimeArtifactError("checkpoint is not an ICU-stay-time GRU")
    if metadata.get("raw_output_meaning") != ICU_TIME_RAW_OUTPUT:
        raise ICUTimeArtifactError("ICU-time raw output meaning is incompatible")
    if metadata.get("postprocess_version") != ICU_TIME_POSTPROCESS_VERSION:
        raise ICUTimeArtifactError("ICU-time postprocess version is incompatible")
    return model, metadata
