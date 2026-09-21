"""Reusable unidirectional GRU encoder for the frozen eight-bin contract."""

from dataclasses import asdict, dataclass
from typing import Mapping, Optional

import torch
from torch.nn.utils.rnn import pack_padded_sequence

from data.collate import CanonicalBatch


SEQUENCE_LENGTH = 8
LEGAL_HIDDEN_DIMS = (64, 128, 192)
LEGAL_LAYER_COUNTS = (1, 2)


class ModelContractError(ValueError):
    """Raised when model configuration or input violates the frozen contract."""


@dataclass(frozen=True)
class GRUEncoderConfig:
    feature_dim: int
    hidden_dim: int
    num_layers: int
    dropout: float
    include_observation_mask: bool = True
    include_tslo: bool = False
    static_dim: int = 0
    bidirectional: bool = False

    def validate(self) -> None:
        if self.feature_dim < 1:
            raise ModelContractError("feature_dim must be positive")
        if self.hidden_dim not in LEGAL_HIDDEN_DIMS:
            raise ModelContractError("hidden_dim is outside the frozen GRU search space")
        if self.num_layers not in LEGAL_LAYER_COUNTS:
            raise ModelContractError("num_layers is outside the frozen GRU search space")
        if not 0.10 <= self.dropout <= 0.40:
            raise ModelContractError("dropout is outside the frozen GRU search space")
        if not self.include_observation_mask:
            raise ModelContractError("observation masks are mandatory model information")
        if isinstance(self.static_dim, bool) or not isinstance(self.static_dim, int) or self.static_dim < 0:
            raise ModelContractError("static_dim must be a nonnegative integer")
        if self.bidirectional:
            raise ModelContractError("bidirectional recurrent lookahead is not permitted")

    def to_dict(self) -> Mapping[str, object]:
        return asdict(self)


class GRUEncoder(torch.nn.Module):
    """Encode genuine ICU-history bins while excluding prefix padding."""

    def __init__(self, config: GRUEncoderConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        multiplier = 2 + int(config.include_tslo)
        self.temporal_input_dim = config.feature_dim * multiplier
        self.gru = torch.nn.GRU(
            input_size=self.temporal_input_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=0.0,
        )
        # The searched dropout is applied once to the final temporal representation.
        self.output_dropout = torch.nn.Dropout(config.dropout)
        self.output_dim = config.hidden_dim + config.static_dim

    def _validate_batch(self, batch: CanonicalBatch) -> None:
        if batch.sequence.ndim != 3:
            raise ModelContractError("sequence must have shape [B,8,F]")
        batch_size, timesteps, feature_dim = batch.sequence.shape
        if timesteps != SEQUENCE_LENGTH:
            raise ModelContractError("sequence must contain exactly 8 timesteps")
        if feature_dim != self.config.feature_dim:
            raise ModelContractError("sequence feature_dim does not match model config")
        if batch.padding_mask.shape != (batch_size, SEQUENCE_LENGTH):
            raise ModelContractError("padding_mask must have shape [B,8]")
        if batch.observation_mask.shape != batch.sequence.shape:
            raise ModelContractError("observation_mask must have shape [B,8,F]")
        if self.config.static_dim == 0:
            if batch.static_features is not None:
                raise ModelContractError("static features are present but disabled")
        elif batch.static_features is None or batch.static_features.shape != (batch_size, self.config.static_dim):
            raise ModelContractError("configured static input must have shape [B,S]")
        elif not torch.isfinite(batch.static_features).all():
            raise ModelContractError("static features must be finite")
        if self.config.include_tslo:
            if batch.tslo is None or batch.tslo.shape != batch.sequence.shape:
                raise ModelContractError("configured TSLO input must have shape [B,8,F]")
        elif batch.tslo is not None:
            raise ModelContractError("TSLO is present but disabled by model configuration")
        for mask_row in batch.padding_mask:
            values = mask_row.tolist()
            first_valid = next((i for i, value in enumerate(values) if not value), len(values))
            if any(values[first_valid:]):
                raise ModelContractError("padding must be a contiguous prefix")

    def _left_align_valid_history(self, inputs: torch.Tensor, padding: torch.Tensor):
        lengths = (~padding).sum(dim=1)
        if torch.any(lengths <= 0):
            raise ModelContractError("every sequence requires at least one non-padding bin")
        aligned = torch.zeros_like(inputs)
        for row_index, length_tensor in enumerate(lengths):
            length = int(length_tensor.item())
            aligned[row_index, :length] = inputs[row_index, -length:]
        return aligned, lengths

    def assemble_temporal_input(self, batch: CanonicalBatch):
        """Return the exact canonical value/mask fields prepared for packing."""

        self._validate_batch(batch)
        fields = [batch.sequence, batch.observation_mask.to(batch.sequence.dtype)]
        if self.config.include_tslo:
            fields.append(batch.tslo)
        temporal = torch.cat(fields, dim=-1)
        temporal = temporal.masked_fill(batch.padding_mask.unsqueeze(-1), 0.0)
        return self._left_align_valid_history(temporal, batch.padding_mask)

    def forward(self, batch: CanonicalBatch) -> torch.Tensor:
        aligned, lengths = self.assemble_temporal_input(batch)
        packed = pack_padded_sequence(
            aligned,
            lengths.detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, hidden = self.gru(packed)
        encoded = self.output_dropout(hidden[-1])
        if self.config.static_dim:
            encoded = torch.cat((encoded, batch.static_features), dim=-1)
        return encoded
