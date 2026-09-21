"""Fixed-sensitivity LSTM models; recurrent cell is the only architecture change."""

import torch
from torch.nn.utils.rnn import pack_padded_sequence

from data.collate import CanonicalBatch
from models.gru import GRUEncoder, GRUEncoderConfig
from models.gru_icu_time import ICU_TIME_RAW_OUTPUT
from models.gru_recovery import RECOVERY_HORIZON_ORDER


class LSTMEncoder(GRUEncoder):
    """Reuse exact GRU input validation/preparation with an LSTM cell."""

    def __init__(self, config: GRUEncoderConfig) -> None:
        super().__init__(config)
        del self.gru
        self.lstm = torch.nn.LSTM(
            input_size=self.temporal_input_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=0.0,
        )

    def forward(self, batch: CanonicalBatch) -> torch.Tensor:
        aligned, lengths = self.assemble_temporal_input(batch)
        packed = pack_padded_sequence(
            aligned,
            lengths.detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, (hidden, _) = self.lstm(packed)
        encoded = self.output_dropout(hidden[-1])
        if self.config.static_dim:
            encoded = torch.cat((encoded, batch.static_features), dim=-1)
        return encoded


class RecoveryLSTM(torch.nn.Module):
    def __init__(self, config: GRUEncoderConfig) -> None:
        super().__init__()
        self.encoder = LSTMEncoder(config)
        self.head = torch.nn.Linear(self.encoder.output_dim, 2)

    @property
    def model_config(self):
        return self.encoder.config.to_dict()

    @property
    def horizon_order(self):
        return RECOVERY_HORIZON_ORDER

    def forward(self, batch: CanonicalBatch) -> torch.Tensor:
        output = self.head(self.encoder(batch))
        if output.ndim != 2 or output.shape[1] != 2:
            raise RuntimeError("recovery LSTM output contract must be [B,2]")
        return output


class ICUTimeLSTM(torch.nn.Module):
    def __init__(self, config: GRUEncoderConfig) -> None:
        super().__init__()
        self.encoder = LSTMEncoder(config)
        self.head = torch.nn.Linear(self.encoder.output_dim, 1)

    @property
    def model_config(self):
        return self.encoder.config.to_dict()

    @property
    def raw_output_meaning(self):
        return ICU_TIME_RAW_OUTPUT

    def forward(self, batch: CanonicalBatch) -> torch.Tensor:
        output = self.head(self.encoder(batch))
        if output.ndim != 2 or output.shape[1] != 1:
            raise RuntimeError("ICU-time LSTM output contract must be [B,1]")
        return output


class OrganSupportLSTM(torch.nn.Module):
    def __init__(self, config: GRUEncoderConfig) -> None:
        super().__init__()
        self.encoder = LSTMEncoder(config)
        self.head = torch.nn.Linear(self.encoder.output_dim, 1)

    @property
    def model_config(self):
        return self.encoder.config.to_dict()

    def forward(self, batch: CanonicalBatch) -> torch.Tensor:
        logits = self.head(self.encoder(batch))
        if logits.ndim != 2 or logits.shape[1] != 1:
            raise RuntimeError("support LSTM output contract must be [B,1]")
        return logits


def trainable_parameter_count(model: torch.nn.Module) -> int:
    """Descriptive evidence only; hidden size is never changed to match counts."""

    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
