"""Minimal, self-contained GRU training loop for Performance-v2 Phase 2,
reusing the existing frozen GRUEncoder/RecoveryGRU/ICUTimeGRU classes
unchanged (src/models/gru.py, gru_recovery.py, gru_icu_time.py) and the
existing CanonicalBatch contract unchanged (src/data/collate.py).

DEVELOPMENT_VALIDATION_ONLY. Bypasses the on-disk CanonicalTensorDataset/
DataLoader pipeline (built for the frozen v1 phase10 file format) in favor
of building CanonicalBatch objects directly from in-memory TRAIN/VALIDATION
rows (performance_v2.data_loading), since those rows carry the V2 context
fields the file-based v1 pipeline does not. The V2 context vector (already
train-only normalized) is concatenated into static_features before
constructing each CanonicalBatch -- see configs/performance_v2/
feature_contract_v2.json's gru_architecture_note for the full rationale.
"""

from __future__ import annotations

import math
from typing import Mapping, Sequence, Tuple

import numpy as np
import torch

from data.collate import CanonicalBatch
from models.gru import GRUEncoder, GRUEncoderConfig
from performance_v2.context_normalization import apply_normalization, fit_normalization


class SingleOutputGRU(torch.nn.Module):
    """Generic single-scalar-output head on the existing, unmodified
    GRUEncoder -- architecturally identical to models.gru_icu_time.ICUTimeGRU
    (a plain Linear(encoder.output_dim, 1) head); reused here for both the
    per-horizon recovery models and the ICU model, per the Phase-2
    instruction to train separate horizon-specific models rather than
    RecoveryGRU's combined two-output head."""

    def __init__(self, config: GRUEncoderConfig) -> None:
        super().__init__()
        self.encoder = GRUEncoder(config)
        self.head = torch.nn.Linear(self.encoder.output_dim, 1)

    def forward(self, batch: CanonicalBatch) -> torch.Tensor:
        return self.head(self.encoder(batch)).squeeze(-1)

TEMPORAL_FEATURE_NAMES = (
    "pao2__latest", "fio2__latest", "platelet_count__latest", "bilirubin_total__latest",
    "mean_arterial_pressure__latest", "glasgow_coma_scale__latest", "creatinine__latest",
    "urine_output_volume__sum", "heart_rate__latest", "systolic_blood_pressure__latest",
    "diastolic_blood_pressure__latest", "respiratory_rate__latest", "oxygen_saturation__latest",
    "temperature__latest", "lactate__latest", "vasopressor_on", "invasive_ventilation_on",
    "norepinephrine_rate", "epinephrine_rate", "dopamine_rate", "dobutamine_rate",
)


SEX_CATEGORIES = ("FEMALE", "MALE", "__UNKNOWN__")
CARDIAC_GROUPS = ("SYNTHETIC_ARRHYTHMIA", "SYNTHETIC_HF", "SYNTHETIC_ISCHEMIC", "__UNKNOWN__")


def fit_age_normalization(rows: Sequence[Mapping[str, object]]) -> Tuple[float, float]:
    """TRAIN-only age z-score parameters, matching the official encoded
    static order's age_years__z convention (artifacts/preprocessors/
    synthetic_feature_preprocessor_v1.json's static_feature_names)."""

    ages = np.asarray([float(row["age_years"]) for row in rows], dtype=np.float64)
    mean = float(ages.mean())
    std = float(ages.std())
    return mean, (std if std > 1e-8 else 1.0)


def encode_statics(row: Mapping[str, object], age_mean: float, age_std: float) -> np.ndarray:
    """Reproduces the official 8-dim encoded static order (age_years__z,
    sex_category one-hot x3, cardiac_condition_group one-hot x4) from the
    pre-split package's RAW statics (age_years, sex_category,
    cardiac_condition_group), since that package -- unlike phase10's
    train.jsonl -- stores raw, not pre-encoded, statics."""

    age_z = (float(row["age_years"]) - age_mean) / age_std
    sex = str(row["sex_category"])
    sex_onehot = [1.0 if sex == category else 0.0 for category in SEX_CATEGORIES]
    cardiac = str(row["cardiac_condition_group"])
    cardiac_onehot = [1.0 if cardiac == category else 0.0 for category in CARDIAC_GROUPS]
    return np.asarray([age_z] + sex_onehot + cardiac_onehot, dtype=np.float32)


def _row_to_tensors(row: Mapping[str, object], context_vector: np.ndarray, age_mean: float, age_std: float):
    values = row["history_values"]
    mask = row["observation_mask"]
    tslo = row["tslo_hours"]
    sequence = np.asarray([[0.0 if v is None else float(v) for v in bin_row] for bin_row in values], dtype=np.float32)
    obs_mask = np.asarray([[bool(v) for v in bin_row] for bin_row in mask], dtype=bool)
    tslo_arr = np.asarray(tslo, dtype=np.float32)
    # Bins before intime are structural padding (e.g. at the earliest
    # eligible cutoff, elapsed=24h < the 48h lookback, so the first 4 bins
    # predate ICU admission). Use the row's own frozen padding_mask -- never
    # assume no padding.
    padding = np.asarray([bool(v) for v in row["padding_mask"]], dtype=bool)
    statics = np.concatenate([encode_statics(row, age_mean, age_std), context_vector.astype(np.float32)])
    return sequence, obs_mask, tslo_arr, padding, statics


def build_canonical_batch(rows: Sequence[Mapping[str, object]], targets: np.ndarray, context_matrix: np.ndarray, device: torch.device, age_mean: float, age_std: float) -> CanonicalBatch:
    sequences, masks, tslos, paddings, statics = [], [], [], [], []
    for row, context_vector in zip(rows, context_matrix):
        sequence, obs_mask, tslo_arr, padding, static = _row_to_tensors(row, context_vector, age_mean, age_std)
        sequences.append(sequence)
        masks.append(obs_mask)
        tslos.append(tslo_arr)
        paddings.append(padding)
        statics.append(static)
    return CanonicalBatch(
        identifiers={"stay_id": tuple(r["stay_id"] for r in rows), "prediction_time": tuple(r["prediction_time"] for r in rows),
                     "subject_id": tuple(r["subject_id"] for r in rows), "grid_index": tuple(r["grid_index"] for r in rows),
                     "split": tuple(r["split"] for r in rows)},
        sequence=torch.tensor(np.stack(sequences), device=device),
        padding_mask=torch.tensor(np.stack(paddings), device=device),
        observation_mask=torch.tensor(np.stack(masks), device=device),
        tslo=torch.tensor(np.stack(tslos), device=device),
        static_features=torch.tensor(np.stack(statics), device=device),
        targets={"y": torch.tensor(targets, dtype=torch.float32, device=device)},
        eligibility={"y": torch.ones(len(rows), dtype=torch.bool, device=device)},
        versions={"performance_v2_gru": "v1"},
        feature_names=TEMPORAL_FEATURE_NAMES,
    )


def make_encoder_config(variant: str, hidden_dim: int, num_layers: int, dropout: float, include_tslo: bool = True) -> GRUEncoderConfig:
    from performance_v2.context_v2 import context_feature_names

    static_dim = 8 + len(context_feature_names(variant))
    return GRUEncoderConfig(
        feature_dim=len(TEMPORAL_FEATURE_NAMES), hidden_dim=hidden_dim, num_layers=num_layers,
        dropout=dropout, include_observation_mask=True, include_tslo=include_tslo, static_dim=static_dim,
    )


def standardize_targets(train_targets: np.ndarray):
    mean = float(np.mean(train_targets))
    std = float(np.std(train_targets))
    std = std if std > 1e-8 else 1.0
    return mean, std


def train_gru(
    model: torch.nn.Module, train_rows, train_targets: np.ndarray, train_context: np.ndarray,
    val_rows, val_targets: np.ndarray, val_context: np.ndarray, *, loss_name: str, lr: float,
    weight_decay: float, batch_size: int, max_epochs: int, patience: int, seed: int, device: torch.device,
    target_mean: float, target_std: float, age_mean: float, age_std: float,
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = torch.nn.HuberLoss(delta=1.0) if loss_name == "huber" else torch.nn.L1Loss()

    train_targets_std = (train_targets - target_mean) / target_std
    val_targets_std = (val_targets - target_mean) / target_std
    val_batch = build_canonical_batch(val_rows, val_targets_std, val_context, device, age_mean, age_std)

    n = len(train_rows)
    rng = np.random.default_rng(seed)
    best_val_loss = math.inf
    best_state = None
    epochs_without_improvement = 0
    for epoch in range(max_epochs):
        model.train()
        order = rng.permutation(n)
        for start in range(0, n, batch_size):
            batch_indices = order[start:start + batch_size]
            batch_rows = [train_rows[i] for i in batch_indices]
            batch_targets = train_targets_std[batch_indices]
            batch_context = train_context[batch_indices]
            batch = build_canonical_batch(batch_rows, batch_targets, batch_context, device, age_mean, age_std)
            optimizer.zero_grad()
            prediction = model(batch)
            loss = loss_fn(prediction, batch.targets["y"])
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_prediction = model(val_batch)
            val_loss = loss_fn(val_prediction, val_batch.targets["y"]).item()
        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val_loss


def predict_gru(model: torch.nn.Module, rows, context_matrix: np.ndarray, device: torch.device, age_mean: float, age_std: float, target_mean: float, target_std: float) -> np.ndarray:
    model.eval()
    dummy_targets = np.zeros(len(rows), dtype=np.float32)
    batch = build_canonical_batch(rows, dummy_targets, context_matrix, device, age_mean, age_std)
    with torch.no_grad():
        prediction_std = model(batch).cpu().numpy()
    return prediction_std * target_std + target_mean
