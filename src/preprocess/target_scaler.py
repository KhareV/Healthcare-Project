"""Training-only, horizon-specific recovery target standardization."""

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Union

import torch

from data.collate import CanonicalBatch
from vedant_infra.hashing import is_sha256, sha256_file


PathLike = Union[str, Path]
SCALER_VERSION = "recovery_target_scaler_v1"


class TargetScalerError(ValueError):
    """Raised when target scaling provenance or values are unsafe."""


@dataclass(frozen=True)
class RecoveryTargetScaler:
    mean_24: float
    scale_24: float
    mean_48: float
    scale_48: float
    eligible_count_24: int
    eligible_count_48: int
    source_partition: str
    split_hash: str
    tensor_contract_version: str
    feature_schema_version: str
    scaler_version: str = SCALER_VERSION
    variance_convention: str = "population_std_ddof_0"
    zero_variance_policy: str = "unit_scale"
    provenance: str = "TRAIN_PARTITION_ELIGIBLE_TARGETS_ONLY"

    def __post_init__(self) -> None:
        if self.source_partition != "train":
            raise TargetScalerError("target scaler source_partition must be train")
        if not is_sha256(self.split_hash):
            raise TargetScalerError("target scaler split_hash must be SHA-256")
        if self.eligible_count_24 < 1 or self.eligible_count_48 < 1:
            raise TargetScalerError("each recovery horizon needs an eligible train target")
        if self.scale_24 <= 0 or self.scale_48 <= 0:
            raise TargetScalerError("target scales must be positive")
        if not all(
            math.isfinite(value)
            for value in (self.mean_24, self.scale_24, self.mean_48, self.scale_48)
        ):
            raise TargetScalerError("target scaler statistics must be finite")

    def transform_horizon(self, values: torch.Tensor, horizon_index: int) -> torch.Tensor:
        mean, scale = self._statistics(horizon_index)
        return (values - mean) / scale

    def inverse_transform(self, standardized: torch.Tensor) -> torch.Tensor:
        if standardized.ndim < 1 or standardized.shape[-1] != 2:
            raise TargetScalerError("recovery predictions must end with horizon dimension 2")
        means = standardized.new_tensor((self.mean_24, self.mean_48))
        scales = standardized.new_tensor((self.scale_24, self.scale_48))
        return standardized * scales + means

    def _statistics(self, horizon_index: int):
        if horizon_index == 0:
            return self.mean_24, self.scale_24
        if horizon_index == 1:
            return self.mean_48, self.scale_48
        raise TargetScalerError("horizon_index must be 0 or 1")

    def to_mapping(self) -> Mapping[str, object]:
        return asdict(self)

    def save(self, path: PathLike):
        artifact_path = Path(path)
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_path.write_text(
            json.dumps(self.to_mapping(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return artifact_path, sha256_file(artifact_path)

    @classmethod
    def load(cls, path: PathLike) -> "RecoveryTargetScaler":
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
            return cls(**payload)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError) as error:
            raise TargetScalerError("target scaler artifact is invalid") from error


def _population_statistics(values):
    tensor = torch.tensor(values, dtype=torch.float64)
    mean = float(tensor.mean().item())
    raw_scale = float(tensor.std(unbiased=False).item())
    return mean, 1.0 if raw_scale == 0.0 else raw_scale


def fit_recovery_target_scaler(loader, *, split_hash: str) -> RecoveryTargetScaler:
    """Fit only from a loader whose dataset and every batch are train-labelled."""

    dataset = getattr(loader, "dataset", None)
    if getattr(dataset, "partition", None) != "train":
        raise TargetScalerError("target scaler may only fit a train-partition loader")
    values_24 = []
    values_48 = []
    versions = None
    for batch in loader:
        if not isinstance(batch, CanonicalBatch):
            raise TargetScalerError("target scaler loader must emit CanonicalBatch")
        if set(batch.identifiers["split"]) != {"train"}:
            raise TargetScalerError("non-train row reached target scaler fit")
        versions = batch.versions
        for name, destination in (("recovery24", values_24), ("recovery48", values_48)):
            mask = batch.eligibility[name]
            selected = batch.targets[name][mask]
            if not torch.isfinite(selected).all():
                raise TargetScalerError("eligible training targets must be finite")
            destination.extend(float(value) for value in selected.tolist())
    if versions is None:
        raise TargetScalerError("cannot fit target scaler from an empty loader")
    if not values_24 or not values_48:
        raise TargetScalerError("both horizons require eligible training targets")
    mean_24, scale_24 = _population_statistics(values_24)
    mean_48, scale_48 = _population_statistics(values_48)
    return RecoveryTargetScaler(
        mean_24=mean_24,
        scale_24=scale_24,
        mean_48=mean_48,
        scale_48=scale_48,
        eligible_count_24=len(values_24),
        eligible_count_48=len(values_48),
        source_partition="train",
        split_hash=split_hash,
        tensor_contract_version=versions["tensor_contract"],
        feature_schema_version=versions["feature_schema"],
    )
