"""Training-only positive-class-weight artifact for organ-support classification."""

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Union

import torch

from data.collate import CanonicalBatch
from vedant_infra.hashing import is_sha256, sha256_file


PathLike = Union[str, Path]
SUPPORT_CLASS_WEIGHT_VERSION = "eligible_train_negative_over_positive_v1"


class ClassWeightError(ValueError):
    """Raised when class-weight provenance or labels are invalid."""


class ScientificFeasibilityError(RuntimeError):
    """Raised when frozen endpoint data cannot support classifier training."""


@dataclass(frozen=True)
class SupportClassWeight:
    eligible_train_count: int
    positive_count: int
    negative_count: int
    pos_weight: float
    source_partition: str
    split_hash: str
    label_contract_version: str
    event_dictionary_version: str
    source_config_hash: str
    formula_version: str = SUPPORT_CLASS_WEIGHT_VERSION
    formula: str = "N_negative / N_positive"
    provenance: str = "ELIGIBLE_TRAIN_ROWS_ONLY"

    def __post_init__(self) -> None:
        if self.source_partition != "train":
            raise ClassWeightError("class weight source_partition must be train")
        if self.positive_count < 1:
            raise ScientificFeasibilityError(
                "SCIENTIFIC FEASIBILITY ISSUE — NO POSITIVE TRAINING EXAMPLES"
            )
        if self.positive_count + self.negative_count != self.eligible_train_count:
            raise ClassWeightError("class-weight counts are inconsistent")
        if self.pos_weight < 0 or not math.isfinite(self.pos_weight):
            raise ClassWeightError("pos_weight must be finite and nonnegative")
        for name, value in (
            ("split_hash", self.split_hash),
            ("source_config_hash", self.source_config_hash),
        ):
            if not is_sha256(value):
                raise ClassWeightError("{} must be SHA-256".format(name))

    def save(self, path: PathLike):
        artifact_path = Path(path)
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_path.write_text(
            json.dumps(asdict(self), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return artifact_path, sha256_file(artifact_path)

    @classmethod
    def load(cls, path: PathLike) -> "SupportClassWeight":
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
            return cls(**payload)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError) as error:
            raise ClassWeightError("support class-weight artifact is invalid") from error


def fit_support_class_weight(
    loader,
    *,
    split_hash: str,
    label_contract_version: str,
    event_dictionary_version: str,
    source_config_hash: str,
) -> SupportClassWeight:
    """Compute N_negative/N_positive from eligible training labels only."""

    if getattr(getattr(loader, "dataset", None), "partition", None) != "train":
        raise ClassWeightError("support class weight may only fit a train loader")
    positive_count = 0
    negative_count = 0
    for batch in loader:
        if not isinstance(batch, CanonicalBatch):
            raise ClassWeightError("class-weight loader must emit CanonicalBatch")
        if set(batch.identifiers["split"]) != {"train"}:
            raise ClassWeightError("non-train row reached class-weight fit")
        mask = batch.eligibility["organ_support"]
        labels = batch.targets["organ_support"][mask]
        if not torch.isfinite(labels).all():
            raise ClassWeightError("eligible support labels must be finite")
        if not torch.all((labels == 0) | (labels == 1)):
            raise ClassWeightError("eligible support labels must be exactly 0 or 1")
        positive_count += int((labels == 1).sum().item())
        negative_count += int((labels == 0).sum().item())
    if positive_count == 0:
        raise ScientificFeasibilityError(
            "SCIENTIFIC FEASIBILITY ISSUE — NO POSITIVE TRAINING EXAMPLES"
        )
    eligible_count = positive_count + negative_count
    return SupportClassWeight(
        eligible_train_count=eligible_count,
        positive_count=positive_count,
        negative_count=negative_count,
        pos_weight=negative_count / positive_count,
        source_partition="train",
        split_hash=split_hash,
        label_contract_version=label_contract_version,
        event_dictionary_version=event_dictionary_version,
        source_config_hash=source_config_hash,
    )
