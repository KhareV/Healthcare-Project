"""Stable canonical batch contract and deterministic DataLoader creation."""

import random
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.dataset import CanonicalTensorDataset, DatasetContractError


@dataclass(frozen=True)
class CanonicalBatch:
    identifiers: Mapping[str, Tuple[object, ...]]
    sequence: torch.Tensor
    padding_mask: torch.Tensor
    observation_mask: torch.Tensor
    tslo: Optional[torch.Tensor]
    static_features: Optional[torch.Tensor]
    targets: Mapping[str, torch.Tensor]
    eligibility: Mapping[str, torch.Tensor]
    versions: Mapping[str, str]
    feature_names: Tuple[str, ...]

    def to(self, device: torch.device) -> "CanonicalBatch":
        return CanonicalBatch(
            identifiers=self.identifiers,
            sequence=self.sequence.to(device),
            padding_mask=self.padding_mask.to(device),
            observation_mask=self.observation_mask.to(device),
            tslo=None if self.tslo is None else self.tslo.to(device),
            static_features=(
                None
                if self.static_features is None
                else self.static_features.to(device)
            ),
            targets={key: value.to(device) for key, value in self.targets.items()},
            eligibility={
                key: value.to(device) for key, value in self.eligibility.items()
            },
            versions=self.versions,
            feature_names=self.feature_names,
        )


def _collate_optional_tensors(
    values: Sequence[Optional[torch.Tensor]], name: str
) -> Optional[torch.Tensor]:
    if all(value is None for value in values):
        return None
    if any(value is None for value in values):
        raise DatasetContractError("{} is inconsistently present in a batch".format(name))
    return torch.stack([value for value in values if value is not None])


def collate_canonical_examples(
    items: Sequence[Mapping[str, object]],
) -> CanonicalBatch:
    """Collate without inferring masks or converting absent targets to zero."""

    if not items:
        raise DatasetContractError("cannot collate an empty batch")
    feature_names = tuple(items[0]["feature_names"])
    versions = dict(items[0]["versions"])
    for item in items:
        if tuple(item["feature_names"]) != feature_names:
            raise DatasetContractError("feature order mismatch inside batch")
        if dict(item["versions"]) != versions:
            raise DatasetContractError("schema/version mismatch inside batch")

    target_names = ("recovery24", "recovery48", "icu_time", "organ_support")
    targets = {}  # type: Dict[str, torch.Tensor]
    eligibility = {}  # type: Dict[str, torch.Tensor]
    for name in target_names:
        mask = torch.tensor(
            [bool(item["eligibility"][name]) for item in items], dtype=torch.bool
        )
        raw_targets = [item["targets"][name] for item in items]
        for raw_target, eligible in zip(raw_targets, mask.tolist()):
            if eligible and raw_target is None:
                raise DatasetContractError("eligible target {} is absent".format(name))
            if not eligible and raw_target is not None:
                raise DatasetContractError("ineligible target {} is populated".format(name))
        targets[name] = torch.tensor(
            [float("nan") if value is None else float(value) for value in raw_targets],
            dtype=torch.float32,
        )
        eligibility[name] = mask

    identifier_names = ("subject_id", "stay_id", "prediction_time", "grid_index", "split")
    identifiers = {
        name: tuple(item["identifiers"][name] for item in items)
        for name in identifier_names
    }
    return CanonicalBatch(
        identifiers=identifiers,
        sequence=torch.stack([item["sequence"] for item in items]),
        padding_mask=torch.stack([item["padding_mask"] for item in items]),
        observation_mask=torch.stack(
            [item["observation_mask"] for item in items]
        ),
        tslo=_collate_optional_tensors([item["tslo"] for item in items], "tslo"),
        static_features=_collate_optional_tensors(
            [item["static_features"] for item in items], "static_features"
        ),
        targets=targets,
        eligibility=eligibility,
        versions=versions,
        feature_names=feature_names,
    )


def _seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def create_dataloader(
    dataset: CanonicalTensorDataset,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int = 0,
) -> DataLoader:
    """Create a reproducible, non-label-aware DataLoader."""

    if batch_size <= 0:
        raise DatasetContractError("batch_size must be positive")
    if num_workers < 0:
        raise DatasetContractError("num_workers cannot be negative")
    if dataset.partition != "train" and shuffle:
        raise DatasetContractError("validation/test DataLoaders must not shuffle")
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_canonical_examples,
        worker_init_fn=_seed_worker if num_workers else None,
        generator=generator,
        drop_last=False,
    )

