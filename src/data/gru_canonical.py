"""Read-only PyTorch Dataset over accepted Phase-10 train/validation rows."""
from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from data.synthetic.provenance import sha256_file
from data.xgb_canonical import verify_phase10


class GRUCanonicalDataError(RuntimeError):
    pass


class Phase10GRUDataset(Dataset):
    """Expose the frozen transformed representation without fitting or rebuilding."""

    def __init__(self, root: Path, partition: str):
        if partition not in ("train", "validation"):
            raise GRUCanonicalDataError("HARD FAIL — FINAL GRU TEST ACCESS FORBIDDEN")
        manifest = verify_phase10(root)
        artifacts = {item["logical_name"]: item for item in manifest["artifacts"]}
        item = artifacts.get("model_ready_" + partition)
        if not item:
            raise GRUCanonicalDataError("accepted Phase-10 partition is missing")
        path = root / item["path"]
        if sha256_file(path) != item["sha256"]:
            raise GRUCanonicalDataError("accepted Phase-10 row artifact hash mismatch")
        with path.open(encoding="utf-8") as handle:
            rows = tuple(json.loads(line) for line in handle if line.strip())
        if len(rows) != item["record_count"] or any(row["split"] != partition for row in rows):
            raise GRUCanonicalDataError("accepted Phase-10 partition identity mismatch")
        first = rows[0]
        self.root = root
        self._partition = partition
        self._rows = rows
        self._feature_names = tuple(first["temporal_feature_names"])
        self._static_names = tuple(first["static_feature_names"])
        self._feature_schema_version = first["feature_schema_version"]
        self._feature_schema_sha256 = first["feature_schema_sha256"]
        self._tensor_contract_version = "synthetic_model_input_contract_v1"
        for row in rows:
            if (tuple(row["temporal_feature_names"]) != self._feature_names
                    or tuple(row["static_feature_names"]) != self._static_names
                    or row["feature_schema_version"] != self._feature_schema_version
                    or row["feature_schema_sha256"] != self._feature_schema_sha256):
                raise GRUCanonicalDataError("feature order/version drift inside Phase-10 rows")

    @property
    def partition(self):
        return self._partition

    @property
    def feature_names(self):
        return self._feature_names

    @property
    def static_feature_names(self):
        return self._static_names

    @property
    def feature_schema_version(self):
        return self._feature_schema_version

    @property
    def feature_schema_sha256(self):
        return self._feature_schema_sha256

    @property
    def tensor_contract_version(self):
        return self._tensor_contract_version

    @property
    def rows(self):
        return self._rows

    def __len__(self):
        return len(self._rows)

    def __getitem__(self, index):
        row = self._rows[index]
        return {
            "identifiers": {name: row[name] for name in ("subject_id", "stay_id", "prediction_time", "grid_index", "split")},
            "sequence": torch.tensor(row["sequence_values"], dtype=torch.float32),
            "padding_mask": torch.tensor(row["padding_mask"], dtype=torch.bool),
            "observation_mask": torch.tensor(row["observation_mask"], dtype=torch.bool),
            "tslo": torch.tensor(row["tslo_hours"], dtype=torch.float32),
            "static_features": torch.tensor(row["static_features"], dtype=torch.float32),
            "targets": {
                "recovery24": row["delta_sofa_24"], "recovery48": row["delta_sofa_48"],
                "icu_time": row["icu_time_log1p"], "organ_support": row["organ_support_label"],
            },
            "eligibility": {
                "recovery24": row["recovery24_eligible"], "recovery48": row["recovery48_eligible"],
                "icu_time": row["icu_time_eligible"], "organ_support": row["organ_support_eligible"],
            },
            "versions": {
                "tensor_contract": self._tensor_contract_version,
                "timestamp_spec": "synthetic_prediction_grid_v2",
                "feature_schema": self._feature_schema_version,
            },
            "feature_names": self._feature_names,
            "synthetic_tensorization_status": "ACCEPTED_PHASE10_TRANSFORM_ONLY",
        }


class TaskEligibleGRUDataset(Dataset):
    """Training-only eligible-row view; validation remains the full canonical grid."""

    def __init__(self, base: Phase10GRUDataset, task: str):
        if base.partition != "train":
            raise GRUCanonicalDataError("task-eligible filtering is training-only")
        if task == "recovery":
            keep = lambda row: row["recovery24_eligible"] or row["recovery48_eligible"]
        elif task == "icu_time":
            keep = lambda row: row["icu_time_eligible"]
        elif task == "organ_support":
            keep = lambda row: row["organ_support_eligible"]
        else:
            raise GRUCanonicalDataError("unsupported task-eligible view")
        self.base = base
        self.task = task
        self.indices = tuple(index for index, row in enumerate(base.rows) if keep(row))
        if not self.indices:
            raise GRUCanonicalDataError("task has no eligible training rows")

    @property
    def partition(self):
        return self.base.partition

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return self.base[self.indices[index]]
