"""Manual synthetic support-label handoff; no OFF/ON derivation is implemented."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Tuple, Union

from torch.utils.data import Dataset

from data.dataset import CanonicalTensorDataset, DatasetContractError


PathLike = Union[str, Path]
SYNTHETIC_SUPPORT_ARTIFACT_KIND = (
    "SYNTHETIC_MANUAL_ORGAN_SUPPORT_LABEL_HANDOFF_NOT_REAL_LABELS"
)


@dataclass(frozen=True)
class SupportLabelRecord:
    stay_id: object
    prediction_time: str
    split: str
    label: Optional[int]
    eligible: bool
    audit_case: str
    audit_metadata: Mapping[str, object]

    @property
    def key(self) -> Tuple[object, str]:
        return self.stay_id, self.prediction_time


@dataclass(frozen=True)
class SyntheticSupportLabelContract:
    contract_version: str
    event_dictionary_version: str
    ownership_status: str
    records: Tuple[SupportLabelRecord, ...]
    interface_cases: Tuple[Mapping[str, object], ...]


def load_synthetic_support_contract(path: PathLike) -> SyntheticSupportLabelContract:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DatasetContractError("synthetic support label artifact is unreadable") from error
    if payload.get("artifact_kind") != SYNTHETIC_SUPPORT_ARTIFACT_KIND:
        raise DatasetContractError("support artifact is not explicitly synthetic/manual")
    records = []
    seen = set()
    for raw in payload.get("records", []):
        record = SupportLabelRecord(
            stay_id=raw["stay_id"],
            prediction_time=raw["prediction_time"],
            split=raw["split"],
            label=raw["label"],
            eligible=raw["eligible"],
            audit_case=raw["audit_case"],
            audit_metadata=raw["audit_metadata"],
        )
        if record.key in seen:
            raise DatasetContractError("duplicate synthetic support label key")
        seen.add(record.key)
        if not isinstance(record.eligible, bool):
            raise DatasetContractError("support eligibility must be boolean")
        if record.eligible and record.label not in (0, 1):
            raise DatasetContractError("eligible support label must be binary")
        if not record.eligible and record.label is not None:
            raise DatasetContractError("ineligible support label must remain null")
        records.append(record)
    if not records:
        raise DatasetContractError("synthetic support label artifact has no records")
    return SyntheticSupportLabelContract(
        contract_version=payload["contract_version"],
        event_dictionary_version=payload["event_dictionary_version"],
        ownership_status=payload["ownership_status"],
        records=tuple(records),
        interface_cases=tuple(payload.get("interface_cases", [])),
    )


class SupportLabelOverlayDataset(Dataset):
    """Replace support target/mask from a supplied handoff without exposing audit data."""

    def __init__(
        self,
        base_dataset: CanonicalTensorDataset,
        contract: SyntheticSupportLabelContract,
    ) -> None:
        self.base_dataset = base_dataset
        self.contract = contract
        by_key = {record.key: record for record in contract.records}
        selected = []
        for example in base_dataset.examples:
            key = (example.stay_id, example.prediction_time)
            record = by_key.get(key)
            if record is None:
                raise DatasetContractError("canonical row lacks support label handoff")
            if record.split != base_dataset.partition:
                raise DatasetContractError("support label split conflicts with canonical row")
            selected.append(record)
        self._records = tuple(selected)

    @property
    def partition(self) -> str:
        return self.base_dataset.partition

    @property
    def examples(self):
        return self.base_dataset.examples

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int):
        item = self.base_dataset[index]
        record = self._records[index]
        targets = dict(item["targets"])
        eligibility = dict(item["eligibility"])
        targets["organ_support"] = record.label
        eligibility["organ_support"] = record.eligible
        result = dict(item)
        result["targets"] = targets
        result["eligibility"] = eligibility
        # Audit metadata is deliberately not returned to the model-facing item.
        return result

    def audit_record(self, index: int) -> SupportLabelRecord:
        return self._records[index]
