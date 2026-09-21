"""Canonical Phase-11 XGBoost matrices with exact GRU information parity."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence, Tuple

import numpy as np

from data.synthetic.config import canonical_json_bytes
from data.synthetic.provenance import sha256_file
from preprocess.target_scaler import RecoveryTargetScaler
from training.class_weights import SupportClassWeight


FLAT_FEATURE_MAP_VERSION = "synthetic_xgb_flat_feature_map_v1"
XGB_INPUT_VERSION = "synthetic_xgb_canonical_input_v1"
ALLOWED_PARTITIONS = ("train", "validation")
TASK_VIEWS = ("recovery24", "recovery48", "icu_stay_time", "organ_support")
PROHIBITED_FEATURE_TOKENS = (
    "subject_id", "stay_id", "prediction_time", "grid_index", "split",
    "eligible", "target", "delta_sofa", "future_sofa", "outtime",
    "remaining_hours", "organ_support_label", "censor", "latent",
)


class XGBCanonicalContractError(ValueError):
    """Raised when an XGBoost view diverges from the shared canonical input."""


@dataclass(frozen=True)
class StructuredModelInput:
    values: np.ndarray
    observation_mask: np.ndarray
    tslo_hours: np.ndarray
    padding_mask: np.ndarray
    statics: np.ndarray


@dataclass(frozen=True)
class CanonicalPartition:
    partition: str
    row_keys: Tuple[Tuple[object, object, str, int], ...]
    X: np.ndarray
    targets: Mapping[str, np.ndarray]
    eligibility: Mapping[str, np.ndarray]


@dataclass(frozen=True)
class TaskMatrix:
    task_view: str
    partition: str
    row_keys: Tuple[Tuple[object, object, str, int], ...]
    X: np.ndarray
    y: np.ndarray


def _json(path: Path) -> Mapping[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _canonical_hash(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def verify_phase10(root: Path) -> Mapping[str, object]:
    manifest_path = root / "artifacts/manifests/synthetic_phase10_manifest_v1.json"
    manifest = _json(manifest_path)
    if manifest.get("status") != "PHASE10_COMPLETE":
        raise XGBCanonicalContractError("BLOCKED — COMPLETE PHASE-10 FREEZE REQUIRED")
    if manifest.get("test_opened") is not False or manifest.get("test_model_ready_artifact") != "NOT_MATERIALIZED_UNTIL_G3_EQUIVALENT_FREEZE":
        raise XGBCanonicalContractError("Phase-10 final-test seal is not intact")
    if any(item["logical_name"] == "model_ready_test" for item in manifest["artifacts"]):
        raise XGBCanonicalContractError("test matrix must not exist before G3")
    for item in manifest["artifacts"]:
        path = root / item["path"]
        if sha256_file(path) != item["sha256"]:
            raise XGBCanonicalContractError("Phase-10 artifact hash mismatch: " + item["logical_name"])
    if sha256_file(root / manifest["contract_path"]) != manifest["contract_sha256"]:
        raise XGBCanonicalContractError("Phase-10 contract hash mismatch")
    if sha256_file(root / manifest["model_input_contract_path"]) != manifest["model_input_contract_sha256"]:
        raise XGBCanonicalContractError("model-input contract hash mismatch")
    return manifest


def build_flat_feature_map(root: Path) -> Mapping[str, object]:
    phase10 = verify_phase10(root)
    feature_schema_path = root / "configs/synthetic/feature_schema_v2.json"
    model_contract_path = root / phase10["model_input_contract_path"]
    preprocessor_path = root / "artifacts/preprocessors/synthetic_feature_preprocessor_v1.json"
    schema = _json(feature_schema_path)
    contract = _json(model_contract_path)
    preprocessor = _json(preprocessor_path)
    temporal = tuple(preprocessor["temporal_feature_names"])
    statics = tuple(preprocessor["encoded_static_feature_names"])
    schema_names = tuple(item["name"] for item in schema["temporal_channels"])
    units = {item["name"]: item["unit"] for item in schema["temporal_channels"]}
    if temporal != schema_names or list(statics) != contract["static_features"]["ordered_names"]:
        raise XGBCanonicalContractError("Phase-10 feature order differs from authoritative schemas")
    entries = []
    intervals = tuple(contract.get("bin_intervals", ())) or (
        "(t-48h,t-42h]", "(t-42h,t-36h]", "(t-36h,t-30h]", "(t-30h,t-24h]",
        "(t-24h,t-18h]", "(t-18h,t-12h]", "(t-12h,t-6h]", "(t-6h,t]",
    )

    def add(name, information_type, *, bin_index=None, feature_name=None, unit=None):
        entries.append({
            "flat_index": len(entries), "flat_name": name,
            "source_family": "canonical_shared_model_input",
            "information_type": information_type, "bin_index": bin_index,
            "bin_order": None if bin_index is None else "OLDEST_TO_NEWEST",
            "bin_interval": None if bin_index is None else intervals[bin_index],
            "canonical_feature_name": feature_name, "original_unit_or_type": unit,
            "source_feature_schema_version": schema["schema_version"],
            "source_feature_schema_sha256": sha256_file(feature_schema_path),
        })

    for information_type, suffix, unit_source in (
        ("value", "value", units), ("observation_mask", "mask", {}), ("tslo", "tslo", {}),
    ):
        for bin_index in range(8):
            for feature_name in temporal:
                unit = unit_source.get(feature_name, "bool" if information_type == "observation_mask" else "hours")
                add(f"bin{bin_index}__{feature_name}__{suffix}", information_type, bin_index=bin_index, feature_name=feature_name, unit=unit)
    for bin_index in range(8):
        add(f"padding__bin{bin_index}", "padding", bin_index=bin_index, unit="bool")
    for feature_name in statics:
        add(f"static__{feature_name}", "static", feature_name=feature_name, unit="float32_encoded")
    names = [item["flat_name"] for item in entries]
    if len(entries) != contract["information_parity"]["xgboost_flat_dimension"] or len(set(names)) != len(names):
        raise XGBCanonicalContractError("flat feature dimension or names violate the model-input contract")
    lowered = tuple(name.lower() for name in names)
    for token in PROHIBITED_FEATURE_TOKENS:
        if any(token in name for name in lowered):
            raise XGBCanonicalContractError("prohibited field entered flat feature map: " + token)
    return {
        "map_version": FLAT_FEATURE_MAP_VERSION,
        "status": "FROZEN_SYNTHETIC_AUTHORIZED",
        "decision_authority": "USER_DELEGATED_AI_PROJECT_DECISION",
        "dtype": "float32",
        "block_order": ["value", "observation_mask", "tslo", "padding", "static"],
        "temporal_bins": 8, "temporal_feature_count": len(temporal),
        "encoded_static_count": len(statics), "flat_dimension": len(entries),
        "feature_schema_version": schema["schema_version"],
        "feature_schema_sha256": sha256_file(feature_schema_path),
        "model_input_contract_version": contract["contract_version"],
        "model_input_contract_sha256": sha256_file(model_contract_path),
        "preprocessor_sha256": sha256_file(preprocessor_path),
        "entries": entries,
    }


def validate_flat_feature_map(payload: Mapping[str, object]) -> None:
    entries = payload.get("entries")
    if payload.get("map_version") != FLAT_FEATURE_MAP_VERSION or not isinstance(entries, list):
        raise XGBCanonicalContractError("unsupported flat feature map")
    if payload.get("block_order") != ["value", "observation_mask", "tslo", "padding", "static"]:
        raise XGBCanonicalContractError("flat block order changed")
    dimension = payload.get("flat_dimension")
    if dimension != len(entries) or [item.get("flat_index") for item in entries] != list(range(len(entries))):
        raise XGBCanonicalContractError("flat indices are not exact contiguous order")
    names = [item.get("flat_name") for item in entries]
    if len(set(names)) != len(names) or any(not isinstance(name, str) or not name for name in names):
        raise XGBCanonicalContractError("flat feature names must be unique and nonempty")
    f = int(payload["temporal_feature_count"])
    s = int(payload["encoded_static_count"])
    expected_types = ["value"] * (8 * f) + ["observation_mask"] * (8 * f) + ["tslo"] * (8 * f) + ["padding"] * 8 + ["static"] * s
    if [item.get("information_type") for item in entries] != expected_types or dimension != 24 * f + 8 + s:
        raise XGBCanonicalContractError("flat feature map omits, adds, or reorders canonical information")


def _structured(row: Mapping[str, object], flat_map: Mapping[str, object]) -> StructuredModelInput:
    f = int(flat_map["temporal_feature_count"])
    s = int(flat_map["encoded_static_count"])
    arrays = StructuredModelInput(
        values=np.asarray(row["sequence_values"], dtype=np.float32),
        observation_mask=np.asarray(row["observation_mask"], dtype=np.bool_),
        tslo_hours=np.asarray(row["tslo_hours"], dtype=np.float32),
        padding_mask=np.asarray(row["padding_mask"], dtype=np.bool_),
        statics=np.asarray(row["static_features"], dtype=np.float32),
    )
    if arrays.values.shape != (8, f) or arrays.observation_mask.shape != (8, f) or arrays.tslo_hours.shape != (8, f):
        raise XGBCanonicalContractError("temporal Phase-10 shape mismatch")
    if arrays.padding_mask.shape != (8,) or arrays.statics.shape != (s,):
        raise XGBCanonicalContractError("padding/static Phase-10 shape mismatch")
    if not np.isfinite(arrays.values).all() or not np.isfinite(arrays.tslo_hours).all() or not np.isfinite(arrays.statics).all():
        raise XGBCanonicalContractError("Phase-10 model inputs must be finite")
    if tuple(row["temporal_feature_names"]) != tuple(
        item["canonical_feature_name"] for item in flat_map["entries"][:f]
    ):
        raise XGBCanonicalContractError("temporal feature order mismatch")
    expected_static = tuple(item["canonical_feature_name"] for item in flat_map["entries"][-s:])
    if tuple(row["static_feature_names"]) != expected_static:
        raise XGBCanonicalContractError("static feature order mismatch")
    if np.any(arrays.values[arrays.padding_mask] != 0.0):
        raise XGBCanonicalContractError("padded value cells must remain zero")
    return arrays


def flatten_row(row: Mapping[str, object], flat_map: Mapping[str, object]) -> tuple[np.ndarray, StructuredModelInput]:
    """Return the XGBoost flat view and the lossless shared structured view."""
    validate_flat_feature_map(flat_map)
    structured = _structured(row, flat_map)
    flat = np.concatenate((
        structured.values.reshape(-1),
        structured.observation_mask.astype(np.float32).reshape(-1),
        structured.tslo_hours.reshape(-1),
        structured.padding_mask.astype(np.float32),
        structured.statics,
    )).astype(np.float32, copy=False)
    if flat.shape != (flat_map["flat_dimension"],) or not np.isfinite(flat).all():
        raise XGBCanonicalContractError("flat XGBoost row has invalid dimension or values")
    return flat, structured


def model_information_views(
    row: Mapping[str, object], flat_map: Mapping[str, object]
) -> Mapping[str, object]:
    """Derive both family views from one transformed canonical row.

    This is the post-Phase-10 equivalent of ``temporal_information_views``:
    it does not preprocess or select features, and the XGBoost view must be
    exactly reversible to the structured GRU information.
    """
    flat, structured = flatten_row(row, flat_map)
    assert_exact_reconstruction(structured, unflatten_row(flat, flat_map))
    return {"gru": structured, "xgboost": flat}


def unflatten_row(flat: np.ndarray, flat_map: Mapping[str, object]) -> StructuredModelInput:
    validate_flat_feature_map(flat_map)
    vector = np.asarray(flat, dtype=np.float32)
    f, s = int(flat_map["temporal_feature_count"]), int(flat_map["encoded_static_count"])
    if vector.shape != (flat_map["flat_dimension"],):
        raise XGBCanonicalContractError("cannot unflatten incompatible vector")
    cursor = 0
    values = vector[cursor:cursor + 8 * f].reshape(8, f); cursor += 8 * f
    masks_raw = vector[cursor:cursor + 8 * f].reshape(8, f); cursor += 8 * f
    tslo = vector[cursor:cursor + 8 * f].reshape(8, f); cursor += 8 * f
    padding_raw = vector[cursor:cursor + 8]; cursor += 8
    statics = vector[cursor:cursor + s]
    if not np.isin(masks_raw, (0.0, 1.0)).all() or not np.isin(padding_raw, (0.0, 1.0)).all():
        raise XGBCanonicalContractError("mask/padding values must be exactly binary")
    return StructuredModelInput(values.copy(), masks_raw.astype(np.bool_), tslo.copy(), padding_raw.astype(np.bool_), statics.copy())


def assert_exact_reconstruction(expected: StructuredModelInput, actual: StructuredModelInput) -> None:
    for field in ("values", "observation_mask", "tslo_hours", "padding_mask", "statics"):
        if not np.array_equal(getattr(expected, field), getattr(actual, field)):
            raise XGBCanonicalContractError("flatten/unflatten parity failure: " + field)


def _read_jsonl(path: Path) -> list[Mapping[str, object]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


class Phase10XGBData:
    """Read-only train/validation matrix gateway; test is structurally forbidden."""

    def __init__(self, root: Path, flat_map_path: Path) -> None:
        self.root = root
        self.phase10 = verify_phase10(root)
        self.flat_map_path = flat_map_path
        self.flat_map = _json(flat_map_path)
        validate_flat_feature_map(self.flat_map)
        expected_map = build_flat_feature_map(root)
        if self.flat_map != expected_map:
            raise XGBCanonicalContractError("flat feature map differs from authoritative reconstruction")
        self.flat_map_sha256 = sha256_file(flat_map_path)
        self.preprocessor_path = root / "artifacts/preprocessors/synthetic_feature_preprocessor_v1.json"
        if self.flat_map["preprocessor_sha256"] != sha256_file(self.preprocessor_path):
            raise XGBCanonicalContractError("flat map preprocessor lineage mismatch")
        self.recovery_scaler_path = root / "artifacts/preprocessors/recovery_target_scaler_synthetic_v1.json"
        self.class_weight_path = root / "artifacts/preprocessors/support_class_weight_synthetic_v1.json"
        self.recovery_scaler = RecoveryTargetScaler.load(self.recovery_scaler_path)
        self.support_class_weight = SupportClassWeight.load(self.class_weight_path)
        self._artifacts = {item["logical_name"]: item for item in self.phase10["artifacts"]}

    def load_partition(self, partition: str) -> CanonicalPartition:
        if partition not in ALLOWED_PARTITIONS:
            raise XGBCanonicalContractError("HARD FAIL — PHASE-11 XGBOOST TEST ACCESS FORBIDDEN")
        artifact = self._artifacts.get("model_ready_" + partition)
        if artifact is None:
            raise XGBCanonicalContractError("Phase-10 partition artifact is absent")
        rows = _read_jsonl(self.root / artifact["path"])
        row_keys = []
        vectors = []
        targets = {name: [] for name in TASK_VIEWS}
        eligibility = {name: [] for name in TASK_VIEWS}
        seen = set()
        for row in rows:
            if row["split"] != partition:
                raise XGBCanonicalContractError("row split conflicts with requested partition")
            key = (row["subject_id"], row["stay_id"], row["prediction_time"], int(row["grid_index"]))
            if key in seen:
                raise XGBCanonicalContractError("duplicate canonical row key")
            seen.add(key); row_keys.append(key)
            views = model_information_views(row, self.flat_map)
            vector = views["xgboost"]
            vectors.append(vector)
            source = {
                "recovery24": (row["delta_sofa_24"], row["recovery24_eligible"]),
                "recovery48": (row["delta_sofa_48"], row["recovery48_eligible"]),
                "icu_stay_time": (row["icu_time_log1p"], row["icu_time_eligible"]),
                "organ_support": (row["organ_support_label"], row["organ_support_eligible"]),
            }
            for name, (target, eligible) in source.items():
                eligibility[name].append(bool(eligible))
                targets[name].append(np.nan if target is None else float(target))
        return CanonicalPartition(
            partition=partition, row_keys=tuple(row_keys),
            X=np.stack(vectors).astype(np.float32, copy=False),
            targets={name: np.asarray(values, dtype=np.float32) for name, values in targets.items()},
            eligibility={name: np.asarray(values, dtype=np.bool_) for name, values in eligibility.items()},
        )

    def task_matrix(self, partition: CanonicalPartition, task_view: str) -> TaskMatrix:
        if partition.partition not in ALLOWED_PARTITIONS:
            raise XGBCanonicalContractError("test partition cannot form a task matrix")
        if task_view not in TASK_VIEWS:
            raise XGBCanonicalContractError("unsupported task view")
        mask = partition.eligibility[task_view]
        selected = partition.targets[task_view][mask]
        if not np.isfinite(selected).all():
            raise XGBCanonicalContractError("eligible targets must be finite")
        if task_view == "recovery24":
            selected = (selected - self.recovery_scaler.mean_24) / self.recovery_scaler.scale_24
        elif task_view == "recovery48":
            selected = (selected - self.recovery_scaler.mean_48) / self.recovery_scaler.scale_48
        elif task_view == "organ_support" and not np.isin(selected, (0.0, 1.0)).all():
            raise XGBCanonicalContractError("support targets must be binary")
        indices = np.flatnonzero(mask)
        return TaskMatrix(
            task_view=task_view, partition=partition.partition,
            row_keys=tuple(partition.row_keys[index] for index in indices),
            X=partition.X[mask], y=selected.astype(np.float32, copy=False),
        )


def row_keys_hash(keys: Sequence[Tuple[object, object, str, int]]) -> str:
    return _canonical_hash([list(key) for key in keys])


def matrix_content_hash(matrix: np.ndarray, keys: Sequence[Tuple[object, object, str, int]]) -> str:
    values = np.asarray(matrix, dtype="<f4", order="C")
    digest = hashlib.sha256()
    digest.update(values.tobytes(order="C"))
    digest.update(canonical_json_bytes([list(key) for key in keys]))
    return digest.hexdigest()
