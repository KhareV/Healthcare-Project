"""Versioned real-canonical adapter with no feature or label derivation.

Field names come exclusively from the upstream processed-schema sidecar. This
module validates and converts an already-built canonical artifact; it never
queries MIMIC, builds features, derives SOFA, or creates support states.
"""

import json
import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Mapping, Optional, Sequence, Tuple

from data.schema import (
    HISTORY_DTYPE,
    MASK_DTYPE,
    TENSOR_CONTRACT_VERSION,
    CanonicalExample,
    FeatureSchemaReference,
    TaskEligibility,
    TaskTargets,
    TensorContractError,
    temporal_information_views,
    validate_canonical_example,
)
from data.split import (
    load_split_csv,
    validate_subject_isolation,
    verify_split_artifacts,
)
from data.timestamps import (
    TIMESTAMP_SPEC_VERSION,
    RetainedICUStay,
    generate_prediction_timestamps,
)
from vedant_infra.hashing import is_sha256, sha256_file


MANIFEST_VERSION = "real_data_input_manifest_v1"
STRUCTURAL_INDEX_VERSION = "canonical_structural_index_v1"
CANONICAL_FORMAT = "canonical_json_v1"
SUPPORT_HANDOFF_VERSION = "organ_support_handoff_v1"
REQUIRED_ARTIFACTS = (
    "canonical_dataset",
    "schema_sidecar",
    "feature_schema",
    "feature_dictionary",
    "structural_index",
    "split",
    "split_metadata",
    "sofa_provenance",
    "label_spec",
    "tensor_contract",
    "timestamp_spec",
)
REQUIRED_FIELD_CONCEPTS = (
    "subject_id",
    "stay_id",
    "prediction_time",
    "grid_index",
    "intime",
    "outtime",
    "anchor_year_group",
    "split",
    "history_values",
    "padding_mask",
    "observation_mask",
    "tslo_hours",
    "static_features",
    "recovery24_eligible",
    "recovery48_eligible",
    "icu_time_eligible",
    "recovery24_delta_sofa",
    "recovery48_delta_sofa",
    "sofa_t",
    "sofa_t24",
    "sofa_t48",
    "sofa_component_observed_flags",
    "icu_remaining_hours",
    "icu_time_log1p_hours",
)


class RealDataContractError(ValueError):
    """Raised when upstream real-data artifacts violate frozen contracts."""


@dataclass(frozen=True)
class InventoryItem:
    name: str
    ref: str
    owner: str
    artifact_type: str
    artifact_version: str
    schema_version: str
    sha256: str
    creation_commit: str
    consumer_contract: str
    status: str


@dataclass(frozen=True)
class StructuralRow:
    subject_id: object
    stay_id: object
    prediction_time: datetime
    grid_index: int
    intime: datetime
    outtime: datetime
    anchor_year_group: str
    split: str

    @property
    def key(self):
        return (type(self.stay_id).__name__, str(self.stay_id), self.prediction_time.isoformat())


@dataclass(frozen=True)
class RealDataBundle:
    examples: Tuple[CanonicalExample, ...]
    structural_rows: Tuple[StructuralRow, ...]
    feature_schema: FeatureSchemaReference
    dynamic_feature_names: Tuple[str, ...]
    static_feature_names: Tuple[str, ...]
    support_status: str
    inventory: Tuple[InventoryItem, ...]
    versions: Mapping[str, str]
    preprocessing: Mapping[str, object]
    leakage_evidence: Mapping[str, object]
    canonical_content_sha256: str
    structural_content_sha256: str


def _load_json(path: Path) -> Mapping[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RealDataContractError("unreadable JSON artifact: " + str(path)) from error
    if not isinstance(value, Mapping):
        raise RealDataContractError("JSON artifact must contain an object: " + str(path))
    return value


def _safe_path(root: Path, ref: object) -> Path:
    if not isinstance(ref, str) or not ref or Path(ref).is_absolute():
        raise RealDataContractError("artifact reference must be repository-relative")
    path = (root / ref).resolve()
    if root.resolve() not in path.parents:
        raise RealDataContractError("artifact reference escapes repository root")
    return path


def _artifact_path(root: Path, record: Mapping[str, object], *, verify=True) -> Path:
    path = _safe_path(root, record.get("ref"))
    digest = record.get("sha256")
    if not isinstance(digest, str) or not is_sha256(digest):
        raise RealDataContractError("artifact inventory has malformed SHA-256")
    if verify and (not path.is_file() or sha256_file(path) != digest):
        raise RealDataContractError("artifact missing or hash mismatch: " + str(record.get("ref")))
    return path


def _parse_time(value: object, name: str) -> datetime:
    if not isinstance(value, str):
        raise RealDataContractError(name + " must be an ISO-8601 string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise RealDataContractError(name + " must be an ISO-8601 string") from error
    return parsed


def _finite(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _field(row: Mapping[str, object], field_map: Mapping[str, object], concept: str):
    name = field_map.get(concept)
    if not isinstance(name, str) or name not in row:
        raise RealDataContractError("required mapped field is absent: " + concept)
    return row[name]


def _inventory(manifest: Mapping[str, object], root: Path) -> Tuple[InventoryItem, ...]:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise RealDataContractError("input manifest artifacts must be an object")
    missing = [name for name in REQUIRED_ARTIFACTS if name not in artifacts]
    if missing:
        raise RealDataContractError("input manifest missing artifacts: " + ", ".join(missing))
    items = []
    for name in sorted(artifacts):
        record = artifacts[name]
        if not isinstance(record, Mapping):
            raise RealDataContractError("artifact inventory record is malformed: " + name)
        protected = record.get("access_policy") == "PROTECTED_NOT_OPENED_PHASE18"
        _artifact_path(root, record, verify=not protected)
        required = ("owner", "artifact_type", "artifact_version", "schema_version", "consumer_contract")
        if any(not isinstance(record.get(key), str) or not record.get(key) for key in required):
            raise RealDataContractError("artifact inventory metadata incomplete: " + name)
        items.append(
            InventoryItem(
                name=name,
                ref=str(record["ref"]),
                owner=str(record["owner"]),
                artifact_type=str(record["artifact_type"]),
                artifact_version=str(record["artifact_version"]),
                schema_version=str(record["schema_version"]),
                sha256=str(record["sha256"]),
                creation_commit=str(record.get("creation_commit", "UNAVAILABLE")),
                consumer_contract=str(record["consumer_contract"]),
                status="PROTECTED_NOT_OPENED" if protected else "HASH_VERIFIED",
            )
        )
    return tuple(items)


def _structural_rows(payload, field_map) -> Tuple[StructuralRow, ...]:
    if payload.get("index_version") != STRUCTURAL_INDEX_VERSION:
        raise RealDataContractError("structural index version mismatch")
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise RealDataContractError("structural index rows must be a list")
    structural_concepts = (
        "subject_id",
        "stay_id",
        "prediction_time",
        "grid_index",
        "intime",
        "outtime",
        "anchor_year_group",
        "split",
    )
    structural_names = {field_map[concept] for concept in structural_concepts}
    parsed = []
    seen = set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise RealDataContractError("structural row must be an object")
        if set(row) != structural_names:
            raise RealDataContractError(
                "structural index contains non-structural or missing fields"
            )
        item = StructuralRow(
            subject_id=_field(row, field_map, "subject_id"),
            stay_id=_field(row, field_map, "stay_id"),
            prediction_time=_parse_time(_field(row, field_map, "prediction_time"), "prediction_time"),
            grid_index=_field(row, field_map, "grid_index"),
            intime=_parse_time(_field(row, field_map, "intime"), "intime"),
            outtime=_parse_time(_field(row, field_map, "outtime"), "outtime"),
            anchor_year_group=_field(row, field_map, "anchor_year_group"),
            split=_field(row, field_map, "split"),
        )
        if isinstance(item.grid_index, bool) or not isinstance(item.grid_index, int):
            raise RealDataContractError("grid_index must be integer")
        if item.key in seen:
            raise RealDataContractError("BLOCKED — CANONICAL ROW IDENTITY VIOLATION")
        seen.add(item.key)
        parsed.append(item)
    return tuple(sorted(parsed, key=lambda item: item.key))


def validate_timestamp_parity(rows: Sequence[StructuralRow]) -> None:
    by_stay = {}
    for row in rows:
        identity = (type(row.stay_id).__name__, str(row.stay_id))
        signature = (row.subject_id, row.intime, row.outtime)
        if identity in by_stay and by_stay[identity][0] != signature:
            raise RealDataContractError("stay metadata conflicts across canonical rows")
        by_stay.setdefault(identity, (signature, []))[1].append(row)
    for signature, actual_rows in by_stay.values():
        subject_id, intime, outtime = signature
        expected = generate_prediction_timestamps(
            (RetainedICUStay(subject_id, actual_rows[0].stay_id, intime, outtime),)
        )
        actual = tuple((row.grid_index, row.prediction_time) for row in actual_rows)
        wanted = tuple((row.grid_index, row.prediction_time) for row in expected)
        if actual != wanted:
            raise RealDataContractError("BLOCKED — TIMESTAMP CONTRACT MISMATCH")


def _validate_split(rows: Sequence[StructuralRow], split_path: Path) -> None:
    split_rows = load_split_csv(split_path)
    validate_subject_isolation(split_rows)
    expected = {(type(row.subject_id).__name__, str(row.subject_id)): row for row in split_rows}
    observed = {}
    for row in rows:
        key = (type(row.subject_id).__name__, str(row.subject_id))
        if key in observed and observed[key] != row.split:
            raise RealDataContractError("subject appears under multiple split assignments")
        observed[key] = row.split
        frozen = expected.get(key)
        if frozen is None or frozen.split != row.split or frozen.anchor_year_group != row.anchor_year_group:
            raise RealDataContractError("canonical split conflicts with split_v1")


def _validate_tslo(example: CanonicalExample, sentinel: float) -> None:
    assert example.tslo_hours is not None
    for feature_index in range(len(example.history_values[0])):
        prior = None
        for bin_index in range(8):
            value = example.tslo_hours[bin_index][feature_index]
            if example.padding_mask[bin_index]:
                if value != sentinel:
                    raise RealDataContractError("padded TSLO must use frozen sentinel")
                continue
            if example.observation_mask[bin_index][feature_index]:
                if value == sentinel or not 0 <= value <= 6:
                    raise RealDataContractError("observed-bin TSLO is incompatible")
                prior = float(value)
            elif prior is None:
                if value != sentinel:
                    raise RealDataContractError("BLOCKED — TSLO CONTRACT MISMATCH")
            else:
                prior += 6.0
                if float(value) != prior:
                    raise RealDataContractError("TSLO does not follow genuine observations")


def _support_by_key(payload, field_map) -> Mapping[tuple, Mapping[str, object]]:
    if payload.get("handoff_version") != SUPPORT_HANDOFF_VERSION:
        raise RealDataContractError("organ-support handoff version mismatch")
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise RealDataContractError("support handoff rows must be a list")
    indexed = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise RealDataContractError("support handoff row is malformed")
        split = _field(row, field_map, "split")
        if split == "test":
            raise RealDataContractError("test support labels cannot be opened in Phase 18")
        key = (
            type(_field(row, field_map, "stay_id")).__name__,
            str(_field(row, field_map, "stay_id")),
            _parse_time(
                _field(row, field_map, "prediction_time"),
                "prediction_time",
            ).isoformat(),
        )
        if key in indexed:
            raise RealDataContractError("duplicate support canonical key")
        indexed[key] = row
    return indexed


def validate_support_evidence(row, prediction_time, outtime):
    """Receiver-side validation of a supplied Pulkit handoff row."""
    eligible = row.get("organ_support_eligible")
    target = row.get("organ_support_target")
    vaso_on = row.get("vasopressor_on_at_t")
    vent_on = row.get("ventilation_on_at_t")
    if not all(isinstance(value, bool) for value in (eligible, vaso_on, vent_on)):
        raise RealDataContractError("support eligibility/state evidence must be boolean")
    if target not in (None, 0, 1):
        raise RealDataContractError("support target must be null, 0, or 1")
    onsets = []
    for name, already_on in (("vasopressor_onset_time", vaso_on), ("ventilation_onset_time", vent_on)):
        value = row.get(name)
        if value is not None and not already_on:
            onsets.append(_parse_time(value, name))
    qualifying = [time for time in onsets if prediction_time < time <= prediction_time + timedelta(hours=24) and time <= outtime]
    if eligible != (not (vaso_on and vent_on) and (bool(qualifying) or outtime >= prediction_time + timedelta(hours=24))):
        raise RealDataContractError("BLOCKED — ORGAN-SUPPORT HANDOFF VIOLATION")
    expected = 1 if qualifying else (0 if eligible else None)
    if target != expected:
        raise RealDataContractError("BLOCKED — ORGAN-SUPPORT HANDOFF VIOLATION")
    return eligible, target


def load_real_bundle(root: Path, manifest_path: Path) -> RealDataBundle:
    """Load and validate hash-bound upstream artifacts without test outcomes."""

    root = root.resolve()
    manifest = _load_json(manifest_path)
    if manifest.get("manifest_version") != MANIFEST_VERSION:
        raise RealDataContractError("real-data input manifest version mismatch")
    inventory = _inventory(manifest, root)
    artifacts = manifest["artifacts"]
    paths = {name: _artifact_path(root, artifacts[name], verify=True) for name in REQUIRED_ARTIFACTS}
    verify_split_artifacts(paths["split"], paths["split_metadata"])
    sidecar = _load_json(paths["schema_sidecar"])
    feature_data = _load_json(paths["feature_schema"])
    feature_dictionary = _load_json(paths["feature_dictionary"])
    label_spec = _load_json(paths["label_spec"])
    sofa = _load_json(paths["sofa_provenance"])
    tensor = _load_json(paths["tensor_contract"])
    if (
        sidecar.get("schema_version") != "processed_schema_v1"
        or feature_data.get("feature_schema_version") != "feature_schema_v1"
        or label_spec.get("label_spec_version") != "label_spec_v1"
        or tensor.get("contract_version") != TENSOR_CONTRACT_VERSION
    ):
        raise RealDataContractError("BLOCKED — UPSTREAM CONTRACT VIOLATION: version mismatch")
    field_map = sidecar.get("field_map")
    if (
        not isinstance(field_map, Mapping)
        or any(concept not in field_map for concept in REQUIRED_FIELD_CONCEPTS)
        or any(
            not isinstance(field_map[concept], str) or not field_map[concept]
            for concept in REQUIRED_FIELD_CONCEPTS
        )
        or len({field_map[concept] for concept in REQUIRED_FIELD_CONCEPTS})
        != len(REQUIRED_FIELD_CONCEPTS)
    ):
        raise RealDataContractError("BLOCKED — INFORMATION REQUIRED: processed field mapping")
    features = feature_data.get("dynamic_feature_names")
    statics = feature_data.get("static_feature_names")
    sentinel = feature_data.get("tslo_no_observation_value")
    if not isinstance(features, list) or not features or len(features) != len(set(features)):
        raise RealDataContractError("feature schema must define unique ordered dynamic features")
    if not isinstance(statics, list) or len(statics) != len(set(statics)):
        raise RealDataContractError("feature schema must define ordered static features")
    if (
        feature_dictionary.get("dictionary_version")
        != "feature_provenance_v1"
        or feature_dictionary.get("all_model_fields_documented") is not True
        or feature_dictionary.get("features") != features + statics
    ):
        raise RealDataContractError(
            "feature provenance dictionary does not match model inputs"
        )
    if not _finite(sentinel):
        raise RealDataContractError("BLOCKED — TSLO CONTRACT MISMATCH")
    if (
        sofa.get("same_function_all_cutoffs") is not True
        or not isinstance(sofa.get("mimic_code_commit"), str)
        or len(sofa["mimic_code_commit"]) not in (40, 64)
    ):
        raise RealDataContractError("BLOCKED — SOFA PROVENANCE REQUIRED")
    leakage = sidecar.get("leakage_evidence")
    required_leakage = (
        "no_post_outtime_events",
        "no_post_cutoff_features",
        "no_prelookback_forward_fill",
        "no_future_duration_or_discharge_features",
    )
    if not isinstance(leakage, Mapping) or any(leakage.get(key) is not True for key in required_leakage):
        raise RealDataContractError("structural leakage evidence is incomplete")
    preprocessing = sidecar.get("preprocessing")
    if (
        not isinstance(preprocessing, Mapping)
        or preprocessing.get("fit_partition") != "train"
        or preprocessing.get("validation_or_test_statistics_used") is not False
        or preprocessing.get("status")
        not in ("NOT_FIT_PHASE19_INTERFACE_READY", "FITTED_TRAIN_ONLY")
    ):
        raise RealDataContractError("preprocessing provenance is not train-only")
    if (
        label_spec.get("recovery24_formula") != "SOFA(t+24)-SOFA(t)"
        or label_spec.get("recovery48_formula") != "SOFA(t+48)-SOFA(t)"
        or label_spec.get("icu_time_formula") != "log1p((outtime-t)_hours)"
    ):
        raise RealDataContractError(
            "label specification contradicts frozen formulas"
        )

    canonical = _load_json(paths["canonical_dataset"])
    if (
        canonical.get("storage_format") != CANONICAL_FORMAT
        or canonical.get("dataset_version") != "feature_dataset_v1"
        or canonical.get("schema_version") != "processed_schema_v1"
        or canonical.get("feature_schema_version") != "feature_schema_v1"
        or canonical.get("label_spec_version") != "label_spec_v1"
        or canonical.get("tensor_contract_version") != TENSOR_CONTRACT_VERSION
        or canonical.get("timestamp_spec_version") != TIMESTAMP_SPEC_VERSION
    ):
        raise RealDataContractError("canonical dataset format/version mismatch")
    if canonical.get("dynamic_feature_names") != features:
        raise RealDataContractError("dynamic feature order mismatch")
    if canonical.get("static_feature_names") != statics:
        raise RealDataContractError("static feature order mismatch")
    dtypes = canonical.get("dtypes")
    if dtypes != sidecar.get("dtypes") or not isinstance(dtypes, Mapping):
        raise RealDataContractError("declared dtypes do not match processed schema")
    expected_dtypes = {
        "history_values": "float32",
        "observation_mask": "bool",
        "padding_mask": "bool",
        "tslo_hours": "float32",
        "static_features": "float32",
        "eligibility": "bool",
        "timestamps": "iso8601",
    }
    if any(dtypes.get(key) != value for key, value in expected_dtypes.items()):
        raise RealDataContractError("canonical dtype declaration violates tensor contract")

    structural = _structural_rows(_load_json(paths["structural_index"]), field_map)
    validate_timestamp_parity(structural)
    _validate_split(structural, paths["split"])
    structural_by_key = {row.key: row for row in structural}
    rows = canonical.get("rows")
    if not isinstance(rows, list):
        raise RealDataContractError("canonical dataset rows must be a list")
    if any(_field(row, field_map, "split") == "test" for row in rows if isinstance(row, Mapping)):
        raise RealDataContractError("test labels/canonical outcome rows cannot be opened in Phase 18")

    support_record = artifacts.get("support_handoff")
    support = None
    support_status = "BLOCKED — PULKIT REAL ORGAN-SUPPORT LABEL ARTIFACT REQUIRED"
    if isinstance(support_record, Mapping):
        support_path = _artifact_path(root, support_record, verify=True)
        support = _support_by_key(_load_json(support_path), field_map)
        support_status = "PASS"

    schema = FeatureSchemaReference(
        version="feature_schema_v1",
        feature_names=tuple(features),
        status="FROZEN_UPSTREAM_REAL_CONTRACT",
        tslo_no_observation_value=float(sentinel),
        tslo_no_observation_source=str(artifacts["feature_schema"]["ref"]),
        static_feature_names=tuple(statics),
    )
    examples = []
    seen = set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise RealDataContractError("canonical row must be an object")
        allowed_extra = sidecar.get("allowed_non_model_fields", [])
        if not isinstance(allowed_extra, list) or any(
            not isinstance(value, str) for value in allowed_extra
        ):
            raise RealDataContractError("allowed_non_model_fields is malformed")
        allowed_fields = set(field_map.values()) | set(allowed_extra)
        if set(row) - allowed_fields:
            raise RealDataContractError(
                "canonical row contains undocumented fields"
            )
        prediction = _parse_time(_field(row, field_map, "prediction_time"), "prediction_time")
        stay = _field(row, field_map, "stay_id")
        key = (type(stay).__name__, str(stay), prediction.isoformat())
        if key in seen:
            raise RealDataContractError("BLOCKED — CANONICAL ROW IDENTITY VIOLATION")
        seen.add(key)
        structural_row = structural_by_key.get(key)
        if structural_row is None:
            raise RealDataContractError("canonical row absent from structural index")
        if (
            _field(row, field_map, "subject_id") != structural_row.subject_id
            or _field(row, field_map, "grid_index") != structural_row.grid_index
            or _field(row, field_map, "anchor_year_group")
            != structural_row.anchor_year_group
            or _field(row, field_map, "split") != structural_row.split
        ):
            raise RealDataContractError(
                "canonical identifiers conflict with structural index"
            )
        sofa_t = _field(row, field_map, "sofa_t")
        sofa24 = _field(row, field_map, "sofa_t24")
        sofa48 = _field(row, field_map, "sofa_t48")
        delta24 = _field(row, field_map, "recovery24_delta_sofa")
        delta48 = _field(row, field_map, "recovery48_delta_sofa")
        component_flags = _field(
            row, field_map, "sofa_component_observed_flags"
        )
        if (
            not isinstance(component_flags, list)
            or len(component_flags) != 6
            or any(not isinstance(value, bool) for value in component_flags)
        ):
            raise RealDataContractError(
                "SOFA six-component observed flags are missing or malformed"
            )
        rec24 = _field(row, field_map, "recovery24_eligible")
        rec48 = _field(row, field_map, "recovery48_eligible")
        icu_eligible = _field(row, field_map, "icu_time_eligible")
        if not all(isinstance(value, bool) for value in (rec24, rec48, icu_eligible)):
            raise RealDataContractError("task eligibility must be explicit booleans")
        if not _finite(sofa_t):
            raise RealDataContractError("baseline SOFA must be finite")
        if not rec24 and (sofa24 is not None or delta24 is not None):
            raise RealDataContractError(
                "ineligible recovery24 target must be null"
            )
        if not rec48 and (sofa48 is not None or delta48 is not None):
            raise RealDataContractError(
                "ineligible recovery48 target must be null"
            )
        if rec24 and prediction + timedelta(hours=24) > structural_row.outtime:
            raise RealDataContractError("recovery 24h eligibility exceeds ICU follow-up")
        if rec48 and prediction + timedelta(hours=48) > structural_row.outtime:
            raise RealDataContractError("recovery 48h eligibility exceeds ICU follow-up")
        if icu_eligible is not True:
            raise RealDataContractError("ICU-time must be eligible for every canonical row")
        if rec24 and (not all(_finite(value) for value in (sofa_t, sofa24, delta24)) or float(delta24) != float(sofa24) - float(sofa_t)):
            raise RealDataContractError("recovery 24h formula mismatch")
        if rec48 and (not all(_finite(value) for value in (sofa_t, sofa48, delta48)) or float(delta48) != float(sofa48) - float(sofa_t)):
            raise RealDataContractError("recovery 48h must remain baseline-relative")
        remaining = _field(row, field_map, "icu_remaining_hours")
        log_remaining = _field(row, field_map, "icu_time_log1p_hours")
        expected_hours = (structural_row.outtime - prediction).total_seconds() / 3600.0
        if not _finite(remaining) or float(remaining) != expected_hours or not _finite(log_remaining) or not math.isclose(float(log_remaining), math.log1p(expected_hours), abs_tol=1e-9):
            raise RealDataContractError("ICU remaining-time formula mismatch")
        support_eligible, support_target = False, None
        if support is not None:
            support_row = support.get(key)
            if support_row is None:
                raise RealDataContractError("support handoff missing canonical development row")
            support_eligible, support_target = validate_support_evidence(
                support_row, prediction, structural_row.outtime
            )
        example = CanonicalExample(
            subject_id=_field(row, field_map, "subject_id"),
            stay_id=stay,
            prediction_time=prediction.isoformat(),
            grid_index=_field(row, field_map, "grid_index"),
            icu_elapsed_hours=24 + 6 * _field(row, field_map, "grid_index"),
            anchor_year_group=_field(row, field_map, "anchor_year_group"),
            split=_field(row, field_map, "split"),
            tensor_contract_version=TENSOR_CONTRACT_VERSION,
            timestamp_spec_version=TIMESTAMP_SPEC_VERSION,
            feature_schema_version="feature_schema_v1",
            history_dtype=HISTORY_DTYPE,
            history_values=tuple(tuple(value for value in values) for values in _field(row, field_map, "history_values")),
            padding_mask_dtype=MASK_DTYPE,
            padding_mask=tuple(_field(row, field_map, "padding_mask")),
            observation_mask_dtype=MASK_DTYPE,
            observation_mask=tuple(tuple(value for value in values) for values in _field(row, field_map, "observation_mask")),
            tslo_hours=tuple(tuple(value for value in values) for values in _field(row, field_map, "tslo_hours")),
            tslo_status="FROZEN_UPSTREAM_REAL_CONTRACT",
            static_features=tuple(_field(row, field_map, "static_features")),
            static_feature_names=tuple(statics),
            static_status="FROZEN_UPSTREAM_REAL_CONTRACT",
            eligibility=TaskEligibility(rec24, rec48, icu_eligible, support_eligible),
            targets=TaskTargets(delta24 if rec24 else None, delta48 if rec48 else None, log_remaining if icu_eligible else None, support_target),
            case_tags=(),
        )
        try:
            validate_canonical_example(example, schema)
        except TensorContractError as error:
            raise RealDataContractError(str(error)) from error
        _validate_tslo(example, float(sentinel))
        valid_ranges = feature_data.get("valid_ranges", {})
        if not isinstance(valid_ranges, Mapping):
            raise RealDataContractError("feature valid_ranges must be an object")
        for feature_index, feature_name in enumerate(features):
            bounds = valid_ranges.get(feature_name)
            if bounds is None:
                continue
            if (
                not isinstance(bounds, list)
                or len(bounds) != 2
                or not all(_finite(value) for value in bounds)
            ):
                raise RealDataContractError("frozen feature range is malformed")
            low, high = map(float, bounds)
            for bin_index, values in enumerate(example.history_values):
                value = values[feature_index]
                if value is not None and not low <= float(value) <= high:
                    raise RealDataContractError(
                        "canonical value violates frozen feature range"
                    )
        examples.append(example)
    non_test_keys = {row.key for row in structural if row.split != "test"}
    if seen != non_test_keys:
        raise RealDataContractError("development canonical rows do not match structural index")
    if support is not None and set(support) != seen:
        raise RealDataContractError(
            "support handoff keys do not match canonical development rows"
        )
    split_digest = sha256_file(paths["split"])
    if sidecar.get("split_sha256") != split_digest:
        raise RealDataContractError("split hash mismatch in processed schema")
    versions = {
        "dataset": "feature_dataset_v1",
        "schema": "processed_schema_v1",
        "feature_schema": "feature_schema_v1",
        "label_spec": "label_spec_v1",
        "tensor_contract": TENSOR_CONTRACT_VERSION,
        "timestamp_spec": TIMESTAMP_SPEC_VERSION,
        "split": "split_v1",
        "split_sha256": split_digest,
        "mimic_code_commit": str(sofa["mimic_code_commit"]),
    }
    from experiments.search_governance import canonical_sha256

    canonical_semantics = [
        {
            "key": [type(item.stay_id).__name__, str(item.stay_id), item.prediction_time],
            "history": item.history_values,
            "mask": item.observation_mask,
            "padding": item.padding_mask,
            "tslo": item.tslo_hours,
            "static": item.static_features,
            "eligibility": item.eligibility.__dict__,
            "targets": item.targets.__dict__,
        }
        for item in sorted(examples, key=lambda value: (str(value.stay_id), value.prediction_time))
    ]
    structural_semantics = [
        {
            "subject_id": row.subject_id,
            "stay_id": row.stay_id,
            "prediction_time": row.prediction_time.isoformat(),
            "grid_index": row.grid_index,
            "intime": row.intime.isoformat(),
            "outtime": row.outtime.isoformat(),
            "anchor_year_group": row.anchor_year_group,
            "split": row.split,
        }
        for row in structural
    ]
    return RealDataBundle(
        examples=tuple(sorted(examples, key=lambda item: (str(item.stay_id), item.prediction_time))),
        structural_rows=structural,
        feature_schema=schema,
        dynamic_feature_names=tuple(features),
        static_feature_names=tuple(statics),
        support_status=support_status,
        inventory=inventory,
        versions=versions,
        preprocessing=preprocessing,
        leakage_evidence=leakage,
        canonical_content_sha256=canonical_sha256(canonical_semantics),
        structural_content_sha256=canonical_sha256(structural_semantics),
    )


def assert_information_parity(bundle: RealDataBundle) -> None:
    """Prove XGBoost is only a row-major flattening of each GRU view."""

    for example in bundle.examples:
        views = temporal_information_views(example)
        gru = views["gru"]
        flat = views["xgboost"]
        expected_history = tuple(value for row in gru["history_values"] for value in row)
        expected_mask = tuple(value for row in gru["observation_mask"] for value in row)
        expected_tslo = tuple(value for row in gru["tslo_hours"] for value in row)
        if flat["history_values"] != expected_history or flat["observation_mask"] != expected_mask or flat["tslo_hours"] != expected_tslo:
            raise RealDataContractError("XGBoost/GRU canonical information mismatch")


def loader_projection(
    example: CanonicalExample, dynamic_feature_names: Sequence[str]
) -> Mapping[str, object]:
    """Deterministic lossless receiver view used for loader-parity checks."""

    return {
        "identifiers": (example.subject_id, example.stay_id, example.prediction_time, example.grid_index),
        "history_values": example.history_values,
        "dynamic_feature_names": tuple(dynamic_feature_names),
        "static_feature_names": tuple(example.static_feature_names or ()),
        "observation_mask": example.observation_mask,
        "padding_mask": example.padding_mask,
        "tslo_hours": example.tslo_hours,
        "static_features": example.static_features,
        "eligibility": example.eligibility,
        "targets": example.targets,
    }
