"""Single cutoff-aware Phase-7 synthetic feature builder."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

from data.schema import (
    CanonicalFeatureInput, FeatureSchemaReference,
    validate_canonical_feature_input,
)
from data.synthetic.provenance import sha256_file
from labels.synthetic_profile import load_synthetic_event_dictionary
from labels.support_state import ExecutionMode, NormalizedActiveInterval, query_vasopressor_state
from labels.ventilation_state import NormalizedVentilationInterval, query_invasive_ventilation_state


BUILDER_VERSION = "synthetic_feature_builder_v2"


class FeatureBuilderError(ValueError):
    pass


@dataclass(frozen=True)
class FeatureBuildContext:
    subject_id: object
    stay_id: object
    intime: datetime
    outtime: datetime
    events: Tuple[Mapping[str, object], ...]
    prediction_time: datetime
    prediction_time_text: str
    grid_index: int
    icu_elapsed_hours: int
    support_intervals: Tuple[Mapping[str, object], ...] = ()


def _dt(value: object) -> datetime:
    if isinstance(value, datetime):
        result = value
    elif isinstance(value, str):
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    else:
        raise FeatureBuilderError("timestamp must be datetime or ISO-8601 string")
    if result.utcoffset() is None:
        raise FeatureBuilderError("timestamps must be timezone-aware")
    return result.astimezone(timezone.utc)


def load_feature_schema(path: Path, root: Path) -> Tuple[Mapping[str, Any], FeatureSchemaReference]:
    path = path if path.is_absolute() else root / path
    payload = json.loads(path.read_text())
    if payload.get("schema_version") not in {"synthetic_feature_schema_v1","synthetic_feature_schema_v2"}:
        raise FeatureBuilderError("feature schema version mismatch")
    for section in ("processed_schema", "concept_map", "timestamp_spec", "tensor_contract"):
        ref = payload[section]
        if sha256_file(root / ref["path"]) != ref["sha256"]:
            raise FeatureBuilderError(section + " hash mismatch")
    if payload.get("schema_version")=="synthetic_feature_schema_v2":
        support=payload.get("support_features",{})
        for path_field,hash_field in (("support_process_path","support_process_sha256"),("event_dictionary_path","event_dictionary_sha256")):
            if sha256_file(root / support[path_field]) != support[hash_field]:
                raise FeatureBuilderError("support feature dependency hash mismatch: "+path_field)
    channels = payload.get("temporal_channels", [])
    if payload["shape"] != {
        "sequence_length": 8, "feature_dimension": len(channels),
        "lookback_hours": 48, "bin_width_hours": 6,
        "order": "OLDEST_TO_NEWEST", "interval": "(t-48h,t]",
        "bin_closure": "LEFT_OPEN_RIGHT_CLOSED",
    }:
        raise FeatureBuilderError("feature shape/time contract mismatch")
    names = tuple(item["name"] for item in channels)
    if len(names) != len(set(names)) or not names:
        raise FeatureBuilderError("temporal feature names must be nonempty and unique")
    allowed = {"LATEST", "SUM_AVAILABLE_INTERVAL_VOLUMES", "STATE_AT_BIN_END"}
    if any(item["aggregation"] not in allowed or item["carry_forward"] != "NONE" for item in channels):
        raise FeatureBuilderError("unsupported aggregation/carry-forward policy")
    static_names = tuple(payload["static_contract"]["ordered_names"])
    schema_hash = sha256_file(path)
    reference = FeatureSchemaReference(
        version=payload["schema_version"], feature_names=names,
        status=payload["status"],
        tslo_no_observation_value=float(payload["tslo"]["no_observation_sentinel"]),
        tslo_no_observation_source=(
            str(path.relative_to(root)) if path.is_relative_to(root) else str(path)
        ),
        static_feature_names=static_names, schema_sha256=schema_hash,
    )
    return payload, reference


class SyntheticCanonicalFeatureBuilder:
    """Authoritative builder shared by modeling and serving adapters."""

    def __init__(self, *, schema_path: Path, root: Path, statics_by_stay: Mapping[object, Mapping[str, object]]) -> None:
        schema_path = schema_path if schema_path.is_absolute() else root / schema_path
        self.root = root
        self.schema_path = schema_path
        self.spec, self.feature_schema = load_feature_schema(schema_path, root)
        self.schema = self.feature_schema
        self.version = str(self.spec["builder_version"])
        self.feature_schema_sha256 = self.feature_schema.schema_sha256
        self._statics = dict(statics_by_stay)
        self.event_dictionary = load_synthetic_event_dictionary(root / "configs/event_dict_v2.yaml") if self.spec["schema_version"]=="synthetic_feature_schema_v2" else None
        if set(self.spec["prohibited_fields"]) & set(self.feature_schema.static_feature_names or ()):
            raise FeatureBuilderError("static model schema contains prohibited future/outcome field")
        concept_map = json.loads((root / self.spec["concept_map"]["path"]).read_text())
        mappings = concept_map["mappings"]
        for channel in self.spec["temporal_channels"]:
            if channel["aggregation"] == "STATE_AT_BIN_END":
                continue
            mapping = mappings.get(channel["source_concept"])
            if mapping is None or mapping["canonical_unit"] != channel["unit"] or mapping["provenance_id"] != channel["provenance_id"]:
                raise FeatureBuilderError("feature schema/concept-map mismatch")

    def __call__(self, history: object) -> CanonicalFeatureInput:
        prediction_row = getattr(history, "prediction_row", None)
        if prediction_row is None:
            raise FeatureBuilderError("history lacks a verified prediction row")
        context = FeatureBuildContext(
            subject_id=getattr(history, "subject_id"), stay_id=getattr(history, "stay_id"),
            intime=_dt(getattr(history, "intime")), outtime=_dt(getattr(history, "outtime")),
            events=tuple(getattr(history, "events")),
            support_intervals=tuple(getattr(history, "support_intervals", ())),
            prediction_time=_dt(prediction_row.prediction_time),
            prediction_time_text=prediction_row.prediction_time.isoformat(),
            grid_index=prediction_row.grid_index,
            icu_elapsed_hours=prediction_row.icu_elapsed_hours,
        )
        return self.build(context)

    def build(self, context: FeatureBuildContext) -> CanonicalFeatureInput:
        t = _dt(context.prediction_time)
        if not context.intime <= t < context.outtime:
            raise FeatureBuilderError("prediction cutoff is outside episode")
        if context.icu_elapsed_hours != 24 + 6 * context.grid_index:
            raise FeatureBuilderError("cutoff is not a legal structural identity")
        if t != _dt(context.intime) + timedelta(hours=context.icu_elapsed_hours):
            raise FeatureBuilderError("prediction cutoff does not match episode-relative grid identity")
        start = t - timedelta(hours=48)
        channels = tuple(self.spec["temporal_channels"])
        raw_channels=tuple(item for item in channels if item["aggregation"]!="STATE_AT_BIN_END")
        by_concept: Dict[str, list] = {item["source_concept"]: [] for item in raw_channels}
        seen_ids = set()
        for row in context.events:
            if row.get("stay_id") != context.stay_id or row.get("subject_id") != context.subject_id:
                continue
            when = _dt(row.get("event_time"))
            if when <= start or when > t or when <= context.intime:
                continue
            concept = row.get("canonical_concept")
            if concept not in by_concept:
                raise FeatureBuilderError("unsupported canonical concept in Phase-7 history")
            event_id = row.get("event_id")
            if not isinstance(event_id, str) or not event_id or event_id in seen_ids:
                raise FeatureBuilderError("event identity is missing or duplicated")
            seen_ids.add(event_id)
            channel = next(item for item in raw_channels if item["source_concept"] == concept)
            if row.get("unit") != channel["unit"] or row.get("provenance_id") != channel["provenance_id"]:
                raise FeatureBuilderError("canonical unit/provenance mismatch")
            value = row.get("value_numeric")
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise FeatureBuilderError("feature value must be finite numeric")
            by_concept[str(concept)].append((when, event_id, float(value), row))
        for events in by_concept.values():
            events.sort(key=lambda item: (item[0], item[1]))

        values, masks, tslos, padding = [], [], [], []
        sentinel = float(self.spec["tslo"]["no_observation_sentinel"])
        for bin_index in range(8):
            left = start + timedelta(hours=6 * bin_index)
            end = left + timedelta(hours=6)
            padded = end <= context.intime
            padding.append(padded)
            row_values, row_masks, row_tslo = [], [], []
            for channel in channels:
                if channel["aggregation"] == "STATE_AT_BIN_END":
                    if padded:
                        row_values.append(None); row_masks.append(False); row_tslo.append(sentinel); continue
                    value=self._support_value(context.support_intervals,context.stay_id,end,channel["name"])
                    row_values.append(value); row_masks.append(True); row_tslo.append(0.0); continue
                events = by_concept[channel["source_concept"]]
                candidates = [item for item in events if left < item[0] <= end]
                if padded:
                    row_values.append(None); row_masks.append(False); row_tslo.append(sentinel)
                    continue
                if not candidates:
                    row_values.append(None); row_masks.append(False)
                elif channel["aggregation"] == "LATEST":
                    row_values.append(candidates[-1][2]); row_masks.append(True)
                else:
                    row_values.append(sum(item[2] for item in candidates)); row_masks.append(True)
                previous = [item[0] for item in events if item[0] <= end]
                row_tslo.append(sentinel if not previous else (end - previous[-1]).total_seconds() / 3600.0)
            values.append(tuple(row_values)); masks.append(tuple(row_masks)); tslos.append(tuple(row_tslo))

        if any(padding[index] and not all(padding[:index + 1]) for index in range(8)):
            raise FeatureBuilderError("padding must be a prefix")
        statics = self._statics.get(context.stay_id)
        if statics is None or statics.get("subject_id") != context.subject_id:
            raise FeatureBuilderError("canonical static row missing or mismatched")
        static_names = self.feature_schema.static_feature_names or ()
        static_values = tuple(statics.get(name) for name in static_names)
        observed = sum(sum(row) for row in masks)
        result = CanonicalFeatureInput(
            subject_id=context.subject_id, stay_id=context.stay_id,
            prediction_time=context.prediction_time_text,
            grid_index=context.grid_index, icu_elapsed_hours=context.icu_elapsed_hours,
            tensor_contract_version="tensor_contract_v1",
            timestamp_spec_version="timestamp_spec_v1",
            feature_schema_version=self.feature_schema.version,
            feature_schema_sha256=str(self.feature_schema.schema_sha256),
            temporal_feature_names=self.feature_schema.feature_names,
            history_dtype="float32", history_values=tuple(values),
            padding_mask_dtype="bool", padding_mask=tuple(padding),
            observation_mask_dtype="bool", observation_mask=tuple(masks),
            tslo_hours=tuple(tslos), tslo_status="PHASE7_TSLO_FROZEN",
            static_features=static_values, static_feature_names=static_names,
            static_status="PHASE7_RAW_STATICS_PREPROCESSOR_INPUT",
            quality_metadata={"padding_bins":sum(padding),"observed_feature_values":observed,"missing_feature_values":8*len(channels)-observed},
            case_tags=("SYNTHETIC_CANONICAL_FEATURE_INPUT_NO_LABELS_NO_SPLIT",),
        )
        validate_canonical_feature_input(result, self.feature_schema)
        return result

    def _support_value(self, rows, stay_id, cutoff, name):
        if self.event_dictionary is None:
            raise FeatureBuilderError("support channel requested without frozen dictionary")
        vaso_rows=[r for r in rows if r.get("support_type")=="VASOPRESSOR"]
        vent_rows=[r for r in rows if r.get("support_type")=="RESPIRATORY"]
        vaso=[NormalizedActiveInterval.from_mapping({"stay_id":r["stay_id"],"agent_key":r["agent_key"],"interval_start":r["interval_start"],"interval_end":r["interval_end"],"source_event_ref":r["support_event_id"],"normalization_provenance_version":self.event_dictionary.synthetic_mapping_provenance_version}) for r in vaso_rows]
        vent=[NormalizedVentilationInterval.from_mapping({"stay_id":r["stay_id"],"category":r["respiratory_category"],"interval_start":r["interval_start"],"interval_end":r["interval_end"],"source_state_ref":r["support_event_id"],"concept_version":self.event_dictionary.ventilation.synthetic_concept_version,"adapter_version":self.event_dictionary.ventilation.synthetic_adapter_version,"normalization_provenance_ref":r["normalization_provenance_ref"]}) for r in vent_rows]
        if name=="vasopressor_on":
            return float(query_vasopressor_state(vaso,stay_id=stay_id,cutoff=cutoff,event_dictionary=self.event_dictionary,execution_mode=ExecutionMode.SYNTHETIC).support_state.value=="ON")
        if name=="invasive_ventilation_on":
            return float(query_invasive_ventilation_state(vent,stay_id=stay_id,cutoff=cutoff,event_dictionary=self.event_dictionary,execution_mode=ExecutionMode.SYNTHETIC).invasive_on)
        agent=name.removesuffix("_rate")
        active=[r for r in vaso_rows if r["agent_key"]==agent and _dt(r["interval_start"])<=cutoff<_dt(r["interval_end"])]
        if len(active)>1: raise FeatureBuilderError("conflicting same-agent active rate segments")
        return 0.0 if not active else float(active[0]["rate_value"])
