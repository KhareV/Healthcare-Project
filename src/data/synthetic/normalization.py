"""Strict raw-to-canonical event normalization without ML preprocessing."""
from __future__ import annotations
import math
from typing import Mapping, Sequence
from .validation import parse_utc

class SyntheticTimelineError(ValueError): pass

def normalize_events(raw_events: Sequence[Mapping], retained_cohort: Sequence[Mapping], concept_map: Mapping, raw_schema_version: str):
    if concept_map.get("input_stage")!="RAW_PHASE3_SYNTHETIC_EVENTS": raise SyntheticTimelineError("input stage mismatch; canonical data cannot be normalized as raw")
    mappings=concept_map["mappings"]; retained={row["stay_id"]:row for row in retained_cohort}; result=[]; filtered=0; seen=set()
    forbidden={"trajectory","trajectory_pattern","deterioration_class","outcome_class","latent_state","future_state","total_episode_duration","remaining_stay","max_future_value","final_severity","delta_sofa","support_label","split"}
    expected={"event_id","subject_id","stay_id","event_time","interval_start","interval_end","concept_code","value_numeric","value_text","unit","generator_component"}
    for raw in raw_events:
        if set(raw)!=expected or set(raw)&forbidden: raise SyntheticTimelineError("raw event has unknown, missing, or prohibited fields")
        event_id=raw["event_id"]
        if event_id in seen: raise SyntheticTimelineError("duplicate event_id")
        seen.add(event_id)
        stay=retained.get(raw["stay_id"])
        if stay is None: filtered+=1; continue
        if raw["subject_id"]!=stay["subject_id"]: raise SyntheticTimelineError("subject/stay mismatch")
        concept=raw["concept_code"]
        if concept not in mappings: raise SyntheticTimelineError(f"unknown raw concept: {concept}")
        mapping=mappings[concept]
        if raw["unit"]!=mapping["raw_unit"]: raise SyntheticTimelineError(f"unsupported unit for {concept}: {raw['unit']}")
        value=raw["value_numeric"]
        if not isinstance(value,(int,float)) or isinstance(value,bool) or not math.isfinite(float(value)): raise SyntheticTimelineError(f"invalid numeric value for {concept}")
        low,high=mapping["domain"]
        if not low<=value<=high: raise SyntheticTimelineError(f"value outside synthetic domain for {concept}")
        if mapping["value_type"]=="integer" and not isinstance(value,int): raise SyntheticTimelineError(f"integer value required for {concept}")
        start,end=parse_utc(stay["intime"]),parse_utc(stay["outtime"]); event_time=parse_utc(raw["event_time"])
        if not start<=event_time<=end: raise SyntheticTimelineError("event outside retained episode bounds")
        kind=mapping["event_kind"]
        if kind=="point" and (raw["interval_start"] is not None or raw["interval_end"] is not None): raise SyntheticTimelineError("point event has interval fields")
        if kind=="quantity_interval":
            if raw["interval_start"] is None or raw["interval_end"] is None: raise SyntheticTimelineError("quantity interval lacks boundaries")
            interval_start,interval_end=parse_utc(raw["interval_start"]),parse_utc(raw["interval_end"])
            if not start<=interval_start<interval_end==event_time: raise SyntheticTimelineError("invalid quantity interval boundaries")
        result.append({"event_id":event_id,"subject_id":raw["subject_id"],"stay_id":raw["stay_id"],"event_time":raw["event_time"],"interval_start":raw["interval_start"],"interval_end":raw["interval_end"],"canonical_concept":mapping["canonical_concept"],"value_numeric":value,"unit":mapping["canonical_unit"],"event_kind":kind,"provenance_id":mapping["provenance_id"],"source_concept":concept,"source_schema_version":raw_schema_version,"normalization_version":concept_map["concept_map_version"]})
    result.sort(key=lambda row:(row["subject_id"],row["stay_id"],row["event_time"],row["canonical_concept"],row["event_id"]))
    return tuple(result),filtered

def validate_timeline(rows: Sequence[Mapping], processed_schema: Mapping, concept_map: Mapping):
    fields={f["name"] for f in processed_schema["fields"]}; mappings=concept_map["mappings"]; provenance={m["provenance_id"] for m in mappings.values()}
    if processed_schema.get("stage")!="CANONICAL_TIMELINE": raise SyntheticTimelineError("processed schema stage mismatch")
    seen=set(); last=None
    for row in rows:
        if set(row)!=fields: raise SyntheticTimelineError("canonical timeline fields differ from processed schema")
        if row["event_id"] in seen: raise SyntheticTimelineError("duplicate event_id")
        seen.add(row["event_id"])
        if row["provenance_id"] not in provenance: raise SyntheticTimelineError("orphan provenance_id")
        mapping=mappings.get(row["source_concept"])
        if not mapping or row["canonical_concept"]!=mapping["canonical_concept"] or row["unit"]!=mapping["canonical_unit"] or row["event_kind"]!=mapping["event_kind"]: raise SyntheticTimelineError("canonical mapping/provenance mismatch")
        key=(row["subject_id"],row["stay_id"],row["event_time"],row["canonical_concept"],row["event_id"])
        if last is not None and key<last: raise SyntheticTimelineError("canonical ordering mismatch")
        last=key
    return True
