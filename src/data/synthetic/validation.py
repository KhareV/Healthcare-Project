"""Schema, role, temporal, and leakage validation for generated raw data."""

from __future__ import annotations

from datetime import datetime
import json
import math
from pathlib import Path
from typing import Mapping, Sequence


class SyntheticValidationError(ValueError):
    pass


def parse_utc(value: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise SyntheticValidationError(f"timestamp expected ISO-8601 UTC Z; observed {value!r}")
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.utcoffset() is None:
        raise SyntheticValidationError("timestamp must be timezone-aware")
    return result


def _field_names(table: Mapping[str, object]) -> set[str]:
    return {str(field["name"]) for field in table["fields"]}


def validate_field_roles(schema: Mapping[str, object]) -> None:
    roles = set(schema["field_role_vocabulary"])
    for table_name, table in schema["tables"].items():
        for field in table["fields"]:
            if field["role"] not in roles:
                raise SyntheticValidationError(f"{table_name}.{field['name']} has unknown field role")
            if field["name"] == "outtime" and (field["role"] != "STRUCTURAL_LABEL_ONLY" or field["model_eligible"]):
                raise SyntheticValidationError("episodes.outtime must be STRUCTURAL_LABEL_ONLY and not model eligible")


def validate_dataset(subjects: Sequence[Mapping], episodes: Sequence[Mapping], events: Sequence[Mapping], supports: Sequence[Mapping], schema: Mapping, inventory: Mapping, *, allow_support: bool = False):
    validate_field_roles(schema)
    forbidden = {"trajectory","trajectory_class","outcome_class","target","delta_sofa_24","delta_sofa_48","remaining_stay","future_support","support_label","organ_support_label","split","latent_state","latent_state_vector","future_state"}
    tables = {"subjects": subjects, "episodes": episodes, "raw_events": events, "support_intervals": supports}
    for name, rows in tables.items():
        allowed = _field_names(schema["tables"][name])
        for index, row in enumerate(rows):
            extra = set(row) - allowed
            bad = set(row) & forbidden
            if extra or bad:
                raise SyntheticValidationError(f"{name}[{index}] contains forbidden/unclassified fields: {sorted(extra | bad)}")
            if set(row) != allowed:
                raise SyntheticValidationError(f"{name}[{index}] fields differ from schema: missing={sorted(allowed-set(row))}")
    subject_ids = [r["subject_id"] for r in subjects]
    stay_ids = [r["stay_id"] for r in episodes]
    if len(set(subject_ids)) != len(subject_ids): raise SyntheticValidationError("duplicate subject_id")
    if len(set(stay_ids)) != len(stay_ids): raise SyntheticValidationError("duplicate stay_id")
    if len(subjects) != len(episodes): raise SyntheticValidationError("expected exactly one episode per subject")
    episode_by_stay = {}
    for row in episodes:
        if row["subject_id"] not in set(subject_ids): raise SyntheticValidationError("episode subject foreign key invalid")
        start, end = parse_utc(row["intime"]), parse_utc(row["outtime"])
        if end <= start: raise SyntheticValidationError("outtime must be strictly after intime")
        if row["episode_sequence"] != 1: raise SyntheticValidationError("episode_sequence must equal 1")
        episode_by_stay[row["stay_id"]] = (row["subject_id"], start, end)
    event_ids = set()
    last_key = None
    for row in events:
        if row["event_id"] in event_ids: raise SyntheticValidationError("duplicate event_id")
        event_ids.add(row["event_id"])
        if row["stay_id"] not in episode_by_stay: raise SyntheticValidationError("event stay foreign key invalid")
        subject_id, start, end = episode_by_stay[row["stay_id"]]
        if row["subject_id"] != subject_id: raise SyntheticValidationError("event subject/stay foreign keys disagree")
        time = parse_utc(row["event_time"])
        if not start <= time <= end: raise SyntheticValidationError("event_time outside episode bounds")
        concept = row["concept_code"]
        if concept not in inventory: raise SyntheticValidationError(f"unknown raw concept: {concept}")
        spec = inventory[concept]
        if row["unit"] != spec["unit"]: raise SyntheticValidationError(f"unit mismatch for {concept}")
        value = row["value_numeric"]
        if row["value_text"] is not None or not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
            raise SyntheticValidationError("numeric raw event must have exactly one finite numeric value")
        low, high = spec["synthetic_generation_domain"]
        if not low <= value <= high: raise SyntheticValidationError(f"value outside configured domain for {concept}")
        if spec["value_type"] == "integer" and not isinstance(value, int): raise SyntheticValidationError(f"{concept} requires integer value")
        interval = row["interval_start"], row["interval_end"]
        if spec.get("interval_start_required"):
            if None in interval or not start <= parse_utc(interval[0]) < parse_utc(interval[1]) == time: raise SyntheticValidationError("invalid interval measurement boundaries")
        elif interval != (None, None): raise SyntheticValidationError("point event must have null interval boundaries")
        key = (row["stay_id"], row["event_time"], row["event_id"])
        if last_key is not None and key < last_key: raise SyntheticValidationError("raw events are not canonically ordered")
        last_key = key
    if supports and not allow_support: raise SyntheticValidationError("non-final Phase 3 support_intervals must remain empty")
    seen_support_ids=set(); last_support=None
    allowed_agents={"norepinephrine","epinephrine","dopamine","dobutamine"}
    allowed_resp={"INVASIVE","NON_INVASIVE","HFNC","ORDINARY_OXYGEN"}
    for row in supports:
        event_id=row["support_event_id"]
        if event_id in seen_support_ids: raise SyntheticValidationError("duplicate support_event_id")
        seen_support_ids.add(event_id)
        if row["stay_id"] not in episode_by_stay: raise SyntheticValidationError("support stay foreign key invalid")
        subject_id,start,end=episode_by_stay[row["stay_id"]]
        left,right=parse_utc(row["interval_start"]),parse_utc(row["interval_end"])
        if row["subject_id"]!=subject_id or not start<=left<right<=end: raise SyntheticValidationError("support interval identity/bounds invalid")
        if row["support_type"]=="VASOPRESSOR":
            if row["agent_key"] not in allowed_agents or row["support_category"]!=row["agent_key"] or row["respiratory_category"] is not None: raise SyntheticValidationError("invalid vasoactive vocabulary")
            if row["rate_unit"]!="ug/kg/min" or not isinstance(row["rate_value"],(int,float)) or row["rate_value"]<=0: raise SyntheticValidationError("invalid vasoactive rate/unit")
        elif row["support_type"]=="RESPIRATORY":
            if row["respiratory_category"] not in allowed_resp or row["support_category"]!=row["respiratory_category"] or row["agent_key"] is not None or row["rate_value"] is not None or row["rate_unit"] is not None: raise SyntheticValidationError("invalid respiratory vocabulary")
        else: raise SyntheticValidationError("unknown support type")
        key=(row["stay_id"],row["interval_start"],row["support_event_id"])
        if last_support is not None and key<last_support: raise SyntheticValidationError("support rows not canonically ordered")
        last_support=key
    return True


def load_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
