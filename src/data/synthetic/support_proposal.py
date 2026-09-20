"""Validation for the non-authorizing Phase-9 support-science proposal."""
from __future__ import annotations

import json, math
from pathlib import Path
from typing import Any, Mapping
import yaml


class SupportProposalError(ValueError):
    pass


def _walk_no_hidden(value: Any, path: str = "root") -> None:
    if value is None:
        raise SupportProposalError(path + " contains an unexplained null")
    if isinstance(value, str) and any(x in value.upper() for x in ("TODO", "TBD", "PLACEHOLDER", "UNRESOLVED", "DRAFT")):
        raise SupportProposalError(path + " contains a hidden unresolved marker")
    if isinstance(value, Mapping):
        for key, item in value.items(): _walk_no_hidden(item, path + "." + str(key))
    elif isinstance(value, list):
        for index, item in enumerate(value): _walk_no_hidden(item, path + "[" + str(index) + "]")


def validate_support_proposal(process_path: Path, dictionary_path: Path, inventory_path: Path):
    process = yaml.safe_load(process_path.read_text())
    dictionary = yaml.safe_load(dictionary_path.read_text())
    inventory = json.loads(inventory_path.read_text())
    _walk_no_hidden(process); _walk_no_hidden(dictionary)
    if process.get("status") != "PROPOSED_NOT_AUTHORIZED" or process["authorization"]["production_generation_allowed"] is not False:
        raise SupportProposalError("support process must remain non-authorizing")
    if dictionary.get("status") != "PROPOSED_NOT_AUTHORIZED" or dictionary.get("production_execution_authorized") is not False:
        raise SupportProposalError("event dictionary must remain non-authorizing")
    if dictionary.get("source_profile") != "SYNTHETIC_SCIENTIFIC_NON_MIMIC":
        raise SupportProposalError("proposed dictionary must be explicitly synthetic/non-MIMIC")
    if process["time_process"]["step_hours"] != 1 or process["time_process"]["interval_convention"] != "CLOSED_OPEN":
        raise SupportProposalError("proposal must match the one-hour closed-open Phase-2 architecture")
    respiratory = process["respiratory_vocabulary"]["categories"]
    names = [x["canonical_name"] for x in respiratory]
    if len(names) != len(set(names)) or set(names) != {"INVASIVE","NON_INVASIVE","HFNC","ORDINARY_OXYGEN"}:
        raise SupportProposalError("respiratory vocabulary is incomplete or duplicated")
    if [x["canonical_name"] for x in respiratory if x["endpoint_qualifying"]] != ["INVASIVE"]:
        raise SupportProposalError("only invasive ventilation may qualify")
    probabilities = process["respiratory_vocabulary"]["category_selection_probabilities_when_respiratory_support_starts"]
    if set(probabilities) != set(names) or not math.isclose(sum(probabilities.values()),1.0):
        raise SupportProposalError("respiratory category probabilities must be complete and sum to one")
    concepts = process["vasoactive_vocabulary"]["concepts"]
    concept_names = [x["canonical_name"] for x in concepts]
    if len(concept_names) != len(set(concept_names)) or not math.isclose(sum(x["selection_probability"] for x in concepts),1.0):
        raise SupportProposalError("vasoactive vocabulary/probabilities are invalid")
    if process["vasoactive_vocabulary"]["canonical_rate_unit"] != "ug/kg/min":
        raise SupportProposalError("SOFA-compatible canonical dose unit is required")
    for concept in concepts:
        lower, upper = concept["valid_rate_domain"]
        dist = concept["rate_distribution"]
        if not (0 < lower < dist["median"] < upper and dist["log_sigma"] > 0):
            raise SupportProposalError("vasoactive rate distribution/domain is invalid")
    for family in ("vasopressor_off_to_on","respiratory_off_to_on"):
        values=process["hazards"][family]["coefficients"].values()
        if any(isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x) for x in values):
            raise SupportProposalError("hazard coefficients must be finite numeric")
    if process["cessation"]["minimum_duration_hours"] <= 0 or process["cessation"]["maximum_untruncated_duration_hours"] <= process["cessation"]["minimum_duration_hours"]:
        raise SupportProposalError("duration bounds are invalid")
    if process["endpoint_alignment"]["unknown_state_policy"] != "FAIL_CLOSED_NOT_OFF" or process["time_process"]["onset_equal_outtime_policy"] != "INVALID_FAIL_CLOSED":
        raise SupportProposalError("unknown/tie policies must be explicit and fail closed")
    if dictionary["sofa_alignment"]["vasoactive_unit"] != process["sofa_alignment"]["dose_unit"]:
        raise SupportProposalError("dictionary/SOFA unit mismatch")
    if set(dictionary["sofa_alignment"]["vasoactive_concepts"]) != set(process["sofa_alignment"]["vasoactive_concepts"]):
        raise SupportProposalError("dictionary/SOFA vocabulary mismatch")
    if inventory.get("status") != "COMPLETE_AUDIT_FOR_FREEZE_PROPOSAL_NOT_AUTHORIZATION":
        raise SupportProposalError("unresolved inventory status mismatch")
    entries=inventory.get("parameters",[]); ids=[x.get("id") for x in entries]
    allowed={"FROZEN_FROM_EXISTING_PROJECT_AUTHORITY","PROPOSED_FOR_EXPLICIT_TEAM_APPROVAL","BLOCKED_REQUIRES_EXTERNAL/AUTHORITATIVE_EVIDENCE"}
    if len(ids)!=len(set(ids)) or not entries or any(x.get("resolution_class") not in allowed for x in entries):
        raise SupportProposalError("inventory IDs or resolution classes are invalid")
    return process, dictionary, inventory
