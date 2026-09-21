"""Adapter from the frozen synthetic v2 dictionary to Pulkit's state contracts."""
from pathlib import Path
import yaml

from labels.event_dict import EventDictionary, OrganSupportLabelContract, VentilationEventContract
from vedant_infra.hashing import sha256_file


def load_synthetic_event_dictionary(path: Path) -> EventDictionary:
    p=yaml.safe_load(path.read_text())
    if p.get("schema_version")!="event_dict_schema_v2" or p.get("status")!="FROZEN_SYNTHETIC_AUTHORIZED" or p.get("production_execution_authorized") is not True:
        raise ValueError("synthetic event dictionary is not frozen/authorized")
    vaso=p["vasopressor"]; vent=p["ventilation"]; endpoint=p["endpoint"]
    qualifying=frozenset(k for k,v in vaso["concepts"].items() if v["endpoint_qualifying"])
    ventilation=VentilationEventContract(
        status="FROZEN_SYNTHETIC_AUTHORIZED", real_scientific_execution_authorized=False,
        state_rule_version=vent["state_rule_version"], normalized_contract_version=vent["normalized_contract_version"],
        scientific_concept_name=None, scientific_concept_version=None, scientific_adapter_version=None, scientific_boundary_policy=None,
        sofa_alignment_status=p["sofa_alignment"]["status"], synthetic_concept_name=vent["concept_name"],
        synthetic_concept_version=vent["concept_version"], synthetic_adapter_version=vent["adapter_version"],
        synthetic_known_categories=frozenset(vent["known_categories"]), synthetic_invasive_categories=frozenset(vent["invasive_categories"]),
        synthetic_non_invasive_categories=frozenset(vent["explicitly_noninvasive_categories"]), synthetic_boundary_policy=p["intervals"]["convention"])
    organ=OrganSupportLabelContract(
        status="FROZEN_ENDPOINT", real_scientific_execution_authorized=False,
        label_rule_version="organ_support_label_rule_v1", onset_evidence_contract_version="component_off_to_on_evidence_v1",
        phase7_handoff_record_contract="SupportLabelRecord_v1_existing_phase7_interface", horizon_hours=int(endpoint["horizon_hours"]),
        future_interval_semantics=endpoint["future_interval"], onset_outtime_tie_status=endpoint["onset_equal_outtime"],
        unknown_component_state_policy_status=endpoint["unknown_state"], audit_reason_vocabulary_status="FROZEN_SYNTHETIC_AUTHORIZED",
        audit_reason_vocabulary_version=endpoint["audit_vocabulary_version"])
    return EventDictionary(
        schema_version=p["schema_version"],event_dictionary_version=p["event_dictionary_version"],status=p["status"],
        real_scientific_execution_authorized=False,state_rule_version=vaso["state_rule_version"],normalized_contract_version=vaso["normalized_contract_version"],
        scientific_qualifying_agent_keys=frozenset(),scientific_boundary_policy=None,mapping_provenance_version=None,
        synthetic_qualifying_agent_keys=qualifying,synthetic_boundary_policy=p["intervals"]["convention"],
        synthetic_mapping_provenance_version=vaso["mapping_provenance_version"],ventilation=ventilation,organ_support=organ,
        config_sha256=sha256_file(path))
