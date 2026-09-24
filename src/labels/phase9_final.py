"""Build the final pre-split Phase-9 scientific package."""
from __future__ import annotations
from collections import Counter
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import json, math, os, shutil, tempfile
from pathlib import Path

from data.synthetic.config import canonical_json_bytes
from data.synthetic.provenance import semantic_sha256, sha256_file
from data.synthetic.sofa import VasoactiveExposure, load_sofa_spec, sofa_at
from data.synthetic.sofa_provider import PulkitStateSOFASupportProvider, bindings_from_processed_manifest
from data.synthetic.validation import load_jsonl
from labels.icu_time import remaining_episode_time_label
from labels.organ_support import ComponentOnsetEvidence, SupportComponent, SupportLabelCutoff, TransitionKind, generate_organ_support_label
from labels.recovery import build_recovery_label
from labels.support_state import ExecutionMode, NormalizedActiveInterval, query_vasopressor_state
from labels.synthetic_profile import load_synthetic_event_dictionary
from labels.ventilation_state import NormalizedVentilationInterval, query_invasive_ventilation_state
from data.timestamps import TIMESTAMP_SPEC_VERSION

VERSION="synthetic_phase9_pre_split_package_v1"

def _dt(x): return datetime.fromisoformat(x.replace("Z","+00:00")).astimezone(timezone.utc)
def _artifact(manifest,name,root):
    a=next(x for x in manifest["artifacts"] if x["logical_name"]==name)
    return root/a.get("repository_relative_path",a.get("path"))

def _normalized(rows,dictionary):
    vaso=[]; vent=[]; exposures=[]
    for r in rows:
        if r["support_type"]=="VASOPRESSOR":
            vaso.append(NormalizedActiveInterval(r["stay_id"],r["agent_key"],_dt(r["interval_start"]),_dt(r["interval_end"]),r["support_event_id"],dictionary.synthetic_mapping_provenance_version))
            exposures.append(VasoactiveExposure(r["stay_id"],r["agent_key"],float(r["rate_value"]),r["rate_unit"],_dt(r["interval_start"]),_dt(r["interval_end"]),r["support_event_id"]))
        else:
            from labels.ventilation_state import RespiratoryCategory
            vent.append(NormalizedVentilationInterval(r["stay_id"],RespiratoryCategory(r["respiratory_category"]),_dt(r["interval_start"]),_dt(r["interval_end"]),r["support_event_id"],dictionary.ventilation.synthetic_concept_version,dictionary.ventilation.synthetic_adapter_version,r["normalization_provenance_ref"]))
    return tuple(vaso),tuple(vent),tuple(exposures)

def _onsets(rows,dictionary):
    """Derive component OFF->ON transitions from closed-open segment unions."""
    result=[]
    for component, filtered in (
        (SupportComponent.VASOPRESSOR,[r for r in rows if r["support_type"]=="VASOPRESSOR" and r["agent_key"] in dictionary.synthetic_qualifying_agent_keys]),
        (SupportComponent.INVASIVE_VENTILATION,[r for r in rows if r["support_type"]=="RESPIRATORY" and r["respiratory_category"]=="INVASIVE"])):
        boundaries=sorted({_dt(r["interval_start"]) for r in filtered})
        for start in boundaries:
            active_before=any(_dt(r["interval_start"])<start<=_dt(r["interval_end"]) for r in filtered)
            active_at=any(_dt(r["interval_start"])<=start<_dt(r["interval_end"]) for r in filtered)
            if active_at and not active_before:
                source=min(r["support_event_id"] for r in filtered if _dt(r["interval_start"])==start)
                if component is SupportComponent.VASOPRESSOR:
                    rule=dictionary.state_rule_version; definition=dictionary.synthetic_mapping_provenance_version; adapter=definition
                else:
                    rule=dictionary.ventilation.state_rule_version; definition=dictionary.ventilation.synthetic_concept_version; adapter=dictionary.ventilation.synthetic_adapter_version
                result.append(ComponentOnsetEvidence(rows[0]["stay_id"],component,start,TransitionKind.OFF_TO_ON,source,"synthetic_support_adapter_v2",rule,definition,adapter,dictionary.event_dictionary_version,dictionary.config_sha256,"component_off_to_on_evidence_v1",ExecutionMode.SYNTHETIC))
    return tuple(result)

def build(*, cohort_manifest_path:Path, processed_manifest_path:Path, feature_manifest_path:Path, output:Path, root:Path, generation_timestamp:str="2026-09-21T00:00:00Z"):
    cohort=json.loads(cohort_manifest_path.read_text()); processed=json.loads(processed_manifest_path.read_text()); feature=json.loads(feature_manifest_path.read_text())
    structural=load_jsonl(_artifact(cohort,"structural_index",root)); retained=load_jsonl(_artifact(cohort,"retained_cohort",root)); timeline=load_jsonl(_artifact(processed,"canonical_timeline",root)); features=load_jsonl(_artifact(feature,"canonical_feature_inputs",root))
    phase3=json.loads((root/processed["phase3_manifest_path"]).read_text()); supports=load_jsonl(_artifact(phase3,"support_intervals",root))
    generation_seed=phase3["primary_seed_identity"]["value"]; requested_subject_count=phase3["requested_subject_count"]
    dictionary=load_synthetic_event_dictionary(root/"configs/event_dict_v2.yaml"); sofa_spec=load_sofa_spec(root/"configs/synthetic/sofa_spec_v1.json")
    support_hash=sha256_file(root/"configs/synthetic/support_process_v1.yaml")
    base_bindings=bindings_from_processed_manifest(processed_manifest_path,root,support_contract_version="placeholder",support_contract_sha256="0"*64)
    stay_by={r["stay_id"]:r for r in retained}; events_by={k:[] for k in stay_by}; support_by={k:[] for k in stay_by}
    for r in timeline: events_by[r["stay_id"]].append(r)
    for r in supports:
        if r["stay_id"] in support_by: support_by[r["stay_id"]].append(r)
    feature_by={(r["stay_id"],r["prediction_time"]):r for r in features}
    rows=[]; sofa_cache={}; dispositions=Counter(); baseline_states=Counter(); onset_counts=Counter()
    for srow in structural:
        stay=stay_by[srow["stay_id"]]; s_support=support_by[srow["stay_id"]]; vaso,vent,exposures=_normalized(s_support,dictionary)
        provider=PulkitStateSOFASupportProvider(event_dictionary=dictionary,execution_mode=ExecutionMode.SYNTHETIC,ventilation_intervals=vent,vasoactive_exposures=exposures,vasoactive_dose_contract_version="synthetic_support_process_v1",vasoactive_dose_contract_sha256=support_hash,vasoactive_coverage_known=True)
        bindings=replace(base_bindings,support_contract_version=provider.contract_version,support_contract_sha256=provider.contract_sha256)
        def score(when):
            key=(stay["stay_id"],when.isoformat())
            if key not in sofa_cache: sofa_cache[key]=sofa_at(events_by[stay["stay_id"]],stay,when,spec=sofa_spec,bindings=bindings,support_provider=provider)
            return sofa_cache[key]
        t=_dt(srow["prediction_time"]); out=_dt(stay["outtime"])
        recovery=build_recovery_label(stay_id=stay["stay_id"],prediction_time=t,outtime=out,recovery24_followup_available=srow["recovery24_followup_available"],recovery48_followup_available=srow["recovery48_followup_available"],score_at=score)
        icu=remaining_episode_time_label(prediction_time=t,outtime=out,icu_time_temporally_eligible=srow["icu_time_temporally_eligible"])
        vs=query_vasopressor_state(vaso,stay_id=stay["stay_id"],cutoff=t,event_dictionary=dictionary,execution_mode=ExecutionMode.SYNTHETIC)
        vents=query_invasive_ventilation_state(vent,stay_id=stay["stay_id"],cutoff=t,event_dictionary=dictionary,execution_mode=ExecutionMode.SYNTHETIC)
        cutoff=SupportLabelCutoff(stay["stay_id"],t,out,srow["support24_full_followup_available"],TIMESTAMP_SPEC_VERSION)
        label=generate_organ_support_label(cutoff=cutoff,vasopressor_state=vs,ventilation_state=vents,onset_evidence=_onsets(s_support,dictionary) if s_support else (),event_dictionary=dictionary,execution_mode=ExecutionMode.SYNTHETIC)
        dispositions[label.disposition.value]+=1; baseline_states[(vs.support_state.value,vents.invasive_state.value)]+=1
        if label.vasopressor_onset_time: onset_counts["vasopressor_positive_rows"]+=1
        if label.ventilation_onset_time: onset_counts["invasive_ventilation_positive_rows"]+=1
        f=feature_by[(stay["stay_id"],srow["prediction_time"])]
        rows.append({"subject_id":srow["subject_id"],"stay_id":stay["stay_id"],"prediction_time":srow["prediction_time"],"grid_index":srow["grid_index"],
            "feature_schema_version":f["feature_schema_version"],"feature_schema_sha256":f["feature_schema_sha256"],"temporal_feature_names":f["temporal_feature_names"],"history_values":f["history_values"],"observation_mask":f["observation_mask"],"tslo_hours":f["tslo_hours"],"padding_mask":f["padding_mask"],"static_features":f["static_features"],"static_feature_names":f["static_feature_names"],
            "delta_sofa_24":recovery.delta_sofa_24,"recovery24_eligible":recovery.recovery24_eligible,"delta_sofa_48":recovery.delta_sofa_48,"recovery48_eligible":recovery.recovery48_eligible,
            "icu_time_log1p":icu.log1p_remaining_hours,"icu_time_eligible":icu.eligible,"organ_support_label":label.label,"organ_support_eligible":label.organ_support_eligible,
            "support_audit_disposition":label.disposition.value,"baseline_sofa":recovery.baseline_sofa,"sofa_spec_sha256":recovery.sofa_spec_sha256,
            "split":None,"preprocessor":None})
    output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists(): raise FileExistsError(output)
    temp=Path(tempfile.mkdtemp(prefix=".phase9-final-",dir=output.parent))
    try:
        package=temp/"pre_split_scientific_package.jsonl"
        with package.open("wb") as h:
            for r in rows: h.write(canonical_json_bytes(r))
        d24=[r["delta_sofa_24"] for r in rows if r["recovery24_eligible"]]; d48=[r["delta_sofa_48"] for r in rows if r["recovery48_eligible"]]; icu=[r["icu_time_log1p"] for r in rows]
        def summary(x): return {"count":len(x),"minimum":min(x),"mean":sum(x)/len(x),"maximum":max(x),"variance":sum((v-sum(x)/len(x))**2 for v in x)/len(x)}
        qa={"qa_version":"phase9_final_qa_v1","total_canonical_rows":len(rows),"support_eligible":sum(r["organ_support_eligible"] for r in rows),"support_positive":sum(r["organ_support_label"]==1 for r in rows),"support_negative":sum(r["organ_support_label"]==0 for r in rows),"censored":dispositions["INTERNAL_CENSORED_EARLY_EXIT_NO_INITIATION"],"not_at_risk":dispositions["INTERNAL_NOT_AT_RISK_BOTH_COMPONENTS_ON"],"baseline_states":{"both_off":baseline_states[("OFF","OFF")],"vaso_on_vent_off":baseline_states[("ON","OFF")],"vaso_off_vent_on":baseline_states[("OFF","ON")],"both_on":baseline_states[("ON","ON")]},"positive_component_rows":dict(onset_counts),"recovery24":summary(d24),"recovery48":summary(d48),"icu_time_log1p":summary(icu),"split":"NOT_CREATED_PHASE10","preprocessor":"NOT_FIT","models":"NOT_TRAINED"}
        (temp/"phase9_qa.json").write_bytes(canonical_json_bytes(qa))
        manifest={"manifest_version":VERSION,"status":"FINAL_PRE_SPLIT_SCIENTIFIC_PACKAGE","decision_authority":"USER_DELEGATED_AI_PROJECT_DECISION","row_count":len(rows),"subject_count":len({r["subject_id"] for r in rows}),"feature_dimension":len(rows[0]["temporal_feature_names"]),"seed":generation_seed,"requested_subject_count":requested_subject_count,"split":"NOT_CREATED_PHASE10","preprocessor":"NOT_FIT","model_training":"NOT_PERFORMED","phase10_plus":"NOT_IMPLEMENTED","generation_timestamp_utc":generation_timestamp,"dependencies":{"phase3_manifest":{"path":str((root/processed["phase3_manifest_path"]).relative_to(root)),"sha256":sha256_file(root/processed["phase3_manifest_path"])},"cohort_manifest":{"path":str(cohort_manifest_path.relative_to(root)),"sha256":sha256_file(cohort_manifest_path)},"processed_manifest":{"path":str(processed_manifest_path.relative_to(root)),"sha256":sha256_file(processed_manifest_path)},"feature_manifest":{"path":str(feature_manifest_path.relative_to(root)),"sha256":sha256_file(feature_manifest_path)},"support_process":{"path":"configs/synthetic/support_process_v1.yaml","sha256":support_hash},"event_dictionary":{"path":"configs/event_dict_v2.yaml","sha256":sha256_file(root/"configs/event_dict_v2.yaml")},"sofa_spec":{"path":"configs/synthetic/sofa_spec_v1.json","sha256":sha256_file(root/"configs/synthetic/sofa_spec_v1.json")}},"artifacts":[{"logical_name":"pre_split_scientific_package","path":str((output/package.name).relative_to(root)),"sha256":sha256_file(package),"semantic_sha256":semantic_sha256("pre_split_scientific_package",rows),"record_count":len(rows)},{"logical_name":"phase9_qa","path":str((output/"phase9_qa.json").relative_to(root)),"sha256":sha256_file(temp/"phase9_qa.json"),"record_count":1}]}
        (temp/"phase9_pre_split_manifest_v1.json").write_bytes(canonical_json_bytes(manifest)); os.replace(temp,output)
    except Exception:
        shutil.rmtree(temp,ignore_errors=True); raise
    return output/"phase9_pre_split_manifest_v1.json"
