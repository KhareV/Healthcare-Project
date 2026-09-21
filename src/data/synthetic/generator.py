"""Phase-3 raw synthetic generator orchestrator (never model-ready output)."""
from __future__ import annotations
from collections import Counter
from datetime import datetime, timedelta, timezone
import json, math, os, shutil, tempfile
from pathlib import Path
from typing import Optional
import numpy as np
import yaml
from .config import GENERATOR_VERSION, RuntimeConfig, canonical_json_bytes, load_runtime_config
from .latent_state import initial_state, simulate
from .manifest import MANIFEST_NAME, artifact_entry, validate_manifest
from .observations import generate_events
from .provenance import environment_identity, git_identity, sha256_file, source_bundle_sha256
from .records import Episode, Subject
from .rng import stream_id, subject_rng
from .support import load_support_contract, simulate_with_support
from .validation import validate_dataset

def _iso(value): return value.isoformat(timespec="microseconds").replace("+00:00", "Z")
def _write_jsonl(path, rows):
    with path.open("wb") as handle:
        for row in rows: handle.write(canonical_json_bytes(row))
def _load_authority(root):
    spec_path=root/"configs/synthetic/synthetic_generator_v1.yaml"; schema_path=root/"configs/synthetic/synthetic_schema_v1.json"
    return yaml.safe_load(spec_path.read_text()), json.loads(schema_path.read_text()), schema_path

def generate(config: RuntimeConfig, repo_root: Path):
    spec,schema,_=_load_authority(repo_root); inventory=spec["raw_variable_inventory"]
    if set(config.values["variables"]) != set(inventory): raise ValueError("runtime variables must exactly match Phase-2 raw_variable_inventory")
    subjects=[]; episodes=[]; events=[]; supports=[]
    support_contract = None
    if config.mode == "final":
        support_contract = load_support_contract(repo_root / config.values["support_process_path"], repo_root / config.values["latent_process_path"])
    epoch=datetime.fromisoformat(spec["calendar_policy"]["deterministic_epoch"].replace("Z","+00:00"))
    candidate_ordinal=0
    while len(subjects)<config.n_subjects:
        candidate_ordinal+=1
        ordinal=len(subjects)+1
        irng=subject_rng(config.seed,candidate_ordinal,"identity"); p=config.values["population"]
        age=int(irng.integers(p["age_min"],p["age_max"]+1)); gi=int(irng.choice(len(p["condition_groups"]),p=p["condition_probabilities"]))
        sex=str(irng.choice(p["sex_categories"],p=p["sex_probabilities"]))
        z0,equilibrium,support_propensity=initial_state(config,candidate_ordinal,gi); erng=subject_rng(config.seed,candidate_ordinal,"episode"); d=config.values["duration"]
        log_hours=float(d["intercept"]+np.dot(d["latent_coefficients"],z0)+d["support_propensity_coefficient"]*support_propensity+erng.normal(0,d["noise_scale"]))
        duration=round(max(d["minimum_hours"],min(d["maximum_hours"],math.exp(log_hours))),6)
        if config.mode=="final" and duration<float(config.values["minimum_accepted_episode_hours"]): continue
        subject=Subject(f"SYN-S-{ordinal:08d}",age,sex,p["condition_groups"][gi])
        offset=int(erng.integers(0,config.values["calendar"]["maximum_intime_offset_days"]*86400+1)); intime=epoch+timedelta(seconds=offset); outtime=intime+timedelta(hours=duration)
        episode=Episode(subject.subject_id,f"SYN-E-{ordinal:08d}",_iso(intime),_iso(outtime),1,stream_id(config.seed,candidate_ordinal))
        if support_contract is None:
            states=simulate(config,candidate_ordinal,z0,equilibrium,int(math.ceil(duration))); subject_supports=[]
        else:
            states,subject_supports=simulate_with_support(config,candidate_ordinal,z0,equilibrium,int(math.ceil(duration)),support_contract,intime)
        subjects.append(subject); episodes.append(episode); events.extend(generate_events(config,subject,episode,intime,duration,states,inventory,rng_subject_ordinal=candidate_ordinal))
        for row in subject_supports:
            row["subject_id"]=subject.subject_id; row["stay_id"]=episode.stay_id
            row["support_event_id"]=f"{episode.stay_id}-SUP-{int(row['support_event_id'].rsplit('-',1)[1]):06d}"
            if datetime.fromisoformat(row["interval_start"].replace("Z","+00:00")) >= outtime: continue
            if datetime.fromisoformat(row["interval_end"].replace("Z","+00:00")) > outtime: row["interval_end"]=_iso(outtime)
            supports.append(row)
    sr=[r.export() for r in sorted(subjects,key=lambda x:x.subject_id)]; ep=[r.export() for r in sorted(episodes,key=lambda x:(x.subject_id,x.stay_id))]
    ev=[r.export() for r in sorted(events,key=lambda x:(x.stay_id,x.event_time,x.event_id))]
    supports.sort(key=lambda x:(x["stay_id"],x["interval_start"],x["support_event_id"]))
    validate_dataset(sr,ep,ev,supports,schema,inventory,allow_support=config.mode=="final"); return sr,ep,ev,supports

def _qa(mode,subjects,episodes,events,supports=()):
    episode_by_subject={e["subject_id"]:e for e in episodes}
    durations=[(datetime.fromisoformat(e["outtime"].replace("Z","+00:00"))-datetime.fromisoformat(e["intime"].replace("Z","+00:00"))).total_seconds()/3600 for e in episodes]
    counts=Counter(); values={}; observed={}; trajectories={s["subject_id"]:[] for s in subjects}; event_counts=Counter()
    for row in events:
        concept=row["concept_code"]; subject=row["subject_id"]
        counts[concept]+=1; values.setdefault(concept,[]).append(row["value_numeric"]); observed.setdefault(concept,set()).add(subject); event_counts[subject]+=1
        start=datetime.fromisoformat(episode_by_subject[subject]["intime"].replace("Z","+00:00"))
        trajectories[subject].append((concept,round((datetime.fromisoformat(row["event_time"].replace("Z","+00:00"))-start).total_seconds(),6),row["value_numeric"]))
    fps=Counter(json.dumps([s["age_years"],s["sex_category"],s["cardiac_condition_group"],trajectories[s["subject_id"]]],separators=(",",":")) for s in subjects)
    legal_counts=[sum(1 for k in range(12) if 24+6*k <= hours-6) for hours in durations]
    per_stay=list(event_counts.values())
    return {"qa_version":"phase3_generator_qa_v1","dataset_scope":mode.upper(),"disclaimer":"Synthetic benchmark only; not clinical validation, representative prevalence, patient-level realism, or a clinical-use guarantee.","predictive_metrics_computed":False,"structural_validation":"PASS","counts":{"subjects":len(subjects),"episodes":len(episodes),"raw_events":len(events),"events_by_concept":dict(sorted(counts.items()))},"episode_duration_hours":{"minimum":min(durations),"median":float(np.median(durations)),"maximum":max(durations)},"observations_per_stay":{"minimum":min(per_stay),"median":float(np.median(per_stay)),"maximum":max(per_stay)},"observation_absence_by_concept":{k:{"subjects_observed":len(observed[k]),"subjects_without_observation":len(subjects)-len(observed[k])} for k in sorted(counts)},"legal_cutoff_possibilities":{"total":sum(legal_counts),"per_episode_counts":legal_counts,"episodes_with_zero":sum(n==0 for n in legal_counts),"episodes_with_multiple":sum(n>1 for n in legal_counts)},"value_ranges":{k:{"minimum":min(v),"maximum":max(v)} for k,v in sorted(values.items())},"exact_subject_clone_count":sum(n-1 for n in fps.values() if n>1),"scientific_support_generation_status":("GENERATED_FROZEN_PHASE9" if supports else "DEFERRED_TO_PHASE_9"),"support_interval_segments":len(supports)}

def generate_to_directory(config_path: Path,output: Path,repo_root: Path,generation_command: str,generation_timestamp: Optional[str]=None):
    config=load_runtime_config(config_path,repo_root)
    if output.exists(): raise FileExistsError(f"output already exists; refusing overwrite: {output}")
    subjects,episodes,events,supports=generate(config,repo_root); output.parent.mkdir(parents=True,exist_ok=True); temporary=Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-",dir=output.parent))
    try:
        tables={"subjects":subjects,"episodes":episodes,"raw_events":events,"support_intervals":supports}
        for name,rows in tables.items(): _write_jsonl(temporary/f"{name}.jsonl",rows)
        (temporary/"generator_qa.json").write_bytes(canonical_json_bytes(_qa(config.mode,subjects,episodes,events,supports)))
        spec,schema,schema_path=_load_authority(repo_root); timestamp=generation_timestamp or datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00","Z")
        manifest={"manifest_version":"synthetic_dataset_manifest_v1","manifest_status":("AUTHORIZED_FINAL_SYNTHETIC_DATA" if config.mode=="final" else config.mode.upper()),"project_scope_version":spec["scope"]["version"],"project_scope_sha256":spec["scope"]["sha256"],"generator_version":GENERATOR_VERSION,"generator_code_identity":{**git_identity(repo_root),"source_bundle_sha256":source_bundle_sha256(repo_root)},"generator_config_version":config.values["runtime_config_version"],"generator_config_sha256":config.sha256,"generator_config_source_path":str(config_path.resolve().relative_to(repo_root)),"generator_config_source_sha256":sha256_file(config_path),"raw_schema_version":schema["schema_version"],"raw_schema_sha256":sha256_file(schema_path),"seed_policy_version":spec["rng_policy"]["version"],"primary_seed_identity":{"value":config.seed,"status":("FROZEN_FINAL_BENCHMARK" if config.mode=="final" else "ENGINEERING_ONLY_NON_FINAL")},"environment_identity":environment_identity(),"generation_timestamp_utc":timestamp,"generation_command":generation_command,"requested_subject_count":config.n_subjects,"generated_subject_count":len(subjects),"record_counts":{k:len(tables[k]) for k in sorted(tables)},"artifacts":[artifact_entry(repo_root,temporary,output,k,k,v) for k,v in tables.items()],"scientific_support_generation_status":("GENERATED_FROZEN_PHASE9" if config.mode=="final" else "DEFERRED_TO_PHASE_9"),"generation_assertions":{"contains_real_patient_data":False,"contains_model_ready_examples":False,"contains_prediction_cutoffs":False,"contains_labels":False,"contains_split_membership":False,"contains_generator_latent_state":False,"final_artifact_manifest_complete":True}}
        (temporary/MANIFEST_NAME).write_bytes(canonical_json_bytes(manifest)); os.replace(temporary,output); validate_manifest(output/MANIFEST_NAME,repo_root,config,schema_path)
    except Exception:
        shutil.rmtree(temporary,ignore_errors=True)
        if output.exists(): shutil.rmtree(output,ignore_errors=True)
        raise
    return output/MANIFEST_NAME
