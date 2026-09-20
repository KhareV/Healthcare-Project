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
    subjects=[]; episodes=[]; events=[]
    epoch=datetime.fromisoformat(spec["calendar_policy"]["deterministic_epoch"].replace("Z","+00:00"))
    for ordinal in range(1,config.n_subjects+1):
        irng=subject_rng(config.seed,ordinal,"identity"); p=config.values["population"]
        age=int(irng.integers(p["age_min"],p["age_max"]+1)); gi=int(irng.choice(len(p["condition_groups"]),p=p["condition_probabilities"]))
        subject=Subject(f"SYN-S-{ordinal:08d}",age,str(irng.choice(p["sex_categories"],p=p["sex_probabilities"])),p["condition_groups"][gi])
        z0,equilibrium,support_propensity=initial_state(config,ordinal,gi); erng=subject_rng(config.seed,ordinal,"episode"); d=config.values["duration"]
        log_hours=float(d["intercept"]+np.dot(d["latent_coefficients"],z0)+d["support_propensity_coefficient"]*support_propensity+erng.normal(0,d["noise_scale"]))
        duration=round(max(d["minimum_hours"],min(d["maximum_hours"],math.exp(log_hours))),6)
        offset=int(erng.integers(0,config.values["calendar"]["maximum_intime_offset_days"]*86400+1)); intime=epoch+timedelta(seconds=offset); outtime=intime+timedelta(hours=duration)
        episode=Episode(subject.subject_id,f"SYN-E-{ordinal:08d}",_iso(intime),_iso(outtime),1,stream_id(config.seed,ordinal))
        states=simulate(config,ordinal,z0,equilibrium,int(math.ceil(duration)))
        subjects.append(subject); episodes.append(episode); events.extend(generate_events(config,subject,episode,intime,duration,states,inventory))
    sr=[r.export() for r in sorted(subjects,key=lambda x:x.subject_id)]; ep=[r.export() for r in sorted(episodes,key=lambda x:(x.subject_id,x.stay_id))]
    ev=[r.export() for r in sorted(events,key=lambda x:(x.stay_id,x.event_time,x.event_id))]
    validate_dataset(sr,ep,ev,[],schema,inventory); return sr,ep,ev,[]

def _qa(mode,subjects,episodes,events):
    durations=[]
    for e in episodes:
        durations.append((datetime.fromisoformat(e["outtime"].replace("Z","+00:00"))-datetime.fromisoformat(e["intime"].replace("Z","+00:00"))).total_seconds()/3600)
    counts=Counter(r["concept_code"] for r in events); values={k:[r["value_numeric"] for r in events if r["concept_code"]==k] for k in counts}; fps=Counter()
    for s in subjects:
        stay=next(e for e in episodes if e["subject_id"]==s["subject_id"]); start=datetime.fromisoformat(stay["intime"].replace("Z","+00:00"))
        trajectory=[(r["concept_code"],round((datetime.fromisoformat(r["event_time"].replace("Z","+00:00"))-start).total_seconds(),6),r["value_numeric"]) for r in events if r["subject_id"]==s["subject_id"]]
        fps[json.dumps([s["age_years"],s["sex_category"],s["cardiac_condition_group"],trajectory],separators=(",",":"))]+=1
    legal_counts=[sum(1 for k in range(12) if 24+6*k <= hours-6) for hours in durations]
    observed_subjects={k:len({r["subject_id"] for r in events if r["concept_code"]==k}) for k in counts}
    return {"qa_version":"phase3_generator_qa_v1","dataset_scope":mode.upper(),"disclaimer":"Synthetic benchmark only; not clinical validation, representative prevalence, patient-level realism, or a clinical-use guarantee.","predictive_metrics_computed":False,"structural_validation":"PASS","counts":{"subjects":len(subjects),"episodes":len(episodes),"raw_events":len(events),"events_by_concept":dict(sorted(counts.items()))},"episode_duration_hours":{"minimum":min(durations),"median":float(np.median(durations)),"maximum":max(durations)},"observations_per_stay":{"minimum":min(sum(r["subject_id"]==s["subject_id"] for r in events) for s in subjects),"median":float(np.median([sum(r["subject_id"]==s["subject_id"] for r in events) for s in subjects])),"maximum":max(sum(r["subject_id"]==s["subject_id"] for r in events) for s in subjects)},"observation_absence_by_concept":{k:{"subjects_observed":observed_subjects[k],"subjects_without_observation":len(subjects)-observed_subjects[k]} for k in sorted(counts)},"legal_cutoff_possibilities":{"total":sum(legal_counts),"per_episode_counts":legal_counts,"episodes_with_zero":sum(n==0 for n in legal_counts),"episodes_with_multiple":sum(n>1 for n in legal_counts)},"value_ranges":{k:{"minimum":min(v),"maximum":max(v)} for k,v in sorted(values.items())},"exact_subject_clone_count":sum(n-1 for n in fps.values() if n>1),"scientific_support_generation_status":"DEFERRED_TO_PHASE_9"}

def generate_to_directory(config_path: Path,output: Path,repo_root: Path,generation_command: str,generation_timestamp: Optional[str]=None):
    config=load_runtime_config(config_path,repo_root)
    if output.exists(): raise FileExistsError(f"output already exists; refusing overwrite: {output}")
    subjects,episodes,events,supports=generate(config,repo_root); output.parent.mkdir(parents=True,exist_ok=True); temporary=Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-",dir=output.parent))
    try:
        tables={"subjects":subjects,"episodes":episodes,"raw_events":events,"support_intervals":supports}
        for name,rows in tables.items(): _write_jsonl(temporary/f"{name}.jsonl",rows)
        (temporary/"generator_qa.json").write_bytes(canonical_json_bytes(_qa(config.mode,subjects,episodes,events)))
        spec,schema,schema_path=_load_authority(repo_root); timestamp=generation_timestamp or datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00","Z")
        manifest={"manifest_version":"synthetic_dataset_manifest_v1","manifest_status":config.mode.upper(),"project_scope_version":spec["scope"]["version"],"project_scope_sha256":spec["scope"]["sha256"],"generator_version":GENERATOR_VERSION,"generator_code_identity":{**git_identity(repo_root),"source_bundle_sha256":source_bundle_sha256(repo_root)},"generator_config_version":config.values["runtime_config_version"],"generator_config_sha256":config.sha256,"generator_config_source_path":str(config_path.resolve().relative_to(repo_root)),"generator_config_source_sha256":sha256_file(config_path),"raw_schema_version":schema["schema_version"],"raw_schema_sha256":sha256_file(schema_path),"seed_policy_version":spec["rng_policy"]["version"],"primary_seed_identity":{"value":config.seed,"status":"ENGINEERING_ONLY_NON_FINAL"},"environment_identity":environment_identity(),"generation_timestamp_utc":timestamp,"generation_command":generation_command,"requested_subject_count":config.n_subjects,"generated_subject_count":len(subjects),"record_counts":{k:len(tables[k]) for k in sorted(tables)},"artifacts":[artifact_entry(repo_root,temporary,output,k,k,v) for k,v in tables.items()],"scientific_support_generation_status":"DEFERRED_TO_PHASE_9","generation_assertions":{"contains_real_patient_data":False,"contains_model_ready_examples":False,"contains_prediction_cutoffs":False,"contains_labels":False,"contains_split_membership":False,"contains_generator_latent_state":False,"final_artifact_manifest_complete":True}}
        (temporary/MANIFEST_NAME).write_bytes(canonical_json_bytes(manifest)); os.replace(temporary,output); validate_manifest(output/MANIFEST_NAME,repo_root,config,schema_path)
    except Exception:
        shutil.rmtree(temporary,ignore_errors=True)
        if output.exists(): shutil.rmtree(output,ignore_errors=True)
        raise
    return output/MANIFEST_NAME
