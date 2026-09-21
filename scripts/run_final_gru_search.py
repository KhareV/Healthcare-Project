#!/usr/bin/env python3
"""Execute the frozen validation-only final GRU scientific search."""
from __future__ import annotations

from datetime import datetime, timezone
import csv
import json
import math
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch

from data.collate import create_dataloader
from data.gru_canonical import Phase10GRUDataset, TaskEligibleGRUDataset
from data.synthetic.config import canonical_json_bytes
from data.synthetic.provenance import sha256_file
from evaluation.metrics import PredictionRecord, evaluate_icu_time, evaluate_organ_support, evaluate_recovery
from evaluation.weighted_stats import weighted_median
from evaluation.weights import compute_stay_weights
from experiments.gru_final import (
    MAX_ATTEMPTS, SEARCH_VERSION, TASKS, canonical_sha256, choose_best,
    validate_candidates, validate_checkpoint_lineage, validate_search_space,
)
from experiments.lineage import ArtifactRecord, read_artifact_index, read_run_registry, validate_artifact_lineage, write_artifact_index
from experiments.search_registry import record_search_attempt
from models.gru import GRUEncoderConfig
from models.gru_icu_time import ICU_TIME_RAW_OUTPUT, ICUTimeGRU, load_icu_time_bundle
from models.gru_recovery import RECOVERY_HORIZON_ORDER, RecoveryGRU, load_recovery_bundle
from models.gru_support import SUPPORT_OUTPUT_TYPE, SUPPORT_PROBABILITY_TRANSFORM, OrganSupportGRU, load_support_bundle, uncalibrated_support_probability
from models.icu_time_postprocess import ICU_TIME_POSTPROCESS_VERSION, remaining_icu_hours_from_log_prediction
from preprocess.target_scaler import RecoveryTargetScaler
from training.checkpoint import save_checkpoint
from training.class_weights import SupportClassWeight
from training.engine import create_optimizer, run_epoch
from training.logging import JsonlRunLogger
from training.reproducibility import set_deterministic_seed
from training.tasks.icu_time import ICUTimeTaskAdapter
from training.tasks.organ_support import OrganSupportTaskAdapter
from training.tasks.recovery import RecoveryTaskAdapter


SEARCH_ROOT = ROOT / "artifacts/search/gru/final_v2"
MASTER_PATH = SEARCH_ROOT / "manifests/search_manifest_v1.json"
REGISTRY_PATH = ROOT / "experiments/registry.csv"
ARTIFACT_INDEX_PATH = ROOT / "experiments/artifacts.csv"
TASK_REGISTRY = {"recovery": "recovery", "icu_time": "icu_stay_time", "organ_support": "organ_support"}


def _load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp"); temporary.write_bytes(canonical_json_bytes(value)); temporary.replace(path)


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        for row in rows: handle.write(canonical_json_bytes(row) + b"\n")
    temporary.replace(path)


def _git(*args):
    return subprocess.check_output(("git",) + args, cwd=ROOT, text=True).strip()


def _utc():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _ensure_code(commit):
    if _git("rev-parse", "HEAD") != commit:
        raise RuntimeError("GRU search execution commit changed")
    result = subprocess.run(("git", "diff", "--quiet", commit, "--", "src", "scripts", "configs", "tests"), cwd=ROOT)
    if result.returncode:
        raise RuntimeError("science-affecting code/config changed during GRU search")


def _verify(master, space, commit):
    _ensure_code(commit); validate_search_space(space)
    checks = {
        master["g1_path"]: master["g1_sha256"], master["phase14_handoff_path"]: master["phase14_handoff_sha256"],
        master["search_space_path"]: master["search_space_file_sha256"], master["environment_path"]: master["environment_sha256"],
        "configs/synthetic/feature_schema_v2.json": master["feature_schema_sha256"],
        "artifacts/splits/synthetic_split_v2.csv": master["split_sha256"],
        "artifacts/preprocessors/synthetic_feature_preprocessor_v1.json": master["preprocessor_sha256"],
        "artifacts/preprocessors/recovery_target_scaler_synthetic_v1.json": master["recovery_target_scaler_sha256"],
        "artifacts/preprocessors/support_class_weight_synthetic_v1.json": master["support_class_weight_sha256"],
    }
    for relative, expected in checks.items():
        if sha256_file(ROOT / relative) != expected: raise RuntimeError("frozen GRU parent changed: " + relative)
    if (ROOT / "artifacts/models/selected_models_v1.json").exists() or (ROOT / "artifacts/governance/g3_freeze.json").exists():
        raise RuntimeError("selected-model/G3 artifact is forbidden during Stage 1")


def _manifest(master, task):
    item = master["candidate_manifests"][task]
    path = ROOT / item["path"]
    if sha256_file(path) != item["sha256"]: raise RuntimeError("candidate manifest hash mismatch")
    payload = _load(path); validate_candidates(payload["candidates"], task)
    if payload["candidate_list_hash"] != item["candidate_list_hash"]: raise RuntimeError("candidate-list hash mismatch")
    return payload


def _model(task, candidate, master):
    cfg = candidate["config"]
    encoder = GRUEncoderConfig(feature_dim=master["dimensions"]["F"], hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"], dropout=cfg["dropout"], include_observation_mask=True,
        include_tslo=True, static_dim=master["dimensions"]["S_model"], bidirectional=False)
    return RecoveryGRU(encoder) if task == "recovery" else (ICUTimeGRU(encoder) if task == "icu_time" else OrganSupportGRU(encoder))


def _record(key, target, prediction, eligible):
    return PredictionRecord(stay_id=key[1], prediction_time=key[2], target=float(target) if eligible else math.nan,
                            prediction=float(prediction), eligible=bool(eligible))


def _metrics_and_rows(task, dataset, predictions):
    source = dataset.rows
    keys = [(r["subject_id"], r["stay_id"], r["prediction_time"], r["grid_index"]) for r in source]
    if task == "recovery":
        records = []
        result_rows = []
        for horizon, column, target_name, eligible_name in (("24h", 0, "delta_sofa_24", "recovery24_eligible"), ("48h", 1, "delta_sofa_48", "recovery48_eligible")):
            horizon_records = tuple(_record(k, r[target_name], p[column], r[eligible_name]) for k, r, p in zip(keys, source, predictions))
            records.append(horizon_records)
            for k, r, p in zip(keys, source, predictions):
                result_rows.append({"subject_id":k[0],"stay_id":k[1],"prediction_time":k[2],"grid_index":k[3],"task":"recovery","horizon":horizon,
                    "eligible":bool(r[eligible_name]),"true_target":float(r[target_name]) if r[eligible_name] else None,"prediction":float(p[column]),"prediction_units":"sofa_delta"})
        evaluated = evaluate_recovery(records[0], records[1], metadata={"model_family":"gru","partition":"validation"})
        metrics = {"mae24":evaluated["24h"].metrics["mae"],"mae48":evaluated["48h"].metrics["mae"],
                   "rmse24":evaluated["24h"].metrics["rmse"],"rmse48":evaluated["48h"].metrics["rmse"],
                   "directional_accuracy24":evaluated["24h"].metrics["directional_agreement"],
                   "directional_accuracy48":evaluated["48h"].metrics["directional_agreement"]}
        for name, values in (("24", records[0]), ("48", records[1])):
            weights = compute_stay_weights([r.stay_id for r in values], [r.eligible for r in values])
            metrics["median_absolute_error" + name] = weighted_median(
                [abs(values[i].prediction-values[i].target) for i in weights.eligible_indices],
                [weights.weights[i] for i in weights.eligible_indices])
        return metrics, result_rows
    if task == "icu_time":
        true_hours = [math.expm1(r["icu_time_log1p"]) if r["icu_time_eligible"] else math.nan for r in source]
        records = tuple(_record(k, t, p, r["icu_time_eligible"]) for k,r,t,p in zip(keys,source,true_hours,predictions["hours"]))
        evaluated = evaluate_icu_time(records, metadata={"model_family":"gru","partition":"validation"})
        rows = [{"subject_id":k[0],"stay_id":k[1],"prediction_time":k[2],"grid_index":k[3],"task":"icu_time","eligible":bool(r["icu_time_eligible"]),
                 "true_target_log1p":float(r["icu_time_log1p"]) if r["icu_time_eligible"] else None,"prediction_log1p":float(raw),
                 "true_target":float(t) if r["icu_time_eligible"] else None,"prediction":float(p),"prediction_units":"hours"}
                for k,r,t,p,raw in zip(keys,source,true_hours,predictions["hours"],predictions["raw"])]
        return {"median_absolute_error_hours":evaluated.metrics["median_absolute_error"],"mae_hours":evaluated.metrics["mae"],"rmse_hours":evaluated.metrics["rmse"]}, rows
    records = tuple(_record(k, r["organ_support_label"], p, r["organ_support_eligible"]) for k,r,p in zip(keys,source,predictions))
    evaluated = evaluate_organ_support(records, probability_type="raw", metadata={"model_family":"gru","partition":"validation"})
    rows = [{"subject_id":k[0],"stay_id":k[1],"prediction_time":k[2],"grid_index":k[3],"task":"organ_support","eligible":bool(r["organ_support_eligible"]),
             "true_target":int(r["organ_support_label"]) if r["organ_support_eligible"] else None,"prediction":float(p),"probability_type":"raw_uncalibrated","calibrated":False}
            for k,r,p in zip(keys,source,predictions)]
    return {"auprc":evaluated.metrics["auprc"],"auroc":evaluated.metrics["auroc"],"brier":evaluated.metrics["brier"],"probability_type":"raw_uncalibrated"}, rows


def _predict(task, model, loader, dataset, device, scaler):
    model.eval(); values=[]
    with torch.no_grad():
        for batch in loader:
            output=model(batch.to(device))
            if task == "recovery": output=scaler.inverse_transform(output)
            elif task == "icu_time":
                raw=output[:,0]; values.append((raw.cpu(),remaining_icu_hours_from_log_prediction(raw).cpu())); continue
            else: output=uncalibrated_support_probability(output)[:,0]
            values.append(output.cpu())
    if task == "icu_time":
        return {"raw":torch.cat([x[0] for x in values]).numpy(),"hours":torch.cat([x[1] for x in values]).numpy()}
    return torch.cat(values).numpy()


def _epoch_key(task, metrics, epoch):
    row={**metrics,"candidate_id":"","status":"COMPLETE"}
    if task == "recovery": return (metrics["mae24"],metrics["mae48"],metrics["rmse24"],epoch)
    if task == "icu_time": return (metrics["median_absolute_error_hours"],metrics["mae_hours"],metrics["rmse_hours"],epoch)
    return (-metrics["auprc"],metrics["brier"],-metrics["auroc"],epoch)


def _metadata(task, run_id, candidate, master, model, validation_value):
    data={"run_id":run_id,"task":TASK_REGISTRY[task],"model_family":"gru","model_config":model.model_config,
        "seed":candidate["model_seed"],"epoch":0,"config_hash":candidate["config_hash"],
        "tensor_contract_version":master["tensor_contract_version"],"feature_schema_version":master["feature_schema_version"],
        "split_hash":master["split_sha256"],"code_commit":master["execution_commit"],"validation_value":float(validation_value),
        "synthetic_smoke_test":False,"synthetic_scientific_search":True,"g1_sha256":master["g1_sha256"],
        "phase14_handoff_sha256":master["phase14_handoff_sha256"],"feature_schema_sha256":master["feature_schema_sha256"],
        "preprocessor_sha256":master["preprocessor_sha256"],"test_accessed":False}
    if task=="recovery": data.update(horizon_order=list(RECOVERY_HORIZON_ORDER),target_scaler_artifact="artifacts/preprocessors/recovery_target_scaler_synthetic_v1.json",target_scaler_sha256=master["recovery_target_scaler_sha256"])
    elif task=="icu_time": data.update(raw_output_meaning=ICU_TIME_RAW_OUTPUT,postprocess_version=ICU_TIME_POSTPROCESS_VERSION,target_scaler=None)
    else:
        weight=SupportClassWeight.load(ROOT/"artifacts/preprocessors/support_class_weight_synthetic_v1.json")
        data.update(output_type=SUPPORT_OUTPUT_TYPE,probability_transform=SUPPORT_PROBABILITY_TRANSFORM,
            class_weight_artifact="artifacts/preprocessors/support_class_weight_synthetic_v1.json",class_weight_sha256=master["support_class_weight_sha256"],
            label_contract_version=weight.label_contract_version,event_dictionary_version=weight.event_dictionary_version,calibration=None,operating_threshold=None)
    return data


def _train(task, candidate, master, train_dataset, validation_dataset, attempt_dir, run_id):
    attempt_dir.mkdir(parents=True, exist_ok=True)
    seed=candidate["model_seed"]; set_deterministic_seed(seed); torch.set_num_threads(1); device=torch.device("cpu")
    config=candidate["config"]
    task_train_dataset=TaskEligibleGRUDataset(train_dataset,task)
    train_loader=create_dataloader(task_train_dataset,batch_size=config["batch_size"],shuffle=True,seed=seed,num_workers=0)
    validation_loader=create_dataloader(validation_dataset,batch_size=config["batch_size"],shuffle=False,seed=seed,num_workers=0)
    model=_model(task,candidate,master).to(device)
    optimizer=create_optimizer(model.parameters(),name="adamw",learning_rate=config["learning_rate"],weight_decay=config["weight_decay"])
    scaler=RecoveryTargetScaler.load(ROOT/"artifacts/preprocessors/recovery_target_scaler_synthetic_v1.json")
    weight=SupportClassWeight.load(ROOT/"artifacts/preprocessors/support_class_weight_synthetic_v1.json")
    adapter=RecoveryTaskAdapter(scaler) if task=="recovery" else (ICUTimeTaskAdapter() if task=="icu_time" else OrganSupportTaskAdapter(weight.pos_weight))
    logger=JsonlRunLogger(attempt_dir/"training_log.jsonl"); checkpoint=attempt_dir/"model.pt"
    best_key=None; best_metrics=None; best_epoch=None; bad=0; started=time.monotonic()
    for epoch in range(1,61):
        train_result=run_epoch(model,train_loader,adapter,device,optimizer=optimizer)
        predictions=_predict(task,model,validation_loader,validation_dataset,device,scaler)
        metrics,_=_metrics_and_rows(task,validation_dataset,predictions); key=_epoch_key(task,metrics,epoch)
        improved=best_key is None or key<best_key
        logger.log({"event":"epoch","run_id":run_id,"epoch":epoch,"train_loss":train_result.mean_loss,"validation_metrics":metrics,"improved":improved})
        if improved:
            best_key,best_metrics,best_epoch,bad=key,metrics,epoch,0
            primary=metrics["mae24"] if task=="recovery" else (metrics["median_absolute_error_hours"] if task=="icu_time" else metrics["auprc"])
            metadata=_metadata(task,run_id,candidate,master,model,primary); metadata["epoch"]=epoch
            save_checkpoint(checkpoint,model,optimizer,metadata)
        else: bad+=1
        if bad>=8: break
    expected={"g1_sha256":master["g1_sha256"],"phase14_handoff_sha256":master["phase14_handoff_sha256"],
              "preprocessor_sha256":master["preprocessor_sha256"],"config_hash":candidate["config_hash"],"seed":seed}
    validate_checkpoint_lineage(checkpoint,expected)
    if task=="recovery": model,_,sidecar=load_recovery_bundle(checkpoint,repository_root=ROOT,expected_tensor_contract_version=master["tensor_contract_version"],expected_feature_schema_version=master["feature_schema_version"])
    elif task=="icu_time": model,sidecar=load_icu_time_bundle(checkpoint,expected_tensor_contract_version=master["tensor_contract_version"],expected_feature_schema_version=master["feature_schema_version"])
    else: model,_,sidecar=load_support_bundle(checkpoint,repository_root=ROOT,expected_tensor_contract_version=master["tensor_contract_version"],expected_feature_schema_version=master["feature_schema_version"])
    model.to(device); final_predictions=_predict(task,model,validation_loader,validation_dataset,device,scaler)
    final_metrics,rows=_metrics_and_rows(task,validation_dataset,final_predictions)
    if canonical_sha256(final_metrics)!=canonical_sha256(best_metrics): raise RuntimeError("restored best GRU metrics differ")
    predictions_path=attempt_dir/"validation_predictions.jsonl"; _write_jsonl(predictions_path,rows)
    return {"metrics":final_metrics,"best_epoch":best_epoch,"epochs_completed":epoch,"checkpoint":checkpoint,
            "checkpoint_sha256":sidecar["checkpoint_sha256"],"predictions_path":predictions_path,"runtime_seconds":time.monotonic()-started}


def _register(task, manifest, candidate, attempt, run_id, result, metrics_path, master, retry_of):
    registry_manifest={**manifest,"task":TASK_REGISTRY[task],"code_commit":master["execution_commit"],"search_space_ref":master["search_space_path"],
        "search_space_hash":master["search_space_sha256"],"split_hash":master["split_sha256"],"feature_version":master["feature_schema_version"],
        "label_version":"synthetic_phase9_final_target_contract_v1","validation_objective":master["validation_objectives"][task]}
    record_search_attempt(REGISTRY_PATH,manifest=registry_manifest,candidates=manifest["candidates"],candidate_id=candidate["candidate_id"],run_id=run_id,
        attempt_number=attempt,attempt_status_detail="COMPLETE",timestamp_utc=_utc(),seed=candidate["model_seed"],retry_of_run_id=retry_of,
        artifact_ref=str(result["checkpoint"].relative_to(ROOT)),artifact_hash=result["checkpoint_sha256"],metrics_ref=str(metrics_path.relative_to(ROOT)),metrics_hash=sha256_file(metrics_path),
        config_ref=master["search_space_path"],run_type="scientific",preprocessor_ref="artifacts/preprocessors/synthetic_feature_preprocessor_v1.json",
        preprocessor_sha256=master["preprocessor_sha256"],derived_feature_hash=master["feature_schema_sha256"],environment_ref=master["environment_path"],environment_sha256=master["environment_sha256"],
        notes=json.dumps({"search_role":"GRU_WITHIN_FAMILY_VALIDATION_CANDIDATE","best_epoch":result["best_epoch"],"g1_sha256":master["g1_sha256"],"phase14_handoff_sha256":master["phase14_handoff_sha256"],"test_accessed":False},sort_keys=True,separators=(",",":")))
    common=dict(producing_run_id=run_id,task=TASK_REGISTRY[task],model_family="gru",split_hash=master["split_sha256"],feature_version=master["feature_schema_version"],
        label_version="synthetic_phase9_final_target_contract_v1",config_hash=candidate["config_hash"],creation_commit=master["execution_commit"],generating_script="scripts/run_final_gru_search.py",
        creation_date_utc=_utc(),preprocessor_sha256=master["preprocessor_sha256"],environment_sha256=master["environment_sha256"],run_type="scientific",status="registered")
    model_id=run_id+":model"; pred_id=run_id+":predictions"
    records=[ArtifactRecord(artifact_id=model_id,artifact_path=str(result["checkpoint"].relative_to(ROOT)),artifact_type="model_checkpoint",artifact_version="final_gru_checkpoint_v1",artifact_sha256=result["checkpoint_sha256"],model_sha256=result["checkpoint_sha256"],metadata_ref=str((result["checkpoint"].with_name("model.pt.metadata.json")).relative_to(ROOT)),**common),
        ArtifactRecord(artifact_id=pred_id,artifact_path=str(result["predictions_path"].relative_to(ROOT)),artifact_type="prediction",artifact_version="final_gru_validation_prediction_v1",artifact_sha256=sha256_file(result["predictions_path"]),parent_artifact_ids=model_id,model_sha256=result["checkpoint_sha256"],partition="validation",probability_type="raw_uncalibrated" if task=="organ_support" else "",prediction_population_hash=master["validation_row_keys_sha256"],**common),
        ArtifactRecord(artifact_id=run_id+":metrics",artifact_path=str(metrics_path.relative_to(ROOT)),artifact_type="metric_table",artifact_version="final_gru_validation_metrics_v1",artifact_sha256=sha256_file(metrics_path),parent_artifact_ids=pred_id,partition="validation",probability_type="raw_uncalibrated" if task=="organ_support" else "",evaluator_version="stay_balanced_metrics_v1",metric_implementation_version="stay_balanced_metrics_v1",**common)]
    combined=list(read_artifact_index(ARTIFACT_INDEX_PATH))+records
    validate_artifact_lineage(combined,read_run_registry(REGISTRY_PATH),repository_root=ROOT); write_artifact_index(ARTIFACT_INDEX_PATH,combined)


def _run_candidate(task,manifest,candidate,attempt,master,train,validation,retry_of=""):
    run_id=f"final-v2-{candidate['candidate_id']}-attempt-{attempt}"; directory=SEARCH_ROOT/task/candidate["candidate_id"]/f"attempt-{attempt}"
    directory.mkdir(parents=True,exist_ok=False); _write_json(directory/"candidate_config.json",candidate["config"])
    result=_train(task,candidate,master,train,validation,directory,run_id)
    payload={"metrics_version":"final_gru_validation_metrics_v1","search_version":SEARCH_VERSION,"candidate_id":candidate["candidate_id"],"run_id":run_id,"task":task,"status":"COMPLETE",
        "partition":"validation","metrics":result["metrics"],"best_epoch":result["best_epoch"],"epochs_completed":result["epochs_completed"],"checkpoint_path":str(result["checkpoint"].relative_to(ROOT)),
        "checkpoint_sha256":result["checkpoint_sha256"],"prediction_path":str(result["predictions_path"].relative_to(ROOT)),"prediction_sha256":sha256_file(result["predictions_path"]),
        "config_hash":candidate["config_hash"],"model_seed":candidate["model_seed"],"g1_sha256":master["g1_sha256"],"phase14_handoff_sha256":master["phase14_handoff_sha256"],
        "feature_schema_sha256":master["feature_schema_sha256"],"split_sha256":master["split_sha256"],"preprocessor_sha256":master["preprocessor_sha256"],
        "recovery_target_scaler_sha256":master["recovery_target_scaler_sha256"] if task=="recovery" else None,"support_class_weight_sha256":master["support_class_weight_sha256"] if task=="organ_support" else None,
        "probability_type":"raw_uncalibrated" if task=="organ_support" else None,"calibrated":False,"threshold":None,"test_accessed":False,"runtime_seconds":result["runtime_seconds"]}
    metrics_path=directory/"validation_metrics.json"; _write_json(metrics_path,payload); _register(task,manifest,candidate,attempt,run_id,result,metrics_path,master,retry_of)
    return payload


def _failure(task,manifest,candidate,attempt,master,error,retry_of):
    run_id=f"final-v2-{candidate['candidate_id']}-attempt-{attempt}"; directory=SEARCH_ROOT/task/candidate["candidate_id"]/f"attempt-{attempt}"
    if not directory.exists(): directory.mkdir(parents=True)
    config_path=directory/"candidate_config.json"
    if not config_path.exists(): _write_json(config_path,candidate["config"])
    failure=directory/"failure.json"; _write_json(failure,{"run_id":run_id,"candidate_id":candidate["candidate_id"],"attempt":attempt,"status":"FAILED_SOFTWARE" if attempt<MAX_ATTEMPTS else "FAILED_SCIENTIFIC","reason_type":type(error).__name__,"reason":str(error),"test_accessed":False})
    reg={**manifest,"task":TASK_REGISTRY[task],"code_commit":master["execution_commit"],"search_space_ref":master["search_space_path"],"search_space_hash":master["search_space_sha256"],"split_hash":master["split_sha256"],"feature_version":master["feature_schema_version"],"label_version":"synthetic_phase9_final_target_contract_v1","validation_objective":master["validation_objectives"][task]}
    record_search_attempt(REGISTRY_PATH,manifest=reg,candidates=manifest["candidates"],candidate_id=candidate["candidate_id"],run_id=run_id,attempt_number=attempt,attempt_status_detail="FAILED_SOFTWARE" if attempt<MAX_ATTEMPTS else "FAILED_SCIENTIFIC",timestamp_utc=_utc(),seed=candidate["model_seed"],retry_of_run_id=retry_of,config_ref=str(config_path.relative_to(ROOT)),run_type="scientific",preprocessor_ref="artifacts/preprocessors/synthetic_feature_preprocessor_v1.json",preprocessor_sha256=master["preprocessor_sha256"],derived_feature_hash=master["feature_schema_sha256"],environment_ref=master["environment_path"],environment_sha256=master["environment_sha256"],notes=json.dumps({"failure_ref":str(failure.relative_to(ROOT)),"test_accessed":False}))
    return run_id


def _summary(task,manifest):
    rows=[]
    for candidate in manifest["candidates"]:
        metrics=sorted((SEARCH_ROOT/task/candidate["candidate_id"]).glob("attempt-*/validation_metrics.json"))
        if metrics:
            payload=_load(metrics[-1]); rows.append({"candidate_id":candidate["candidate_id"],"config_hash":candidate["config_hash"],"status":"COMPLETE","run_id":payload["run_id"],"best_epoch":payload["best_epoch"],"metrics_path":str(metrics[-1].relative_to(ROOT)),"checkpoint_path":payload["checkpoint_path"],"checkpoint_sha256":payload["checkpoint_sha256"],**payload["metrics"]})
        else: rows.append({"candidate_id":candidate["candidate_id"],"config_hash":candidate["config_hash"],"status":"FAILED"})
    json_path=SEARCH_ROOT/f"{task}_validation_summary.json"; _write_json(json_path,{"task":task,"search_version":SEARCH_VERSION,"candidates":rows,"test_accessed":False})
    csv_path=SEARCH_ROOT/f"{task}_validation_summary.csv"; fields=sorted({k for row in rows for k in row})
    with csv_path.open("w",encoding="utf-8",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=fields,lineterminator="\n"); writer.writeheader(); writer.writerows(rows)
    return rows,json_path,csv_path


def _audit_winner(task,manifest,candidate,master,train,validation,source):
    directory=SEARCH_ROOT/task/candidate["candidate_id"]/"audit-reproduction"
    result=_train(task,candidate,master,train,validation,directory,"audit-"+candidate["candidate_id"])
    expected=source["metrics"]; actual=result["metrics"]
    differences={key:abs(float(actual[key])-float(expected[key])) for key in actual if isinstance(actual[key],(int,float)) and actual[key] is not None and expected.get(key) is not None}
    reproduced_prediction_hash=sha256_file(result["predictions_path"])
    prediction_exact=reproduced_prediction_hash==source["prediction_sha256"]
    passed=all(value<=1e-8 for value in differences.values()) and prediction_exact
    return {"candidate_id":candidate["candidate_id"],"task":task,"status":"PASS" if passed else "FAIL","tolerance":1e-8,"metric_absolute_differences":differences,"prediction_exact":prediction_exact,"source_prediction_sha256":source["prediction_sha256"],"reproduced_prediction_sha256":reproduced_prediction_hash,"test_accessed":False}


def main():
    master=_load(MASTER_PATH)
    if master["status"]!="FROZEN_READY_TO_RUN": raise RuntimeError("final GRU search is not frozen-ready")
    commit=_git("rev-parse","HEAD"); space=_load(ROOT/master["search_space_path"]); _verify(master,space,commit)
    manifests={task:_manifest(master,task) for task in TASKS}
    master.update(status="RUNNING",execution_commit=commit,started_at_utc=_utc()); _write_json(MASTER_PATH,master)
    train,validation=Phase10GRUDataset(ROOT,"train"),Phase10GRUDataset(ROOT,"validation")
    summaries={}
    try:
        for task in TASKS:
            manifest=manifests[task]
            for candidate in manifest["candidates"]:
                _verify(master,space,commit); retry=""; complete=False
                for attempt in range(1,MAX_ATTEMPTS+1):
                    try: _run_candidate(task,manifest,candidate,attempt,master,train,validation,retry); complete=True; break
                    except Exception as error: retry=_failure(task,manifest,candidate,attempt,master,error,retry)
                print(json.dumps({"task":task,"candidate_id":candidate["candidate_id"],"complete":complete}),flush=True)
            rows,jpath,cpath=_summary(task,manifest); summaries[task]={"rows":rows,"json":jpath,"csv":cpath,"best":choose_best(task,rows)}
        audits={}
        for task,value in summaries.items():
            winner=value["best"]; candidate=next(c for c in manifests[task]["candidates"] if c["candidate_id"]==winner["candidate_id"])
            source=_load(ROOT/winner["metrics_path"]); audits[task]=_audit_winner(task,manifests[task],candidate,master,train,validation,source)
            if audits[task]["status"]!="PASS": raise RuntimeError("winner reproducibility failed: "+task)
        audit_path=SEARCH_ROOT/"winner_reproducibility_audit_v1.json"; _write_json(audit_path,{"status":"PASS","search_version":SEARCH_VERSION,"tasks":audits,"search_slots_consumed":0,"test_accessed":False})
        best={"manifest_version":"best_gru_candidates_v1","status":"GRU_WITHIN_FAMILY_VALIDATION_WINNER","search_version":SEARCH_VERSION,
            "serving_selection":False,"final_model_family_selection":False,"selection_status":"NOT_FINAL_SERVING_SELECTION","test_status":"TEST_NOT_ACCESSED",
            "test_accessed":False,"support_calibrated":False,"support_status":"SUPPORT_UNCALIBRATED","support_threshold":None,
            "g1_sha256":master["g1_sha256"],"phase14_handoff_sha256":master["phase14_handoff_sha256"],"tasks":{}}
        for task,value in summaries.items():
            row=value["best"]; best["tasks"][task]={"candidate_id":row["candidate_id"],"run_id":row["run_id"],"config_hash":row["config_hash"],"checkpoint_path":row["checkpoint_path"],"checkpoint_sha256":row["checkpoint_sha256"],"metrics_path":row["metrics_path"],"metrics":{k:v for k,v in row.items() if k not in ("candidate_id","run_id","config_hash","status","checkpoint_path","checkpoint_sha256","metrics_path")},"candidate_manifest_sha256":master["candidate_manifests"][task]["sha256"],"lineage":{"g1_sha256":master["g1_sha256"],"phase14_handoff_sha256":master["phase14_handoff_sha256"],"feature_schema_sha256":master["feature_schema_sha256"],"split_sha256":master["split_sha256"],"preprocessor_sha256":master["preprocessor_sha256"]}}
        best_path=SEARCH_ROOT/"best_gru_candidates_v1.json"; _write_json(best_path,best)
        ledger=SEARCH_ROOT/"retry_failure_ledger_v1.json"; _write_json(ledger,{"status":"COMPLETE","max_attempts":MAX_ATTEMPTS,"tasks":{task:{"complete":sum(r["status"]=="COMPLETE" for r in value["rows"]),"failed":sum(r["status"]=="FAILED" for r in value["rows"])} for task,value in summaries.items()},"test_accessed":False})
        master.update(status="COMPLETE",completed_at_utc=_utc(),terminal_candidate_counts={task:{"complete":sum(r["status"]=="COMPLETE" for r in value["rows"]),"failed":sum(r["status"]=="FAILED" for r in value["rows"])} for task,value in summaries.items()},best_gru_manifest_path=str(best_path.relative_to(ROOT)),best_gru_manifest_sha256=sha256_file(best_path),reproducibility_audit_path=str(audit_path.relative_to(ROOT)),reproducibility_audit_sha256=sha256_file(audit_path))
        _write_json(MASTER_PATH,master)
    except Exception:
        master.update(status="FAILED_WITH_RECORDED_CANDIDATES",completed_at_utc=_utc()); _write_json(MASTER_PATH,master); raise
    print(json.dumps({"status":"COMPLETE","terminal_candidate_counts":master["terminal_candidate_counts"],"best":{task:value["best"]["candidate_id"] for task,value in summaries.items()},"test_accessed":False},sort_keys=True))


if __name__=="__main__": main()
