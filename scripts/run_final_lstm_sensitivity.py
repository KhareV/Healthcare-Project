#!/usr/bin/env python3
"""Run exactly one fixed, validation-only LSTM sensitivity model per task."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import torch

from data.collate import create_dataloader
from data.gru_canonical import Phase10GRUDataset, TaskEligibleGRUDataset
from data.synthetic.config import canonical_json_bytes
from data.synthetic.provenance import sha256_file
from experiments.gru_final import validate_checkpoint_lineage
from experiments.lineage import ArtifactRecord, read_artifact_index, read_run_registry, validate_artifact_lineage, write_artifact_index
from experiments.stage2_finalization import STAGE2_VERSION, TASKS, canonical_sha256, derive_lstm_config, validate_exact_run_set, validate_lstm_config
from models.gru import GRUEncoderConfig
from models.gru_icu_time import ICU_TIME_RAW_OUTPUT, ICUTimeGRU
from models.gru_recovery import RECOVERY_HORIZON_ORDER, RecoveryGRU
from models.gru_support import SUPPORT_OUTPUT_TYPE, SUPPORT_PROBABILITY_TRANSFORM, OrganSupportGRU
from models.icu_time_postprocess import ICU_TIME_POSTPROCESS_VERSION
from models.lstm import ICUTimeLSTM, OrganSupportLSTM, RecoveryLSTM, trainable_parameter_count
from preprocess.target_scaler import RecoveryTargetScaler
from training.checkpoint import load_checkpoint, save_checkpoint
from training.class_weights import SupportClassWeight
from training.engine import create_optimizer, run_epoch
from training.logging import JsonlRunLogger
from training.reproducibility import set_deterministic_seed
from training.tasks.icu_time import ICUTimeTaskAdapter
from training.tasks.organ_support import OrganSupportTaskAdapter
from training.tasks.recovery import RecoveryTaskAdapter
from vedant_infra.registry import make_registry_record, upsert_registry_record

# Reuse the final-GRU evaluation/prediction implementation verbatim.
from scripts.run_final_gru_search import _epoch_key, _metrics_and_rows, _predict, _write_jsonl


SELECTION_PATH = ROOT / "artifacts/selection/validation_family_selection_v1.json"
GRU_BEST_PATH = ROOT / "artifacts/search/gru/final_v2/best_gru_candidates_v1.json"
GRU_MASTER_PATH = ROOT / "artifacts/search/gru/final_v2/manifests/search_manifest_v1.json"
OUTPUT_PATH = ROOT / "artifacts/models/lstm_sensitivity_v1.json"
RUN_ROOT = ROOT / "artifacts/models/lstm_sensitivity_v1"
REGISTRY_PATH = ROOT / "experiments/registry.csv"
ARTIFACT_INDEX_PATH = ROOT / "experiments/artifacts.csv"
TASK_REGISTRY = {"recovery": "recovery", "icu_time": "icu_stay_time", "organ_support": "organ_support"}
OBJECTIVES = {
    "recovery": "MIN_STAY_BALANCED_MAE24_ORIGINAL_DELTA_SOFA_UNITS",
    "icu_time": "MIN_STAY_BALANCED_WEIGHTED_MEDIAN_ABSOLUTE_ERROR_HOURS",
    "organ_support": "MAX_STAY_BALANCED_AUPRC_RAW_UNCALIBRATED",
}


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp"); temp.write_bytes(canonical_json_bytes(value)); temp.replace(path)


def write_exact_config(path, value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    if canonical_sha256(value) != __import__("hashlib").sha256(raw).hexdigest():
        raise RuntimeError("LSTM config serialization/hash mismatch")
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp"); temp.write_bytes(raw); temp.replace(path)


def git(*args):
    return subprocess.check_output(("git",) + args, cwd=ROOT, text=True).strip()


def utc():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def ensure_execution_state(selection, science_commit):
    head = git("rev-parse", "HEAD")
    if selection["implementation_commit"] != science_commit:
        raise RuntimeError("Stage-2 science implementation differs from the family freeze")
    if subprocess.run(("git", "merge-base", "--is-ancestor", science_commit, head), cwd=ROOT).returncode:
        raise RuntimeError("Stage-2 science implementation is not an ancestor of HEAD")
    if subprocess.run(("git", "diff", "--quiet", head, "--", "src", "scripts", "configs", "tests"), cwd=ROOT).returncode:
        raise RuntimeError("science-affecting implementation is dirty")
    if selection.get("status") != "VALIDATION_FAMILY_SELECTION_FROZEN" or selection.get("frozen_before_lstm_interpretation") is not True:
        raise RuntimeError("family selection must be frozen before LSTM execution")
    if selection.get("test_accessed") is not False:
        raise RuntimeError("family selection violated the absolute test lock")
    forbidden = (ROOT / "artifacts/models/selected_models_v1.json", ROOT / "artifacts/governance/g3_freeze.json",
                 ROOT / "artifacts/acceptance/g3_model_selection_freeze_v1.json")
    if any(path.exists() for path in forbidden):
        raise RuntimeError("selected-model/G3 artifact is forbidden in Stage 2")


def model_pair(task, config, master):
    encoder = GRUEncoderConfig(feature_dim=master["dimensions"]["F"], hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"], dropout=config["dropout"], include_observation_mask=True,
        include_tslo=True, static_dim=master["dimensions"]["S_model"], bidirectional=False)
    if task == "recovery": return RecoveryLSTM(encoder), RecoveryGRU(encoder)
    if task == "icu_time": return ICUTimeLSTM(encoder), ICUTimeGRU(encoder)
    return OrganSupportLSTM(encoder), OrganSupportGRU(encoder)


def metadata(task, run_id, seed, config_hash, master, model, primary, parent, science_commit):
    value = {
        "run_id": run_id, "task": TASK_REGISTRY[task], "model_family": "lstm",
        "model_config": model.model_config, "seed": seed, "epoch": 0, "config_hash": config_hash,
        "tensor_contract_version": master["tensor_contract_version"],
        "feature_schema_version": master["feature_schema_version"], "split_hash": master["split_sha256"],
        "code_commit": science_commit, "validation_value": float(primary),
        "synthetic_smoke_test": False, "synthetic_scientific_search": False,
        "synthetic_scientific_sensitivity": True, "sensitivity_only": True,
        "participates_in_search": False, "parent_best_gru_candidate_id": parent["candidate_id"],
        "parent_gru_config_hash": parent["config_hash"], "g1_sha256": master["g1_sha256"],
        "phase14_handoff_sha256": master["phase14_handoff_sha256"],
        "feature_schema_sha256": master["feature_schema_sha256"],
        "preprocessor_sha256": master["preprocessor_sha256"], "test_accessed": False,
    }
    if task == "recovery":
        value.update(horizon_order=list(RECOVERY_HORIZON_ORDER),
            target_scaler_artifact="artifacts/preprocessors/recovery_target_scaler_synthetic_v1.json",
            target_scaler_sha256=master["recovery_target_scaler_sha256"], target_scaler_fit_calls=0)
    elif task == "icu_time":
        value.update(raw_output_meaning=ICU_TIME_RAW_OUTPUT, postprocess_version=ICU_TIME_POSTPROCESS_VERSION,
                     target_scaler=None)
    else:
        weight = SupportClassWeight.load(ROOT / "artifacts/preprocessors/support_class_weight_synthetic_v1.json")
        value.update(output_type=SUPPORT_OUTPUT_TYPE, probability_transform=SUPPORT_PROBABILITY_TRANSFORM,
            class_weight_artifact="artifacts/preprocessors/support_class_weight_synthetic_v1.json",
            class_weight_sha256=master["support_class_weight_sha256"],
            label_contract_version=weight.label_contract_version, event_dictionary_version=weight.event_dictionary_version,
            calibration=None, operating_threshold=None)
    return value


def train_one(task, parent, master, train, validation, directory, science_commit):
    source_config = load(ROOT / parent["checkpoint_path"].replace("model.pt", "candidate_config.json"))
    config = derive_lstm_config(source_config); validate_lstm_config(source_config, config)
    config_hash = canonical_sha256(config); config_path = directory / "config.json"; write_exact_config(config_path, config)
    source_metrics = load(ROOT / parent["metrics_path"])
    seed = source_metrics["model_seed"]
    run_id = "stage2-lstm-" + ("support" if task == "organ_support" else task.replace("_", "-")) + "-001"
    set_deterministic_seed(seed); torch.set_num_threads(1); device = torch.device("cpu")
    task_train = TaskEligibleGRUDataset(train, task)
    train_loader = create_dataloader(task_train, batch_size=config["batch_size"], shuffle=True, seed=seed, num_workers=0)
    validation_loader = create_dataloader(validation, batch_size=config["batch_size"], shuffle=False, seed=seed, num_workers=0)
    model, gru_model = model_pair(task, config, master); model.to(device)
    optimizer = create_optimizer(model.parameters(), name="adamw", learning_rate=config["learning_rate"], weight_decay=config["weight_decay"])
    scaler = RecoveryTargetScaler.load(ROOT / "artifacts/preprocessors/recovery_target_scaler_synthetic_v1.json")
    weight = SupportClassWeight.load(ROOT / "artifacts/preprocessors/support_class_weight_synthetic_v1.json")
    adapter = RecoveryTaskAdapter(scaler) if task == "recovery" else (ICUTimeTaskAdapter() if task == "icu_time" else OrganSupportTaskAdapter(weight.pos_weight))
    logger = JsonlRunLogger(directory / "training_log.jsonl"); checkpoint = directory / "model.pt"
    best_key = best_metrics = best_epoch = None; bad = 0; started = time.monotonic()
    for epoch in range(1, config["max_epochs"] + 1):
        training = run_epoch(model, train_loader, adapter, device, optimizer=optimizer)
        predictions = _predict(task, model, validation_loader, validation, device, scaler)
        metrics, _ = _metrics_and_rows(task, validation, predictions); key = _epoch_key(task, metrics, epoch)
        improved = best_key is None or key < best_key
        logger.log({"event": "epoch", "run_id": run_id, "epoch": epoch, "train_loss": training.mean_loss,
                    "validation_metrics": metrics, "improved": improved})
        if improved:
            best_key, best_metrics, best_epoch, bad = key, metrics, epoch, 0
            primary = metrics["mae24"] if task == "recovery" else (metrics["median_absolute_error_hours"] if task == "icu_time" else metrics["auprc"])
            meta = metadata(task, run_id, seed, config_hash, master, model, primary, parent, science_commit); meta["epoch"] = epoch
            save_checkpoint(checkpoint, model, optimizer, meta)
        else:
            bad += 1
        if bad >= config["early_stopping_patience"]:
            break
    validate_checkpoint_lineage(checkpoint, {"g1_sha256": master["g1_sha256"],
        "phase14_handoff_sha256": master["phase14_handoff_sha256"], "preprocessor_sha256": master["preprocessor_sha256"],
        "config_hash": config_hash, "seed": seed, "parent_best_gru_candidate_id": parent["candidate_id"]})
    fresh, _ = model_pair(task, config, master)
    sidecar = load_checkpoint(checkpoint, fresh, expected_tensor_contract_version=master["tensor_contract_version"],
                              expected_feature_schema_version=master["feature_schema_version"])
    fresh.to(device)
    final_predictions = _predict(task, fresh, validation_loader, validation, device, scaler)
    final_metrics, rows = _metrics_and_rows(task, validation, final_predictions)
    if canonical_sha256(final_metrics) != canonical_sha256(best_metrics):
        raise RuntimeError("restored best LSTM metrics differ")
    predictions_path = directory / "validation_predictions.jsonl"; _write_jsonl(predictions_path, rows)
    result = {
        "task": task, "family": "lstm", "run_id": run_id, "status": "SENSITIVITY_ONLY",
        "sensitivity_only": True, "participates_in_search": False, "parent_best_gru_candidate_id": parent["candidate_id"],
        "parent_gru_config_hash": parent["config_hash"], "config_path": str(config_path.relative_to(ROOT)),
        "config_hash": config_hash, "seed": seed, "best_epoch": best_epoch, "epochs_completed": epoch,
        "validation_metrics": final_metrics, "checkpoint_path": str(checkpoint.relative_to(ROOT)),
        "checkpoint_sha256": sidecar["checkpoint_sha256"], "prediction_path": str(predictions_path.relative_to(ROOT)),
        "prediction_sha256": sha256_file(predictions_path), "gru_parameter_count": trainable_parameter_count(gru_model),
        "lstm_parameter_count": trainable_parameter_count(fresh), "runtime_seconds": time.monotonic() - started,
        "probability_type": "raw_uncalibrated" if task == "organ_support" else None,
        "support_calibrated": False, "support_threshold": None, "test_accessed": False,
    }
    metrics_path = directory / "validation_metrics.json"; write(metrics_path, result)
    result["metrics_path"] = str(metrics_path.relative_to(ROOT)); result["metrics_sha256"] = sha256_file(metrics_path)
    return result


def register(result, selection, master, parent, commit):
    run_id, task = result["run_id"], result["task"]
    record = make_registry_record(run_id=run_id, timestamp_utc=utc(), task=TASK_REGISTRY[task], model_family="lstm",
        seed=str(result["seed"]), code_commit=commit, config_ref=result["config_path"], config_hash=result["config_hash"],
        split_hash=master["split_sha256"], feature_version=master["feature_schema_version"],
        label_version="synthetic_phase9_final_target_contract_v1", model_artifact_ref=result["checkpoint_path"],
        model_sha256=result["checkpoint_sha256"], metrics_ref=result["metrics_path"], metrics_sha256=result["metrics_sha256"],
        status="completed", parent_run_id=parent["run_id"], notes=json.dumps({"role": "SENSITIVITY_ONLY",
        "parent_best_gru_candidate_id": parent["candidate_id"], "participates_in_search": False,
        "selection_frozen_sha256": sha256_file(SELECTION_PATH), "test_accessed": False}, sort_keys=True, separators=(",", ":")),
        candidate_id=result["run_id"], search_version=STAGE2_VERSION, candidate_list_hash=canonical_sha256(["one_fixed_lstm_per_task"]),
        attempt_number="1", attempt_status_detail="COMPLETE", validation_objective=OBJECTIVES[task],
        run_type="scientific_sensitivity", preprocessor_ref="artifacts/preprocessors/synthetic_feature_preprocessor_v1.json",
        preprocessor_sha256=master["preprocessor_sha256"], derived_feature_hash=master["feature_schema_sha256"],
        environment_ref=master["environment_path"], environment_sha256=master["environment_sha256"], dirty_worktree="false")
    upsert_registry_record(REGISTRY_PATH, record)
    common = dict(producing_run_id=run_id, task=TASK_REGISTRY[task], model_family="lstm", split_hash=master["split_sha256"],
        feature_version=master["feature_schema_version"], label_version="synthetic_phase9_final_target_contract_v1",
        config_hash=result["config_hash"], creation_commit=commit, generating_script="scripts/run_final_lstm_sensitivity.py",
        creation_date_utc=utc(), preprocessor_sha256=master["preprocessor_sha256"], environment_sha256=master["environment_sha256"],
        run_type="scientific_sensitivity", status="registered")
    model_id, pred_id = run_id + ":model", run_id + ":predictions"
    records = [
        ArtifactRecord(artifact_id=model_id, artifact_path=result["checkpoint_path"], artifact_type="model_checkpoint",
            artifact_version="fixed_lstm_sensitivity_checkpoint_v1", artifact_sha256=result["checkpoint_sha256"],
            model_sha256=result["checkpoint_sha256"], metadata_ref=result["checkpoint_path"] + ".metadata.json", **common),
        ArtifactRecord(artifact_id=pred_id, artifact_path=result["prediction_path"], artifact_type="prediction",
            artifact_version="fixed_lstm_validation_prediction_v1", artifact_sha256=result["prediction_sha256"],
            parent_artifact_ids=model_id, model_sha256=result["checkpoint_sha256"], partition="validation",
            probability_type="raw_uncalibrated" if task == "organ_support" else "",
            prediction_population_hash=master["validation_row_keys_sha256"], **common),
        ArtifactRecord(artifact_id=run_id + ":metrics", artifact_path=result["metrics_path"], artifact_type="metric_table",
            artifact_version="fixed_lstm_validation_metrics_v1", artifact_sha256=result["metrics_sha256"],
            parent_artifact_ids=pred_id, partition="validation", probability_type="raw_uncalibrated" if task == "organ_support" else "",
            evaluator_version="stay_balanced_metrics_v1", metric_implementation_version="stay_balanced_metrics_v1", **common),
    ]
    combined = list(read_artifact_index(ARTIFACT_INDEX_PATH)) + records
    validate_artifact_lineage(combined, read_run_registry(REGISTRY_PATH), repository_root=ROOT)
    write_artifact_index(ARTIFACT_INDEX_PATH, combined)


def main():
    if OUTPUT_PATH.exists():
        raise RuntimeError("LSTM sensitivity is immutable and cannot be rerun")
    selection, best, master = load(SELECTION_PATH), load(GRU_BEST_PATH), load(GRU_MASTER_PATH)
    commit = selection["implementation_commit"]; ensure_execution_state(selection, commit)
    for path, expected in ((ROOT / master["g1_path"], master["g1_sha256"]),
                           (ROOT / master["phase14_handoff_path"], master["phase14_handoff_sha256"]),
                           (GRU_BEST_PATH, master["best_gru_manifest_sha256"])):
        if sha256_file(path) != expected:
            raise RuntimeError("frozen LSTM parent changed: " + str(path.relative_to(ROOT)))
    train, validation = Phase10GRUDataset(ROOT, "train"), Phase10GRUDataset(ROOT, "validation")
    entries = []
    for task in TASKS:
        directory = RUN_ROOT / task
        metrics_path = directory / "validation_metrics.json"
        if directory.exists():
            if not metrics_path.is_file():
                raise RuntimeError("partial LSTM task directory cannot be resumed: " + task)
            result = load(metrics_path)
            result["metrics_path"] = str(metrics_path.relative_to(ROOT))
            result["metrics_sha256"] = sha256_file(metrics_path)
            if (sha256_file(ROOT / result["checkpoint_path"]) != result["checkpoint_sha256"]
                    or sha256_file(ROOT / result["prediction_path"]) != result["prediction_sha256"]):
                raise RuntimeError("resumable LSTM artifacts changed: " + task)
        else:
            directory.mkdir(parents=True, exist_ok=False)
            result = train_one(task, best["tasks"][task], master, train, validation, directory, commit)
        register(result, selection, master, best["tasks"][task], commit); entries.append(result)
        print(json.dumps({"task": task, "status": "SENSITIVITY_ONLY", "best_epoch": result["best_epoch"]}), flush=True)
    validate_exact_run_set(entries)
    comparison = {}
    for result in entries:
        task = result["task"]; gru_metrics = best["tasks"][task]["metrics"]; lstm_metrics = result["validation_metrics"]
        metric = {"recovery": "mae24", "icu_time": "median_absolute_error_hours", "organ_support": "auprc"}[task]
        comparison[task] = {"metric": metric, "best_gru_value": gru_metrics[metric], "fixed_lstm_value": lstm_metrics[metric],
            "signed_lstm_minus_gru": lstm_metrics[metric] - gru_metrics[metric],
            "gru_parameter_count": result["gru_parameter_count"], "lstm_parameter_count": result["lstm_parameter_count"],
            "descriptive_only": True, "family_selection_unchanged": True}
    manifest = {"manifest_version": "lstm_sensitivity_v1", "stage_version": STAGE2_VERSION,
        "status": "SENSITIVITY_ONLY", "statuses": ["SENSITIVITY_ONLY", "NO_LSTM_SEARCH", "NOT_SERVING_SELECTION", "TEST_NOT_ACCESSED"],
        "run_count": 3, "runs": entries, "comparison_to_best_gru": comparison,
        "family_selection_path": str(SELECTION_PATH.relative_to(ROOT)), "family_selection_sha256": sha256_file(SELECTION_PATH),
        "family_selection_frozen_before_runs": True, "implementation_commit": commit,
        "g1_sha256": master["g1_sha256"], "phase14_handoff_sha256": master["phase14_handoff_sha256"],
        "split_sha256": master["split_sha256"], "feature_schema_sha256": master["feature_schema_sha256"],
        "preprocessor_sha256": master["preprocessor_sha256"], "support_calibrated": False,
        "support_threshold": None, "selected_models_v1_created": False, "g3_created": False, "test_accessed": False}
    write(OUTPUT_PATH, manifest)
    print(json.dumps({"status": "COMPLETE", "run_count": 3, "manifest_sha256": sha256_file(OUTPUT_PATH),
                      "test_accessed": False}, sort_keys=True))


if __name__ == "__main__":
    main()
