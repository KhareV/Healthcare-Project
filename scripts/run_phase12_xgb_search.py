#!/usr/bin/env python3
"""Execute the frozen Phase-12 validation-only XGBoost scientific search."""
from __future__ import annotations

from dataclasses import asdict
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

from data.synthetic.config import canonical_json_bytes
from data.synthetic.provenance import sha256_file
from data.xgb_canonical import Phase10XGBData, row_keys_hash
from evaluation.metrics import (
    METRIC_IMPLEMENTATION_VERSION,
    PredictionRecord,
    evaluate_icu_time,
    evaluate_organ_support,
    evaluate_recovery,
)
from evaluation.weighted_stats import weighted_median
from evaluation.weights import compute_stay_weights
from experiments.lineage import (
    ArtifactRecord,
    read_artifact_index,
    read_run_registry,
    validate_artifact_lineage,
    write_artifact_index,
)
from experiments.search_governance import GovernanceError
from experiments.search_registry import record_search_attempt
from experiments.xgb_phase12 import (
    MAX_ATTEMPTS,
    RankedCandidate,
    SEARCH_VERSION,
    TASKS,
    assert_lifecycle_transition,
    canonical_sha256,
    choose_best,
    validate_candidate_manifest,
    validate_master_manifest,
    validate_search_space,
)
from models.icu_time_postprocess import remaining_icu_hours_from_log_prediction
from models.xgb_canonical import (
    RUN_PHASE12_SCIENTIFIC,
    XGBCandidateConfig,
    XGBLineage,
    train_xgb_candidate,
)


SEARCH_ROOT = ROOT / "artifacts/search/xgb/phase12"
MANIFEST_ROOT = SEARCH_ROOT / "manifests"
MASTER_PATH = MANIFEST_ROOT / "phase12_search_manifest_v1.json"
REGISTRY_PATH = ROOT / "experiments/registry.csv"
ARTIFACT_INDEX_PATH = ROOT / "experiments/artifacts.csv"
REGISTRY_TASK = {"recovery": "recovery", "icu_time": "icu_stay_time", "organ_support": "organ_support"}


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(canonical_json_bytes(value))
    temporary.replace(path)


def _write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        for row in rows:
            handle.write(canonical_json_bytes(row) + b"\n")
    temporary.replace(path)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _git(*args: str) -> str:
    return subprocess.check_output(("git",) + args, cwd=ROOT, text=True).strip()


def _ensure_code_unchanged(execution_commit: str) -> None:
    if _git("rev-parse", "HEAD") != execution_commit:
        raise GovernanceError("search execution commit changed mid-search")
    result = subprocess.run(
        ("git", "diff", "--quiet", execution_commit, "--", "src", "scripts", "configs", "tests"),
        cwd=ROOT,
    )
    if result.returncode != 0:
        raise GovernanceError("science-affecting code/config changed mid-search")


def _manifest_for_task(master, task):
    item = master["candidate_manifests"][task]
    return _load(ROOT / item["path"])


def _verify_frozen(master, space, phase11, execution_commit):
    validate_master_manifest(ROOT, master)
    validate_search_space(space)
    if canonical_sha256(space) != master["search_space_sha256"]:
        raise GovernanceError("search-space canonical hash changed")
    if sha256_file(ROOT / master["phase11_manifest_path"]) != master["phase11_manifest_sha256"]:
        raise GovernanceError("Phase-11 manifest changed")
    checks = {
        "split_sha256": phase11["split_sha256"],
        "feature_schema_sha256": phase11["feature_schema_sha256"],
        "flat_feature_map_sha256": phase11["flat_feature_map_sha256"],
        "preprocessor_sha256": phase11["preprocessor_sha256"],
        "target_contract_sha256": phase11["target_contract_sha256"],
        "recovery_target_scaler_sha256": phase11["recovery_target_scaler_sha256"],
        "support_class_weight_sha256": phase11["support_class_weight_sha256"],
    }
    if any(master[key] != value for key, value in checks.items()):
        raise GovernanceError("Phase-10/11 lineage changed during search")
    if sha256_file(ROOT / master["environment_path"]) != master["environment_sha256"]:
        raise GovernanceError("Phase-12 scientific environment identity changed")
    _ensure_code_unchanged(execution_commit)


def _lineage(master, phase11, execution_commit):
    preprocessor = _load(ROOT / phase11["preprocessor_path"])
    split_meta = _load(ROOT / "artifacts/splits/synthetic_split_v2.metadata.json")
    target_contract = _load(ROOT / phase11["target_contract_path"])
    return XGBLineage(
        feature_schema_version=phase11["feature_schema_version"],
        feature_schema_sha256=phase11["feature_schema_sha256"],
        flat_feature_map_version=phase11["flat_feature_map_version"],
        flat_feature_map_sha256=phase11["flat_feature_map_sha256"],
        preprocessor_version=preprocessor["artifact_version"],
        preprocessor_sha256=phase11["preprocessor_sha256"],
        split_version=split_meta["artifact_version"],
        split_sha256=phase11["split_sha256"],
        target_schema_version=target_contract["contract_version"],
        target_schema_sha256=phase11["target_contract_sha256"],
        recovery_target_scaler_sha256=phase11["recovery_target_scaler_sha256"],
        support_class_weight_sha256=phase11["support_class_weight_sha256"],
        event_dictionary_sha256=phase11["event_dictionary_sha256"],
        code_commit=execution_commit,
    )


def _xgb_config(candidate):
    config = candidate["config"]
    return XGBCandidateConfig(
        max_depth=config["max_depth"], learning_rate=config["learning_rate"],
        subsample=config["subsample"], colsample_bytree=config["colsample_bytree"],
        min_child_weight=config["min_child_weight"], reg_alpha=config["reg_alpha"],
        reg_lambda=config["reg_lambda"], n_estimators=config["n_estimators"],
    )


def _prediction_record(key, target, prediction, eligible):
    return PredictionRecord(
        stay_id=key[1], prediction_time=key[2], target=float(target) if eligible else math.nan,
        prediction=float(prediction), eligible=bool(eligible),
    )


def _recovery_metrics(validation, predictions):
    records24 = tuple(
        _prediction_record(key, target, prediction, eligible)
        for key, target, prediction, eligible in zip(
            validation.row_keys, validation.targets["recovery24"], predictions[:, 0], validation.eligibility["recovery24"]
        )
    )
    records48 = tuple(
        _prediction_record(key, target, prediction, eligible)
        for key, target, prediction, eligible in zip(
            validation.row_keys, validation.targets["recovery48"], predictions[:, 1], validation.eligibility["recovery48"]
        )
    )
    evaluated = evaluate_recovery(records24, records48, metadata={"model_family": "xgboost", "partition": "validation"})
    result = {
        "mae24": evaluated["24h"].metrics["mae"], "mae48": evaluated["48h"].metrics["mae"],
        "rmse24": evaluated["24h"].metrics["rmse"], "rmse48": evaluated["48h"].metrics["rmse"],
        "directional_accuracy24": evaluated["24h"].metrics["directional_agreement"],
        "directional_accuracy48": evaluated["48h"].metrics["directional_agreement"],
    }
    for name, records in (("24", records24), ("48", records48)):
        weights = compute_stay_weights([row.stay_id for row in records], [row.eligible for row in records])
        eligible = [records[index] for index in weights.eligible_indices]
        selected_weights = [weights.weights[index] for index in weights.eligible_indices]
        result["median_absolute_error" + name] = weighted_median(
            [abs(row.prediction - row.target) for row in eligible], selected_weights
        )
    return result, records24, records48, evaluated


def _icu_metrics(validation, raw_predictions, hour_predictions):
    true_hours = np.expm1(validation.targets["icu_stay_time"].astype(np.float64))
    records = tuple(
        _prediction_record(key, target, prediction, eligible)
        for key, target, prediction, eligible in zip(
            validation.row_keys, true_hours, hour_predictions, validation.eligibility["icu_stay_time"]
        )
    )
    evaluated = evaluate_icu_time(records, metadata={"model_family": "xgboost", "partition": "validation"})
    return {
        "median_absolute_error_hours": evaluated.metrics["median_absolute_error"],
        "mae_hours": evaluated.metrics["mae"], "rmse_hours": evaluated.metrics["rmse"],
    }, records, true_hours, evaluated


def _support_metrics(validation, probabilities):
    records = tuple(
        _prediction_record(key, target, probability, eligible)
        for key, target, probability, eligible in zip(
            validation.row_keys, validation.targets["organ_support"], probabilities, validation.eligibility["organ_support"]
        )
    )
    evaluated = evaluate_organ_support(
        records, probability_type="raw", metadata={"model_family": "xgboost", "partition": "validation"}
    )
    return {
        "auprc": evaluated.metrics["auprc"], "auroc": evaluated.metrics["auroc"],
        "brier": evaluated.metrics["brier"],
    }, records, evaluated


def _identity(key):
    return {"subject_id": key[0], "stay_id": key[1], "prediction_time": key[2], "grid_index": key[3]}


def _persist_predictions(task, candidate_id, attempt_dir, validation, bundle):
    if task == "recovery":
        predictions = bundle.predict(validation.X)
        if predictions.shape != (len(validation.row_keys), 2) or not np.isfinite(predictions).all():
            raise GovernanceError("recovery validation prediction shape/finite check failed")
        metrics, records24, records48, evaluated = _recovery_metrics(validation, predictions)
        paths = []
        for view, column, records in (("24h", 0, records24), ("48h", 1, records48)):
            eligibility = validation.eligibility["recovery" + view[:2]]
            targets = validation.targets["recovery" + view[:2]]
            path = attempt_dir / f"validation_predictions_{view}.jsonl"
            _write_jsonl(path, (
                {**_identity(key), "candidate_id": candidate_id, "task": "recovery", "horizon": view,
                 "eligible": bool(ok), "true_target": float(target) if ok else None,
                 "prediction": float(prediction), "prediction_units": "sofa_delta"}
                for key, ok, target, prediction in zip(validation.row_keys, eligibility, targets, predictions[:, column])
            ))
            paths.append(path)
        evaluation_detail = {key: {"counts": value.counts, "metadata": value.metadata} for key, value in evaluated.items()}
        return metrics, paths, evaluation_detail
    if task == "icu_time":
        raw = bundle.predict_log(validation.X)
        hours = remaining_icu_hours_from_log_prediction(torch.from_numpy(raw)).numpy()
        if raw.shape != (len(validation.row_keys),) or not np.isfinite(raw).all() or not np.isfinite(hours).all():
            raise GovernanceError("ICU validation prediction shape/finite check failed")
        metrics, records, true_hours, evaluated = _icu_metrics(validation, raw, hours)
        path = attempt_dir / "validation_predictions.jsonl"
        _write_jsonl(path, (
            {**_identity(key), "candidate_id": candidate_id, "task": "icu_time", "eligible": bool(ok),
             "true_target_log1p": float(target_log) if ok else None, "prediction_log1p": float(pred_log),
             "true_target": float(target_hours) if ok else None, "prediction": float(pred_hours),
             "prediction_units": "remaining_current_episode_hours"}
            for key, ok, target_log, pred_log, target_hours, pred_hours in zip(
                validation.row_keys, validation.eligibility["icu_stay_time"], validation.targets["icu_stay_time"], raw, true_hours, hours
            )
        ))
        return metrics, [path], {"counts": evaluated.counts, "metadata": evaluated.metadata}
    probabilities = bundle.predict(validation.X)
    if probabilities.shape != (len(validation.row_keys),) or not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
        raise GovernanceError("support validation probability check failed")
    metrics, records, evaluated = _support_metrics(validation, probabilities)
    path = attempt_dir / "validation_predictions.jsonl"
    _write_jsonl(path, (
        {**_identity(key), "candidate_id": candidate_id, "task": "organ_support", "eligible": bool(ok),
         "true_target": int(target) if ok else None, "prediction": float(probability),
         "probability_type": "raw_uncalibrated", "calibrated": False}
        for key, ok, target, probability in zip(
            validation.row_keys, validation.eligibility["organ_support"], validation.targets["organ_support"], probabilities
        )
    ))
    return metrics, [path], {"counts": evaluated.counts, "metadata": evaluated.metadata}


def _enrich_bundle_metadata(path, *, master, candidate, execution_commit):
    payload = _load(path)
    payload["phase12_search"] = {
        "search_version": SEARCH_VERSION,
        "candidate_config_sha256": candidate["config_hash"],
        "candidate_list_hash": master["candidate_manifests"][payload["task"] if payload["task"] != "icu_stay_time" else "icu_time"]["candidate_list_hash"] if payload["task"] != "organ_support" else master["candidate_manifests"]["organ_support"]["candidate_list_hash"],
        "execution_commit": execution_commit,
        "validation_only": True, "test_accessed": False,
    }
    _write_json(path, payload)
    return payload


def _artifact_common(task, run_id, candidate, master, lineage, execution_commit, created):
    return dict(
        producing_run_id=run_id, task=REGISTRY_TASK[task], model_family="xgboost",
        split_hash=master["split_sha256"], feature_version=lineage.feature_schema_version,
        label_version=lineage.target_schema_version, config_hash=candidate["config_hash"],
        creation_commit=execution_commit, generating_script="scripts/run_phase12_xgb_search.py",
        creation_date_utc=created, preprocessor_sha256=master["preprocessor_sha256"],
        environment_sha256=master["environment_sha256"], run_type="scientific", status="registered",
    )


def _register_artifacts(records):
    existing = list(read_artifact_index(ARTIFACT_INDEX_PATH))
    candidate = existing + list(records)
    runs = read_run_registry(REGISTRY_PATH)
    validate_artifact_lineage(candidate, runs, repository_root=ROOT)
    write_artifact_index(ARTIFACT_INDEX_PATH, candidate)


def _artifact_records(task, run_id, candidate, attempt_dir, bundle_meta, prediction_paths, metrics_path, master, lineage, execution_commit, created):
    common = _artifact_common(task, run_id, candidate, master, lineage, execution_commit, created)
    prefix = run_id
    records = []
    model_ids = []
    for view, entry in bundle_meta["models"].items():
        path = attempt_dir / "bundle" / entry["path"]
        artifact_id = prefix + ":model:" + view
        model_ids.append(artifact_id)
        records.append(ArtifactRecord(
            artifact_id=artifact_id, artifact_path=str(path.relative_to(ROOT)),
            artifact_type="model_checkpoint", artifact_version="xgb_native_json_v1",
            artifact_sha256=sha256_file(path), model_sha256=sha256_file(path),
            metadata_ref=str((attempt_dir / "bundle/bundle.metadata.json").relative_to(ROOT)), **common,
        ))
    bundle_path = attempt_dir / "bundle/bundle.metadata.json"
    records.append(ArtifactRecord(
        artifact_id=prefix + ":bundle", artifact_path=str(bundle_path.relative_to(ROOT)),
        artifact_type="model_bundle_metadata", artifact_version="canonical_xgb_candidate_bundle_v1",
        artifact_sha256=sha256_file(bundle_path), parent_artifact_ids=";".join(model_ids), **common,
    ))
    prediction_ids = []
    for index, path in enumerate(prediction_paths):
        model_id = model_ids[index] if task == "recovery" else model_ids[0]
        artifact_id = prefix + ":prediction:" + str(index + 1)
        prediction_ids.append(artifact_id)
        records.append(ArtifactRecord(
            artifact_id=artifact_id, artifact_path=str(path.relative_to(ROOT)),
            artifact_type="prediction", artifact_version="phase12_validation_prediction_v1",
            artifact_sha256=sha256_file(path), parent_artifact_ids=model_id,
            model_sha256=next(item.artifact_sha256 for item in records if item.artifact_id == model_id),
            partition="validation", probability_type="raw_uncalibrated" if task == "organ_support" else "",
            prediction_population_hash=master["validation_row_keys_sha256"], **common,
        ))
    records.append(ArtifactRecord(
        artifact_id=prefix + ":metrics", artifact_path=str(metrics_path.relative_to(ROOT)),
        artifact_type="metric_table", artifact_version="phase12_validation_metrics_v1",
        artifact_sha256=sha256_file(metrics_path), parent_artifact_ids=";".join(prediction_ids),
        partition="validation", probability_type="raw_uncalibrated" if task == "organ_support" else "",
        evaluator_version=METRIC_IMPLEMENTATION_VERSION,
        metric_implementation_version=METRIC_IMPLEMENTATION_VERSION, **common,
    ))
    return records


def _task_matrices(gateway, train, validation, task):
    views = ("recovery24", "recovery48") if task == "recovery" else (("icu_stay_time",) if task == "icu_time" else ("organ_support",))
    return (
        {view: gateway.task_matrix(train, view) for view in views},
        {view: gateway.task_matrix(validation, view) for view in views},
    )


def _run_candidate(task, manifest, candidate, attempt, gateway, train, validation, master, phase11, lineage, execution_commit, retry_of=""):
    candidate_id = candidate["candidate_id"]
    run_id = f"phase12-{candidate_id}-attempt-{attempt}"
    attempt_dir = SEARCH_ROOT / task / candidate_id / f"attempt-{attempt}"
    attempt_dir.mkdir(parents=True, exist_ok=False)
    config_path = attempt_dir / "candidate_config.json"
    config_path.write_bytes(canonical_json_bytes(candidate["config"]))
    started = _utc_now(); start = time.monotonic()
    train_matrices, validation_matrices = _task_matrices(gateway, train, validation, task)
    kwargs = {}
    if task == "recovery":
        kwargs.update(recovery_scaler=gateway.recovery_scaler, recovery_scaler_path=gateway.recovery_scaler_path)
        model_task = "recovery"
    elif task == "icu_time":
        model_task = "icu_stay_time"
    else:
        kwargs.update(support_class_weight=gateway.support_class_weight, support_class_weight_path=gateway.class_weight_path)
        model_task = "organ_support"
    bundle = train_xgb_candidate(
        task=model_task, candidate_id=candidate_id, config=_xgb_config(candidate),
        train_matrices=train_matrices, validation_matrices=validation_matrices,
        output_directory=attempt_dir / "bundle", lineage=lineage,
        feature_names=[entry["flat_name"] for entry in gateway.flat_map["entries"]],
        seed=candidate["model_seed"], run_type=RUN_PHASE12_SCIENTIFIC,
        phase12_authorization="PHASE12_FROZEN_CANDIDATE_MANIFEST", **kwargs,
    )
    bundle_path = attempt_dir / "bundle/bundle.metadata.json"
    bundle_meta = _enrich_bundle_metadata(bundle_path, master=master, candidate=candidate, execution_commit=execution_commit)
    metrics, prediction_paths, evaluation_detail = _persist_predictions(task, candidate_id, attempt_dir, validation, bundle)
    metrics_payload = {
        "metrics_version": "phase12_xgb_validation_metrics_v1", "search_version": SEARCH_VERSION,
        "candidate_id": candidate_id, "run_id": run_id, "task": task, "status": "COMPLETE",
        "partition": "validation", "metrics": metrics, "evaluation_detail": evaluation_detail,
        "config_hash": candidate["config_hash"], "candidate_list_hash": manifest["candidate_list_hash"],
        "split_sha256": master["split_sha256"], "feature_schema_sha256": master["feature_schema_sha256"],
        "flat_feature_map_sha256": master["flat_feature_map_sha256"], "preprocessor_sha256": master["preprocessor_sha256"],
        "target_contract_sha256": master["target_contract_sha256"],
        "recovery_target_scaler_sha256": master["recovery_target_scaler_sha256"] if task == "recovery" else None,
        "support_class_weight_sha256": master["support_class_weight_sha256"] if task == "organ_support" else None,
        "environment_sha256": master["environment_sha256"], "execution_commit": execution_commit,
        "probability_type": "raw_uncalibrated" if task == "organ_support" else None,
        "calibrated": False, "threshold": None, "test_accessed": False,
        "runtime_seconds": time.monotonic() - start, "started_at_utc": started, "completed_at_utc": _utc_now(),
    }
    metrics_path = attempt_dir / "validation_metrics.json"
    _write_json(metrics_path, metrics_payload)
    created = metrics_payload["completed_at_utc"]
    registry_manifest = {
        **manifest, "task": REGISTRY_TASK[task], "code_commit": execution_commit,
        "search_space_ref": master["search_space_path"],
        "split_hash": master["split_sha256"], "feature_version": lineage.feature_schema_version,
        "label_version": lineage.target_schema_version,
        "validation_objective": master["validation_objectives"][task],
    }
    record_search_attempt(
        REGISTRY_PATH, manifest=registry_manifest, candidates=manifest["candidates"],
        candidate_id=candidate_id, run_id=run_id, attempt_number=attempt,
        attempt_status_detail="COMPLETE", timestamp_utc=created, seed=candidate["model_seed"],
        retry_of_run_id=retry_of, artifact_ref=str(bundle_path.relative_to(ROOT)),
        artifact_hash=sha256_file(bundle_path), metrics_ref=str(metrics_path.relative_to(ROOT)),
        metrics_hash=sha256_file(metrics_path), config_ref=str(config_path.relative_to(ROOT)),
        run_type="scientific", preprocessor_ref=phase11["preprocessor_path"],
        preprocessor_sha256=master["preprocessor_sha256"],
        derived_feature_hash=master["flat_feature_map_sha256"],
        environment_ref=master["environment_path"], environment_sha256=master["environment_sha256"],
        notes=json.dumps({
            "search_role": "WITHIN_FAMILY_VALIDATION_CANDIDATE", "test_accessed": False,
            "candidate_config": candidate["config"], "candidate_list_hash": manifest["candidate_list_hash"],
            "best_iterations": {view: item["best_iteration"] for view, item in bundle_meta["models"].items()},
            "model_hashes": {view: item["sha256"] for view, item in bundle_meta["models"].items()},
            "target_scaler_sha256": master["recovery_target_scaler_sha256"] if task == "recovery" else None,
            "class_weight_sha256": master["support_class_weight_sha256"] if task == "organ_support" else None,
        }, sort_keys=True, separators=(",", ":")),
    )
    _register_artifacts(_artifact_records(
        task, run_id, candidate, attempt_dir, bundle_meta, prediction_paths, metrics_path,
        master, lineage, execution_commit, created,
    ))
    return metrics_payload


def _record_failure(task, manifest, candidate, attempt, master, phase11, lineage, execution_commit, error, retry_of):
    run_id = f"phase12-{candidate['candidate_id']}-attempt-{attempt}"
    attempt_dir = SEARCH_ROOT / task / candidate["candidate_id"] / f"attempt-{attempt}"
    failure_path = attempt_dir / "failure.json"
    _write_json(failure_path, {
        "candidate_id": candidate["candidate_id"], "run_id": run_id, "attempt": attempt,
        "status": "FAILED_SOFTWARE" if attempt < MAX_ATTEMPTS else "FAILED_SCIENTIFIC",
        "reason_type": type(error).__name__, "reason": str(error), "test_accessed": False,
    })
    registry_manifest = {
        **manifest, "task": REGISTRY_TASK[task], "code_commit": execution_commit,
        "search_space_ref": master["search_space_path"], "split_hash": master["split_sha256"],
        "feature_version": lineage.feature_schema_version, "label_version": lineage.target_schema_version,
        "validation_objective": master["validation_objectives"][task],
    }
    record_search_attempt(
        REGISTRY_PATH, manifest=registry_manifest, candidates=manifest["candidates"],
        candidate_id=candidate["candidate_id"], run_id=run_id, attempt_number=attempt,
        attempt_status_detail="FAILED_SOFTWARE" if attempt < MAX_ATTEMPTS else "FAILED_SCIENTIFIC",
        timestamp_utc=_utc_now(), seed=candidate["model_seed"], retry_of_run_id=retry_of,
        config_ref=str((attempt_dir / "candidate_config.json").relative_to(ROOT)), run_type="scientific",
        preprocessor_ref=phase11["preprocessor_path"], preprocessor_sha256=master["preprocessor_sha256"],
        derived_feature_hash=master["flat_feature_map_sha256"],
        environment_ref=master["environment_path"], environment_sha256=master["environment_sha256"],
        notes=json.dumps({"failure_ref": str(failure_path.relative_to(ROOT)), "test_accessed": False}),
    )
    return run_id


def _summary(task, manifest):
    rows = []
    for candidate in manifest["candidates"]:
        candidate_dir = SEARCH_ROOT / task / candidate["candidate_id"]
        attempts = sorted(candidate_dir.glob("attempt-*/validation_metrics.json"))
        if attempts:
            payload = _load(attempts[-1])
            bundle = _load(attempts[-1].parent / "bundle/bundle.metadata.json")
            row = {"candidate_id": candidate["candidate_id"], "config_hash": candidate["config_hash"], "status": "COMPLETE", **payload["metrics"]}
            row["best_iterations"] = {name: item["best_iteration"] for name, item in bundle["models"].items()}
            row["model_hashes"] = {name: item["sha256"] for name, item in bundle["models"].items()}
            row["run_id"] = payload["run_id"]
            row["metrics_path"] = str(attempts[-1].relative_to(ROOT))
            row["bundle_path"] = str((attempts[-1].parent / "bundle/bundle.metadata.json").relative_to(ROOT))
        else:
            row = {"candidate_id": candidate["candidate_id"], "config_hash": candidate["config_hash"], "status": "FAILED"}
        rows.append(row)
    path = SEARCH_ROOT / f"{task}_validation_summary.json"
    _write_json(path, {"task": task, "search_version": SEARCH_VERSION, "candidates": rows, "test_accessed": False})
    csv_path = SEARCH_ROOT / f"{task}_validation_summary.csv"
    flattened = []
    for row in rows:
        item = {key: json.dumps(value, sort_keys=True) if isinstance(value, dict) else value for key, value in row.items()}
        flattened.append(item)
    columns = sorted({key for row in flattened for key in row})
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n"); writer.writeheader(); writer.writerows(flattened)
    return rows, path, csv_path


def _rank(task, rows):
    results = [
        RankedCandidate(
            row["candidate_id"], row["status"], row,
            "raw_uncalibrated" if task == "organ_support" and row["status"] == "COMPLETE" else "",
        )
        for row in rows
    ]
    return choose_best(results, task)


def main() -> None:
    master = _load(MASTER_PATH)
    if master["status"] != "FROZEN_READY_TO_RUN":
        raise RuntimeError("Phase-12 search is not in the pre-run frozen state")
    if _git("status", "--porcelain"):
        raise RuntimeError("scientific search must start from a clean worktree")
    execution_commit = _git("rev-parse", "HEAD")
    space = _load(ROOT / master["search_space_path"])
    phase11 = _load(ROOT / master["phase11_manifest_path"])
    _verify_frozen(master, space, phase11, execution_commit)
    manifests = {task: _manifest_for_task(master, task) for task in TASKS}
    for task in TASKS:
        validate_candidate_manifest(manifests[task], space)
    before = dict(master)
    master.update(status="RUNNING", execution_commit=execution_commit, started_at_utc=_utc_now())
    assert_lifecycle_transition(before, master)
    _write_json(MASTER_PATH, master)
    gateway = Phase10XGBData(ROOT, ROOT / phase11["flat_feature_map_path"])
    train, validation = gateway.load_partition("train"), gateway.load_partition("validation")
    lineage = _lineage(master, phase11, execution_commit)
    task_summaries = {}
    try:
        for task in TASKS:
            manifest = manifests[task]
            for candidate in manifest["candidates"]:
                _verify_frozen(master, space, phase11, execution_commit)
                retry_of = ""
                complete = False
                for attempt in range(1, MAX_ATTEMPTS + 1):
                    try:
                        _run_candidate(
                            task, manifest, candidate, attempt, gateway, train, validation,
                            master, phase11, lineage, execution_commit, retry_of,
                        )
                        complete = True
                        break
                    except Exception as error:
                        retry_of = _record_failure(
                            task, manifest, candidate, attempt, master, phase11, lineage,
                            execution_commit, error, retry_of,
                        )
                print(json.dumps({"task": task, "candidate_id": candidate["candidate_id"], "complete": complete}, sort_keys=True), flush=True)
            rows, json_path, csv_path = _summary(task, manifest)
            task_summaries[task] = {"rows": rows, "json": json_path, "csv": csv_path, "best": _rank(task, rows)}
        best = {
            "manifest_version": "best_xgb_candidates_v1",
            "status": "WITHIN_FAMILY_VALIDATION_SELECTION",
            "search_version": SEARCH_VERSION,
            "serving_selection": False, "final_model_family_selection": False,
            "test_accessed": False, "support_calibrated": False, "support_threshold": None,
            "tasks": {
                task: {
                    "candidate_id": value["best"].candidate_id,
                    "run_id": next(row["run_id"] for row in value["rows"] if row["candidate_id"] == value["best"].candidate_id),
                    "metrics": value["best"].metrics,
                    "candidate_manifest_sha256": master["candidate_manifests"][task]["sha256"],
                    "split_sha256": master["split_sha256"], "feature_schema_sha256": master["feature_schema_sha256"],
                    "flat_feature_map_sha256": master["flat_feature_map_sha256"], "preprocessor_sha256": master["preprocessor_sha256"],
                }
                for task, value in task_summaries.items()
            },
        }
        best_path = SEARCH_ROOT / "best_xgb_candidates_v1.json"; _write_json(best_path, best)
        before = dict(master)
        master.update(
            status="COMPLETE", completed_at_utc=_utc_now(),
            terminal_candidate_counts={task: {"complete": sum(row["status"] == "COMPLETE" for row in value["rows"]), "failed": sum(row["status"] == "FAILED" for row in value["rows"])} for task, value in task_summaries.items()},
            best_xgb_manifest_path=str(best_path.relative_to(ROOT)), best_xgb_manifest_sha256=sha256_file(best_path),
        )
        assert_lifecycle_transition(before, master); _write_json(MASTER_PATH, master)
    except Exception:
        before = dict(master); master["status"] = "FAILED_WITH_RECORDED_CANDIDATES"; master["completed_at_utc"] = _utc_now()
        assert_lifecycle_transition(before, master); _write_json(MASTER_PATH, master)
        raise
    print(json.dumps({"status": "COMPLETE", "terminal_candidate_counts": master["terminal_candidate_counts"], "best": {task: value["best"].candidate_id for task, value in task_summaries.items()}, "test_accessed": False}, sort_keys=True))


if __name__ == "__main__":
    main()
