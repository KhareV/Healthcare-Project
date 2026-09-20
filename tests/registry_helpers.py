import csv
from pathlib import Path

from experiments.lineage import ArtifactRecord
from vedant_infra.hashing import sha256_file
from vedant_infra.registry import REGISTRY_COLUMNS, make_registry_record


def run(run_id="run-A", **updates):
    values = {
        "run_id": run_id,
        "timestamp_utc": "2026-09-18T00:00:00Z",
        "task": "recovery",
        "model_family": "gru",
        "seed": "123",
        "code_commit": "1" * 40,
        "config_ref": "configs/run.json",
        "config_hash": "a" * 64,
        "split_hash": "b" * 64,
        "feature_version": "features-v1",
        "label_version": "labels-v1",
        "model_artifact_ref": "artifacts/model.bin",
        "model_sha256": "c" * 64,
        "metrics_ref": "artifacts/metrics.csv",
        "status": "completed",
        "run_type": "scientific",
        "finalized": "true",
    }
    values.update(updates)
    return make_registry_record(**values)


def write_runs(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REGISTRY_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def chain(root: Path, selected_run=None):
    selected_run = selected_run or run()
    artifact_dir = root / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    model = artifact_dir / "model.bin"
    prediction = artifact_dir / "predictions.json"
    metric = artifact_dir / "metrics.csv"
    model.write_bytes(b"model")
    prediction.write_bytes(b"predictions")
    metric.write_bytes(b"metrics")
    model_record = ArtifactRecord(
        "model", "artifacts/model.bin", "model_checkpoint", "v1",
        sha256_file(model), selected_run["run_id"], task="recovery", model_family="gru",
        split_hash=selected_run["split_hash"], feature_version="features-v1",
        label_version="labels-v1", config_hash=selected_run["config_hash"],
        creation_commit=selected_run["code_commit"], run_type=selected_run["run_type"],
    )
    prediction_record = ArtifactRecord(
        "prediction", "artifacts/predictions.json", "prediction", "v1",
        sha256_file(prediction), selected_run["run_id"], "model", task="recovery",
        model_family="gru", split_hash=selected_run["split_hash"],
        feature_version="features-v1", label_version="labels-v1",
        model_sha256=model_record.artifact_sha256,
        partition="validation", prediction_population_hash="d" * 64,
        creation_commit=selected_run["code_commit"], run_type=selected_run["run_type"],
    )
    metric_record = ArtifactRecord(
        "metric", "artifacts/metrics.csv", "metric_table", "v1",
        sha256_file(metric), selected_run["run_id"], "prediction", task="recovery",
        model_family="gru", split_hash=selected_run["split_hash"],
        feature_version="features-v1", label_version="labels-v1",
        creation_commit=selected_run["code_commit"],
        generating_script="src/evaluation/metrics.py", creation_date_utc="2026-09-18T00:01:00Z",
        partition="validation", evaluator_version="evaluation-v1",
        metric_implementation_version="metrics-v1",
        run_type=selected_run["run_type"],
    )
    return (model_record, prediction_record, metric_record)
