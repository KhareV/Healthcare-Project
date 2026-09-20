import json
from dataclasses import replace
from pathlib import Path

from data.split import SubjectSplit, serialize_split_csv
from evaluation.selection_validation import EXPLANATION_METHOD
from experiments.lineage import ArtifactRecord, write_artifact_index
from experiments.search_governance import canonical_sha256
from registry_helpers import write_runs
from vedant_infra.hashing import sha256_file
from vedant_infra.registry import make_registry_record


TASKS = ("recovery", "icu_stay_time", "organ_support")
FAMILIES = ("xgboost", "gru")


def _write(path: Path, content: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return sha256_file(path)


def build_complete_g3_fixture(root: Path):
    split = root / "artifacts/splits/split_v1.csv"
    split.parent.mkdir(parents=True, exist_ok=True)
    split.write_bytes(
        serialize_split_csv(
            (
                SubjectSplit("train-subject", "train", "2008-2010"),
                SubjectSplit("validation-subject", "validation", "2014-2016"),
                SubjectSplit("test-subject", "test", "2017-2019"),
            )
        )
    )
    split_hash = sha256_file(split)
    common_metrics = root / "artifacts/synthetic/search/metrics.json"
    metrics_hash = _write(common_metrics, b'{"partition":"validation"}\n')
    rows = []
    selected_rows = {}
    gru_rows = {}
    for task in TASKS:
        for family in FAMILIES:
            candidate_list_hash = canonical_sha256(
                ["{}-{}-{:02d}".format(task, family, index) for index in range(30)]
            )
            search_space_hash = canonical_sha256(
                {"task": task, "family": family, "space": "synthetic"}
            )
            for index in range(30):
                run_id = "search-{}-{}-{:02d}".format(task, family, index)
                candidate_id = "candidate-{}-{}-{:02d}".format(
                    task, family, index
                )
                config = root / (
                    "configs/synthetic/{}-{}-{:02d}.json".format(
                        task, family, index
                    )
                )
                config_hash = _write(
                    config,
                    json.dumps(
                        {"task": task, "family": family, "index": index},
                        sort_keys=True,
                    ).encode("utf-8"),
                )
                model = root / (
                    "artifacts/synthetic/search/{}-{}-{:02d}.model".format(
                        task, family, index
                    )
                )
                model_hash = _write(
                    model, run_id.encode("utf-8")
                )
                row = make_registry_record(
                    run_id=run_id,
                    timestamp_utc="2026-09-18T00:00:00Z",
                    task=task,
                    model_family=family,
                    seed=str(index),
                    code_commit="1" * 40,
                    config_ref=str(config.relative_to(root)),
                    config_hash=config_hash,
                    search_space_hash=search_space_hash,
                    split_hash=split_hash,
                    feature_version="features-v1",
                    label_version=task + "-labels-v1",
                    model_artifact_ref=str(model.relative_to(root)),
                    model_sha256=model_hash,
                    metrics_ref=str(common_metrics.relative_to(root)),
                    metrics_sha256=metrics_hash,
                    status="completed",
                    candidate_id=candidate_id,
                    search_version="search-{}-{}-v1".format(task, family),
                    candidate_list_hash=candidate_list_hash,
                    attempt_number="1",
                    attempt_status_detail="COMPLETE",
                    validation_objective="synthetic-validation-only",
                    run_type="synthetic",
                    finalized="true",
                )
                rows.append(row)
                if index == 0:
                    if family == "gru":
                        gru_rows[task] = row
                    chosen = {
                        "recovery": "xgboost",
                        "icu_stay_time": "gru",
                        "organ_support": "xgboost",
                    }[task]
                    if family == chosen:
                        selected_rows[task] = row
    for task in TASKS:
        model = root / ("artifacts/synthetic/lstm-{}.model".format(task))
        model_hash = _write(model, ("lstm-" + task).encode("utf-8"))
        config = root / ("configs/synthetic/lstm-{}.json".format(task))
        config_hash = _write(config, b'{"cell":"lstm"}')
        rows.append(
            make_registry_record(
                run_id="lstm-" + task,
                timestamp_utc="2026-09-18T00:10:00Z",
                task=task,
                model_family="lstm",
                seed="7",
                code_commit="1" * 40,
                config_ref=str(config.relative_to(root)),
                config_hash=config_hash,
                split_hash=split_hash,
                feature_version="features-v1",
                label_version=task + "-labels-v1",
                model_artifact_ref=str(model.relative_to(root)),
                model_sha256=model_hash,
                metrics_ref=str(common_metrics.relative_to(root)),
                metrics_sha256=metrics_hash,
                status="completed",
                parent_run_id=gru_rows[task]["run_id"],
                run_type="sensitivity_smoke",
                finalized="true",
            )
        )
    write_runs(root / "experiments/registry.csv", rows)

    artifact_records = []
    task_entries = {}
    model_records = {}
    for task in TASKS:
        selected = selected_rows[task]
        model_id = task + "-selected-model"
        model_record = ArtifactRecord(
            artifact_id=model_id,
            artifact_path=selected["model_artifact_ref"],
            artifact_type="model_checkpoint",
            artifact_version="synthetic-g3-v1",
            artifact_sha256=selected["model_sha256"],
            producing_run_id=selected["run_id"],
            task=task,
            model_family=selected["model_family"],
            split_hash=split_hash,
            feature_version="features-v1",
            label_version=task + "-labels-v1",
            config_hash=selected["config_hash"],
            creation_commit="1" * 40,
            run_type="synthetic",
        )
        model_records[task] = model_record
        artifact_records.append(model_record)
        preprocessor = root / (
            "artifacts/preprocessors/{}-preprocessor.json".format(task)
        )
        preprocessor_hash = _write(
            preprocessor, b'{"fit_partition":"train"}\n'
        )
        artifact_records.append(
            ArtifactRecord(
                artifact_id=task + "-preprocessor",
                artifact_path=str(preprocessor.relative_to(root)),
                artifact_type="preprocessor",
                artifact_version="synthetic-g3-v1",
                artifact_sha256=preprocessor_hash,
                producing_run_id=selected["run_id"],
                task=task,
                model_family=selected["model_family"],
                split_hash=split_hash,
                feature_version="features-v1",
                label_version=task + "-labels-v1",
                run_type="synthetic",
            )
        )
        task_entries[task] = {
            "family": selected["model_family"],
            "run_id": selected["run_id"],
            "candidate_id": selected["candidate_id"],
            "artifact_ref": selected["model_artifact_ref"],
            "artifact_sha256": selected["model_sha256"],
            "feature_version": "features-v1",
            "label_version": task + "-labels-v1",
            "split_hash": split_hash,
            "preprocessor_ref": str(preprocessor.relative_to(root)),
            "preprocessor_sha256": preprocessor_hash,
            "preprocessing_fit_partition": "train",
            "explanation_method": EXPLANATION_METHOD[
                selected["model_family"]
            ],
            "selected_gru_run_id": gru_rows[task]["run_id"],
        }
    support_model = model_records["organ_support"]
    prediction = root / "artifacts/synthetic/support-validation-predictions.json"
    prediction_hash = _write(
        prediction, b'{"partition":"validation","probability_type":"raw"}\n'
    )
    prediction_record = ArtifactRecord(
        artifact_id="support-validation-prediction",
        artifact_path=str(prediction.relative_to(root)),
        artifact_type="prediction",
        artifact_version="synthetic-g3-v1",
        artifact_sha256=prediction_hash,
        producing_run_id=selected_rows["organ_support"]["run_id"],
        parent_artifact_ids=support_model.artifact_id,
        task="organ_support",
        model_family=support_model.model_family,
        split_hash=split_hash,
        feature_version="features-v1",
        label_version="organ_support-labels-v1",
        model_sha256=support_model.artifact_sha256,
        partition="validation",
        probability_type="raw",
        prediction_population_hash="d" * 64,
        run_type="synthetic",
    )
    artifact_records.append(prediction_record)
    calibrator = root / "artifacts/synthetic/support-calibrator.json"
    calibrator_hash = _write(
        calibrator, b'{"method":"isotonic","partition":"validation"}\n'
    )
    calibrator_record = ArtifactRecord(
        artifact_id="support-calibrator",
        artifact_path=str(calibrator.relative_to(root)),
        artifact_type="calibrator",
        artifact_version="synthetic-g3-v1",
        artifact_sha256=calibrator_hash,
        producing_run_id=selected_rows["organ_support"]["run_id"],
        parent_artifact_ids=(
            support_model.artifact_id
            + ";"
            + prediction_record.artifact_id
        ),
        task="organ_support",
        model_family=support_model.model_family,
        split_hash=split_hash,
        feature_version="features-v1",
        label_version="organ_support-labels-v1",
        model_sha256=support_model.artifact_sha256,
        calibration_method="isotonic",
        partition="validation",
        run_type="synthetic",
    )
    artifact_records.append(calibrator_record)
    threshold = root / "artifacts/synthetic/support-threshold.json"
    threshold_hash = _write(
        threshold,
        b'{"criterion":"validation_f1","threshold":0.4}\n',
    )
    threshold_record = ArtifactRecord(
        artifact_id="support-threshold",
        artifact_path=str(threshold.relative_to(root)),
        artifact_type="threshold",
        artifact_version="synthetic-g3-v1",
        artifact_sha256=threshold_hash,
        producing_run_id=selected_rows["organ_support"]["run_id"],
        parent_artifact_ids=(
            calibrator_record.artifact_id
            + ";"
            + prediction_record.artifact_id
        ),
        task="organ_support",
        model_family=support_model.model_family,
        split_hash=split_hash,
        feature_version="features-v1",
        label_version="organ_support-labels-v1",
        calibrator_sha256=calibrator_hash,
        threshold_criterion="validation_f1",
        threshold_value="0.4",
        partition="validation",
        run_type="synthetic",
    )
    artifact_records.append(threshold_record)
    task_entries["organ_support"]["calibrator"] = {
        "artifact_ref": str(calibrator.relative_to(root)),
        "artifact_sha256": calibrator_hash,
        "selected_model_sha256": support_model.artifact_sha256,
        "partition": "validation",
        "method": "isotonic",
    }
    task_entries["organ_support"]["threshold"] = {
        "artifact_ref": str(threshold.relative_to(root)),
        "artifact_sha256": threshold_hash,
        "calibrator_sha256": calibrator_hash,
        "criterion": "validation_f1",
        "partition": "validation",
        "value": 0.4,
    }
    manifest_content = {
        "manifest_version": "selected_models_calibrated_v1",
        "selection_mode": "synthetic",
        "status": "SYNTHETIC_G3_TEST_FIXTURE_ONLY",
        "test_accessed": False,
        "tasks": task_entries,
    }
    manifest = {
        **manifest_content,
        "manifest_sha256": canonical_sha256(manifest_content),
    }
    manifest_path = root / "artifacts/models/selected_models_v1.json"
    _write(
        manifest_path,
        (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode(
            "utf-8"
        ),
    )
    manifest_record = ArtifactRecord(
        artifact_id="selected-manifest",
        artifact_path=str(manifest_path.relative_to(root)),
        artifact_type="selected_model_manifest",
        artifact_version="synthetic-g3-v1",
        artifact_sha256=sha256_file(manifest_path),
        producing_run_id=selected_rows["recovery"]["run_id"],
        parent_artifact_ids=";".join(
            model_records[task].artifact_id for task in TASKS
        ),
        split_hash=split_hash,
        feature_version="features-v1",
        creation_commit="1" * 40,
        run_type="synthetic",
    )
    artifact_records.append(manifest_record)
    write_artifact_index(
        root / "experiments/artifacts.csv", artifact_records
    )
    governance = root / "artifacts/governance"
    governance.mkdir(parents=True, exist_ok=True)
    (governance / "explanation_adapters_review.json").write_text(
        json.dumps(
            {
                "scope": "synthetic_test",
                "tree_shap_implemented": True,
                "integrated_gradients_implemented": True,
                "reviewed": True,
            }
        ),
        encoding="utf-8",
    )
    (governance / "g3_review_signoffs.json").write_text(
        json.dumps(
            {
                "scope": "synthetic_test",
                "approved_reviewers": [
                    "sanskruti",
                    "vedant",
                    "pulkit",
                ],
            }
        ),
        encoding="utf-8",
    )
    return {
        "split": split,
        "manifest": manifest_path,
        "models": {
            task: root / model_records[task].artifact_path
            for task in TASKS
        },
        "calibrator": calibrator,
        "threshold": threshold,
    }


def rewrite_manifest_hash(path: Path, mutate):
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutate(payload)
    content = dict(payload)
    content.pop("manifest_sha256", None)
    payload["manifest_sha256"] = canonical_sha256(content)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
