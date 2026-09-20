import pytest

from evaluation.final_test import FinalTestError, validate_final_result_traceability
from experiments.lineage import ArtifactRecord, write_artifact_index
from vedant_infra.hashing import sha256_file


def _fixture(root):
    paths = {}
    for name in ("model", "preprocessor", "prediction", "manifest", "g3"):
        path = root / "artifacts" / (name + ".json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"name":"' + name + '"}', encoding="utf-8")
        paths[name] = path
    split_hash = "5" * 64
    model = ArtifactRecord(
        "model", "artifacts/model.json", "model_checkpoint", "v1",
        sha256_file(paths["model"]), "run-1", task="recovery",
        model_family="gru", split_hash=split_hash, run_type="synthetic",
    )
    preprocessor = ArtifactRecord(
        "preprocessor", "artifacts/preprocessor.json", "preprocessor", "v1",
        sha256_file(paths["preprocessor"]), "run-1", task="recovery",
        model_family="gru", split_hash=split_hash, run_type="synthetic",
    )
    prediction = ArtifactRecord(
        "prediction", "artifacts/prediction.json", "prediction", "v1",
        sha256_file(paths["prediction"]), "run-1", parent_artifact_ids="model",
        task="recovery", model_family="gru", split_hash=split_hash,
        model_sha256=model.artifact_sha256, partition="test", run_type="synthetic",
    )
    manifest = ArtifactRecord(
        "manifest", "artifacts/manifest.json", "selected_model_manifest", "v1",
        sha256_file(paths["manifest"]), "run-1", parent_artifact_ids="model",
        split_hash=split_hash, run_type="synthetic",
    )
    write_artifact_index(
        root / "experiments/artifacts.csv",
        (model, preprocessor, prediction, manifest),
    )
    return paths, split_hash


def _row(root):
    paths, split_hash = _fixture(root)
    return {
        "task": "recovery",
        "model_run_id": "run-1",
        "feature_version": "features-v1",
        "label_version": "labels-v1",
        "code_commit": "1" * 40,
        "prediction_artifact_id": "prediction",
        "prediction_sha256": sha256_file(paths["prediction"]),
        "model_artifact_id": "model",
        "model_sha256": sha256_file(paths["model"]),
        "manifest_artifact_id": "manifest",
        "manifest_sha256": sha256_file(paths["manifest"]),
        "preprocessor_artifact_id": "preprocessor",
        "preprocessor_sha256": sha256_file(paths["preprocessor"]),
        "g3_marker_ref": "artifacts/g3.json",
        "g3_marker_sha256": sha256_file(paths["g3"]),
        "split_hash": split_hash,
    }


def test_final_result_requires_complete_frozen_lineage(tmp_path):
    row = _row(tmp_path)
    validate_final_result_traceability(tmp_path, (row,))
    broken = dict(row)
    broken["g3_marker_sha256"] = ""
    with pytest.raises(FinalTestError, match="hash lineage"):
        validate_final_result_traceability(tmp_path, (broken,))
