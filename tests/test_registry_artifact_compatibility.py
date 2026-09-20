from dataclasses import replace

import pytest

from experiments.lineage import ArtifactRecord, LineageValidationError, validate_artifact_lineage
from registry_helpers import chain, run
from vedant_infra.hashing import sha256_file


def test_calibrator_model_and_threshold_calibrator_binding(tmp_path):
    selected = run(task="organ_support", label_version="support-v1")
    base = list(chain(tmp_path, selected))[:2]
    base[0] = replace(base[0], task="organ_support", label_version="support-v1")
    base[1] = replace(base[1], task="organ_support", label_version="support-v1")
    calibrator_file = tmp_path / "artifacts/calibrator.json"
    threshold_file = tmp_path / "artifacts/threshold.json"
    calibrator_file.write_text("{}", encoding="utf-8")
    threshold_file.write_text("{}", encoding="utf-8")
    calibrator = ArtifactRecord(
        "calibrator", "artifacts/calibrator.json", "calibrator", "v1",
        sha256_file(calibrator_file), selected["run_id"], "model;prediction",
        task="organ_support", model_family="gru", split_hash=selected["split_hash"],
        feature_version="features-v1", label_version="support-v1",
        model_sha256=base[0].artifact_sha256, run_type="scientific",
        calibration_method="isotonic", partition="validation",
    )
    threshold = ArtifactRecord(
        "threshold", "artifacts/threshold.json", "threshold", "v1",
        sha256_file(threshold_file), selected["run_id"], "calibrator;prediction",
        task="organ_support", model_family="gru", split_hash=selected["split_hash"],
        feature_version="features-v1", label_version="support-v1",
        calibrator_sha256=calibrator.artifact_sha256, run_type="scientific",
        threshold_criterion="validation_f1", threshold_value="0.4",
        partition="validation",
    )
    validate_artifact_lineage(tuple(base) + (calibrator, threshold), (selected,), repository_root=tmp_path)
    with pytest.raises(LineageValidationError, match="calibrator/model"):
        validate_artifact_lineage(tuple(base) + (replace(calibrator, model_sha256="f" * 64),), (selected,), repository_root=tmp_path)
    with pytest.raises(LineageValidationError, match="threshold/calibrator"):
        validate_artifact_lineage(tuple(base) + (calibrator, replace(threshold, calibrator_sha256="f" * 64)), (selected,), repository_root=tmp_path)


def test_synthetic_run_is_excluded_from_final_query():
    from experiments.lineage import final_scientific_runs
    assert final_scientific_runs((run(run_type="synthetic"),)) == ()


def test_selected_manifest_requires_registered_model_parents(tmp_path):
    selected = run()
    model = chain(tmp_path, selected)[0]
    manifest_file = tmp_path / "artifacts/selected.json"
    manifest_file.write_text("{}", encoding="utf-8")
    manifest = ArtifactRecord(
        "selected", "artifacts/selected.json", "selected_model_manifest", "v1",
        sha256_file(manifest_file), selected["run_id"], "model",
        split_hash=selected["split_hash"], feature_version="features-v1",
        label_version="labels-v1", run_type="scientific",
    )
    validate_artifact_lineage((model, manifest), (selected,), repository_root=tmp_path)
    with pytest.raises(LineageValidationError, match="requires model parents"):
        validate_artifact_lineage((replace(manifest, parent_artifact_ids=""),), (selected,), repository_root=tmp_path)
