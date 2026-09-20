import json
from pathlib import Path

import pytest

from serving.artifacts import (
    ArtifactCompatibilityError,
    ArtifactHashMismatchError,
    ArtifactMissingError,
    ManifestInvalidError,
    SyntheticArtifactProhibitedError,
    UnregisteredArtifactError,
    UnsupportedServingFamilyError,
    load_serving_bundle,
)
from serving_helpers import (
    FIXTURE_RELATIVE,
    FixtureRuntime,
    build_kwargs,
    copy_serving_fixture,
    manifest_path,
    read_json,
    rehash_manifest,
    write_json,
)
from vedant_infra.hashing import sha256_file


def _load(root, runtime=None, **overrides):
    runtime = runtime or FixtureRuntime()
    kwargs = build_kwargs(root, runtime)
    kwargs.update(overrides)
    return load_serving_bundle(**kwargs), runtime


def _convert_fixture_to_unregistered_real(root):
    directory = root / FIXTURE_RELATIVE
    manifest = read_json(manifest_path(root))
    manifest.update(
        manifest_version="selected_models_calibrated_v1",
        selection_mode="real",
        status="MODEL_SELECTION_CALIBRATION_FROZEN",
        serving_ready=True,
        split_version="split_spec_v1",
    )
    names = {
        "recovery": ("recovery_model.metadata.json", "recovery_preprocessor.json", "recovery_model_v1", "feature_schema_v1", "recovery_labels_v1"),
        "icu_stay_time": ("icu_model.metadata.json", "icu_preprocessor.json", "icu_model_v1", "feature_schema_v1", "icu_time_labels_v1"),
        "organ_support": ("support_model.metadata.json", "support_preprocessor.json", "support_model_v1", "feature_schema_v1", "organ_support_labels_v1"),
    }
    preprocessor_hashes = {}
    for task, (metadata_name, preprocessor_name, model_version, feature_version, label_version) in names.items():
        family = "xgboost" if task == "recovery" else manifest["tasks"][task]["family"]
        preprocessor_path = directory / preprocessor_name
        preprocessor = read_json(preprocessor_path)
        preprocessor.update(
            feature_version=feature_version, synthetic=False, serving_authorized=True
        )
        preprocessor_hash = write_json(preprocessor_path, preprocessor)
        preprocessor_hashes[task] = preprocessor_hash
        metadata_path = directory / metadata_name
        metadata = read_json(metadata_path)
        metadata.update(
            family=family,
            model_version=model_version,
            feature_version=feature_version,
            label_version=label_version,
            preprocessor_sha256=preprocessor_hash,
            synthetic=False,
            serving_authorized=True,
        )
        entry = manifest["tasks"][task]
        entry.update(
            family=family,
            explanation_method=(
                "tree_shap" if family == "xgboost" else "integrated_gradients"
            ),
            model_version=model_version,
            feature_version=feature_version,
            label_version=label_version,
            preprocessor_sha256=preprocessor_hash,
            artifact_metadata_sha256=write_json(metadata_path, metadata),
        )
    calibrator_path = directory / "support_calibrator.json"
    calibrator = read_json(calibrator_path)
    calibrator.update(
        feature_version="feature_schema_v1",
        label_version="organ_support_labels_v1",
        preprocessor_sha256=preprocessor_hashes["organ_support"],
        synthetic=False,
        serving_authorized=True,
    )
    calibrator_hash = write_json(calibrator_path, calibrator)
    manifest["tasks"]["organ_support"]["calibrator_sha256"] = calibrator_hash
    threshold_path = directory / "support_threshold.json"
    threshold = read_json(threshold_path)
    threshold.update(
        calibrator_sha256=calibrator_hash,
        synthetic=False,
        serving_authorized=True,
    )
    manifest["tasks"]["organ_support"]["threshold_sha256"] = write_json(
        threshold_path, threshold
    )
    rehash_manifest(root, lambda payload: (payload.clear(), payload.update(manifest)))


def test_valid_synthetic_mixed_family_bundle_loads_hash_first(tmp_path):
    root = copy_serving_fixture(tmp_path)
    bundle, runtime = _load(root)
    assert [item.identity.task for item in bundle.tasks] == [
        "recovery", "icu_stay_time", "organ_support"
    ]
    assert [item.identity.family for item in bundle.tasks] == ["gru", "xgboost", "gru"]
    assert runtime.loader_calls == [
        ("gru", "recovery"), ("xgboost", "icu_stay_time"), ("gru", "organ_support")
    ]
    assert bundle.support_threshold == 0.5


def test_missing_manifest_fails_before_any_loader(tmp_path):
    root = copy_serving_fixture(tmp_path)
    runtime = FixtureRuntime()
    with pytest.raises(ArtifactMissingError, match="selected manifest"):
        load_serving_bundle(
            root,
            manifest_ref="tests/fixtures/serving/missing.json",
            expected_manifest_sha256="1" * 64,
            scope="synthetic",
            model_loaders=runtime.model_loaders,
            preprocessor_loader=runtime.preprocessor_loader,
            calibrator_loader=runtime.calibrator_loader,
        )
    assert runtime.loader_calls == []


def test_real_mode_rejects_synthetic_manifest_before_loading(tmp_path):
    root = copy_serving_fixture(tmp_path)
    runtime = FixtureRuntime()
    with pytest.raises(SyntheticArtifactProhibitedError, match="synthetic manifest"):
        _load(root, runtime, scope="real")
    assert runtime.loader_calls == []


def test_unregistered_real_artifacts_fail_before_deserializing_loaders(tmp_path):
    root = copy_serving_fixture(tmp_path)
    _convert_fixture_to_unregistered_real(root)
    runtime = FixtureRuntime()
    with pytest.raises(UnregisteredArtifactError, match="not traceable"):
        _load(root, runtime, scope="real")
    assert runtime.loader_calls == []
    assert runtime.preprocessors == {}


@pytest.mark.parametrize("field", ("calibrator_ref", "threshold_ref"))
def test_real_manifest_missing_support_dependency_fails_before_registry_or_loader(tmp_path, field):
    root = copy_serving_fixture(tmp_path)
    _convert_fixture_to_unregistered_real(root)
    rehash_manifest(root, lambda payload: payload["tasks"]["organ_support"].pop(field))
    runtime = FixtureRuntime()
    with pytest.raises(ManifestInvalidError, match=field):
        _load(root, runtime, scope="real")
    assert runtime.loader_calls == []


@pytest.mark.parametrize("change", ("missing", "unknown"))
def test_missing_or_unknown_task_fails_before_loading(tmp_path, change):
    root = copy_serving_fixture(tmp_path)
    def mutate(payload):
        support = payload["tasks"].pop("organ_support")
        if change == "unknown":
            payload["tasks"]["unknown_task"] = support
    rehash_manifest(root, mutate)
    runtime = FixtureRuntime()
    with pytest.raises(ManifestInvalidError, match="exactly three"):
        _load(root, runtime)
    assert runtime.loader_calls == []


@pytest.mark.parametrize("family", ("lstm", "naive"))
def test_unsupported_serving_family_fails_before_loading(tmp_path, family):
    root = copy_serving_fixture(tmp_path)
    rehash_manifest(root, lambda payload: payload["tasks"]["recovery"].update(family=family))
    runtime = FixtureRuntime()
    with pytest.raises(UnsupportedServingFamilyError):
        _load(root, runtime)
    assert runtime.loader_calls == []


def test_outer_manifest_file_hash_mismatch_fails_before_parsing_or_loading(tmp_path):
    root = copy_serving_fixture(tmp_path)
    path = manifest_path(root)
    expected = sha256_file(path)
    path.write_text(path.read_text() + "\n")
    runtime = FixtureRuntime()
    with pytest.raises(ArtifactHashMismatchError, match="selected manifest"):
        _load(root, runtime, expected_manifest_sha256=expected)
    assert runtime.loader_calls == []


def test_internal_manifest_content_hash_mismatch_fails_before_loading(tmp_path):
    root = copy_serving_fixture(tmp_path)
    path = manifest_path(root)
    payload = read_json(path)
    payload["split_version"] = "SYNTHETIC_MUTATION_NOT_REAL"
    file_hash = write_json(path, payload)
    runtime = FixtureRuntime()
    with pytest.raises(ArtifactHashMismatchError, match="content hash"):
        _load(root, runtime, expected_manifest_sha256=file_hash)
    assert runtime.loader_calls == []


def test_model_hash_drift_fails_before_loader_and_preprocessor(tmp_path):
    root = copy_serving_fixture(tmp_path)
    (root / FIXTURE_RELATIVE / "recovery_gru.mock").write_text("mutated")
    runtime = FixtureRuntime()
    with pytest.raises(ArtifactHashMismatchError, match="recovery model"):
        _load(root, runtime)
    assert runtime.loader_calls == []
    assert runtime.preprocessors == {}


def test_missing_model_file_fails_before_loading(tmp_path):
    root = copy_serving_fixture(tmp_path)
    (root / FIXTURE_RELATIVE / "icu_xgboost.mock").unlink()
    runtime = FixtureRuntime()
    with pytest.raises(ArtifactMissingError, match="icu_stay_time model"):
        _load(root, runtime)
    assert runtime.loader_calls == []


@pytest.mark.parametrize("field", ("feature_version", "label_version"))
def test_model_version_compatibility_mismatch_fails_before_loading(tmp_path, field):
    root = copy_serving_fixture(tmp_path)
    rehash_manifest(
        root,
        lambda payload: payload["tasks"]["recovery"].update({field: "SYNTHETIC_WRONG_VERSION"}),
    )
    runtime = FixtureRuntime()
    with pytest.raises(ArtifactCompatibilityError, match=field):
        _load(root, runtime)
    assert runtime.loader_calls == []


def test_split_provenance_mismatch_fails_before_loading(tmp_path):
    root = copy_serving_fixture(tmp_path)
    rehash_manifest(
        root,
        lambda payload: payload["tasks"]["recovery"].update(split_hash="b" * 64),
    )
    runtime = FixtureRuntime()
    with pytest.raises(ArtifactCompatibilityError, match="split provenance"):
        _load(root, runtime)
    assert runtime.loader_calls == []


def test_required_target_transform_must_resolve_before_loading(tmp_path):
    root = copy_serving_fixture(tmp_path)
    rehash_manifest(
        root,
        lambda payload: payload["tasks"]["recovery"].update(
            target_transform_required=True
        ),
    )
    runtime = FixtureRuntime()
    with pytest.raises(ManifestInvalidError, match="target_transform_ref"):
        _load(root, runtime)
    assert runtime.loader_calls == []


def test_preprocessor_hash_drift_fails_before_any_loader(tmp_path):
    root = copy_serving_fixture(tmp_path)
    path = root / FIXTURE_RELATIVE / "support_preprocessor.json"
    path.write_text(path.read_text() + "\n")
    runtime = FixtureRuntime()
    with pytest.raises(ArtifactHashMismatchError, match="preprocessor"):
        _load(root, runtime)
    assert runtime.loader_calls == []


def test_wrong_task_model_metadata_fails_before_loading(tmp_path):
    root = copy_serving_fixture(tmp_path)
    def mutate(payload):
        recovery = payload["tasks"]["recovery"]
        icu = payload["tasks"]["icu_stay_time"]
        icu["artifact_ref"] = recovery["artifact_ref"]
        icu["artifact_sha256"] = recovery["artifact_sha256"]
        icu["artifact_metadata_ref"] = recovery["artifact_metadata_ref"]
        icu["artifact_metadata_sha256"] = recovery["artifact_metadata_sha256"]
    rehash_manifest(root, mutate)
    runtime = FixtureRuntime()
    with pytest.raises(ArtifactCompatibilityError, match="icu_stay_time model metadata mismatch: task"):
        _load(root, runtime)
    assert runtime.loader_calls == []


def test_support_calibrator_bound_to_wrong_model_fails_before_loading(tmp_path):
    root = copy_serving_fixture(tmp_path)
    calibrator_path = root / FIXTURE_RELATIVE / "support_calibrator.json"
    calibrator = read_json(calibrator_path)
    calibrator["selected_model_sha256"] = "b" * 64
    calibrator_hash = write_json(calibrator_path, calibrator)
    rehash_manifest(
        root,
        lambda payload: payload["tasks"]["organ_support"].update(
            calibrator_sha256=calibrator_hash
        ),
    )
    runtime = FixtureRuntime()
    with pytest.raises(ArtifactCompatibilityError, match="calibrator/model mismatch"):
        _load(root, runtime)
    assert runtime.loader_calls == []


def test_support_threshold_bound_to_wrong_calibrator_fails_before_loading(tmp_path):
    root = copy_serving_fixture(tmp_path)
    threshold_path = root / FIXTURE_RELATIVE / "support_threshold.json"
    threshold = read_json(threshold_path)
    threshold["calibrator_sha256"] = "b" * 64
    threshold_hash = write_json(threshold_path, threshold)
    rehash_manifest(
        root,
        lambda payload: payload["tasks"]["organ_support"].update(
            threshold_sha256=threshold_hash
        ),
    )
    runtime = FixtureRuntime()
    with pytest.raises(ArtifactCompatibilityError, match="threshold/calibrator mismatch"):
        _load(root, runtime)
    assert runtime.loader_calls == []


@pytest.mark.parametrize("field", ("calibrator_ref", "threshold_ref"))
def test_missing_support_calibration_dependency_is_a_startup_failure(tmp_path, field):
    root = copy_serving_fixture(tmp_path)
    rehash_manifest(
        root, lambda payload: payload["tasks"]["organ_support"].pop(field)
    )
    runtime = FixtureRuntime()
    with pytest.raises(ManifestInvalidError, match=field):
        _load(root, runtime)
    assert runtime.loader_calls == []


def test_wrong_explanation_route_fails_before_loading(tmp_path):
    root = copy_serving_fixture(tmp_path)
    rehash_manifest(
        root,
        lambda payload: payload["tasks"]["icu_stay_time"].update(
            explanation_method="integrated_gradients"
        ),
    )
    runtime = FixtureRuntime()
    with pytest.raises(ArtifactCompatibilityError, match="explanation route"):
        _load(root, runtime)
    assert runtime.loader_calls == []


def test_absolute_and_traversal_paths_are_rejected(tmp_path):
    root = copy_serving_fixture(tmp_path)
    rehash_manifest(
        root,
        lambda payload: payload["tasks"]["recovery"].update(
            artifact_ref="../../outside.mock"
        ),
    )
    runtime = FixtureRuntime()
    with pytest.raises(ManifestInvalidError, match="escapes"):
        _load(root, runtime)
    assert runtime.loader_calls == []


def test_loader_identity_cross_wiring_is_rejected(tmp_path):
    root = copy_serving_fixture(tmp_path)
    runtime = FixtureRuntime()
    def wrong_loader(task, path, metadata, identity):
        predictor = runtime.model_loader(identity.family)(task, path, metadata, identity)
        predictor.task = "icu_stay_time" if task == "recovery" else task
        return predictor
    loaders = runtime.model_loaders
    loaders["gru"] = wrong_loader
    with pytest.raises(ArtifactCompatibilityError, match="loader returned incompatible"):
        _load(root, runtime, model_loaders=loaders)
