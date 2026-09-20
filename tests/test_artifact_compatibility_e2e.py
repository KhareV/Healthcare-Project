"""Phase-14 hostile selected-bundle compatibility tests."""

import pytest

from serving.artifacts import (
    ArtifactCompatibilityError,
    ArtifactHashMismatchError,
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


@pytest.mark.parametrize(
    ("relative", "label"),
    (
        ("recovery_gru.mock", "recovery model"),
        ("recovery_preprocessor.json", "recovery preprocessor"),
        ("support_calibrator.json", "support calibrator"),
        ("support_threshold.json", "support threshold"),
    ),
)
def test_artifact_byte_drift_fails_before_any_loader(tmp_path, relative, label):
    root = copy_serving_fixture(tmp_path)
    path = root / FIXTURE_RELATIVE / relative
    path.write_bytes(path.read_bytes() + b"HOSTILE_DRIFT")
    runtime = FixtureRuntime()
    with pytest.raises(ArtifactHashMismatchError, match=label):
        load_serving_bundle(**build_kwargs(root, runtime))
    assert runtime.loader_calls == []
    assert runtime.preprocessors == {}
    assert runtime.calibrator is None


@pytest.mark.parametrize("field", ("feature_version", "label_version"))
def test_model_metadata_contract_drift_fails_before_inference(tmp_path, field):
    root = copy_serving_fixture(tmp_path)
    path = root / FIXTURE_RELATIVE / "recovery_model.metadata.json"
    metadata = read_json(path)
    metadata[field] = "SYNTHETIC_HOSTILE_MISMATCH"
    digest = write_json(path, metadata)
    rehash_manifest(
        root,
        lambda value: value["tasks"]["recovery"].update(
            artifact_metadata_sha256=digest
        ),
    )
    runtime = FixtureRuntime()
    with pytest.raises(ArtifactCompatibilityError, match=field):
        load_serving_bundle(**build_kwargs(root, runtime))
    assert runtime.loader_calls == []


def test_preprocessor_train_fit_provenance_drift_fails_before_loader(tmp_path):
    root = copy_serving_fixture(tmp_path)
    directory = root / FIXTURE_RELATIVE
    preprocessor_path = directory / "recovery_preprocessor.json"
    preprocessor = read_json(preprocessor_path)
    preprocessor["fit_partition"] = "validation"
    preprocessor_hash = write_json(preprocessor_path, preprocessor)
    metadata_path = directory / "recovery_model.metadata.json"
    metadata = read_json(metadata_path)
    metadata["preprocessor_sha256"] = preprocessor_hash
    metadata_hash = write_json(metadata_path, metadata)
    rehash_manifest(
        root,
        lambda value: value["tasks"]["recovery"].update(
            preprocessor_sha256=preprocessor_hash,
            artifact_metadata_sha256=metadata_hash,
        ),
    )
    runtime = FixtureRuntime()
    with pytest.raises(ArtifactCompatibilityError, match="fit_partition"):
        load_serving_bundle(**build_kwargs(root, runtime))
    assert runtime.loader_calls == []


def test_split_and_explanation_route_drift_fail_before_loader(tmp_path):
    for name, mutation, expected in (
        (
            "split",
            lambda value: value["tasks"]["recovery"].update(split_hash="b" * 64),
            "split provenance",
        ),
        (
            "route",
            lambda value: value["tasks"]["icu_stay_time"].update(
                explanation_method="integrated_gradients"
            ),
            "explanation route",
        ),
    ):
        root = copy_serving_fixture(tmp_path / name)
        rehash_manifest(root, mutation)
        runtime = FixtureRuntime()
        with pytest.raises(ArtifactCompatibilityError, match=expected):
            load_serving_bundle(**build_kwargs(root, runtime))
        assert runtime.loader_calls == []


def test_stale_selected_manifest_hash_is_rejected_before_loading(tmp_path):
    root = copy_serving_fixture(tmp_path)
    expected = sha256_file(manifest_path(root))
    rehash_manifest(
        root,
        lambda value: value.update(split_version="SYNTHETIC_STALE_MANIFEST"),
    )
    runtime = FixtureRuntime()
    kwargs = build_kwargs(root, runtime)
    kwargs["expected_manifest_sha256"] = expected
    with pytest.raises(ArtifactHashMismatchError, match="selected manifest"):
        load_serving_bundle(**kwargs)
    assert runtime.loader_calls == []


def test_support_calibrator_and_threshold_cross_binding_fail_closed(tmp_path):
    for name, filename, key, expected in (
        ("calibrator", "support_calibrator.json", "selected_model_sha256", "calibrator/model"),
        ("threshold", "support_threshold.json", "calibrator_sha256", "threshold/calibrator"),
    ):
        root = copy_serving_fixture(tmp_path / name)
        path = root / FIXTURE_RELATIVE / filename
        payload = read_json(path)
        payload[key] = "b" * 64
        digest = write_json(path, payload)
        manifest_key = "calibrator_sha256" if name == "calibrator" else "threshold_sha256"
        rehash_manifest(
            root,
            lambda value, manifest_key=manifest_key, digest=digest: value["tasks"][
                "organ_support"
            ].update({manifest_key: digest}),
        )
        runtime = FixtureRuntime()
        with pytest.raises(ArtifactCompatibilityError, match=expected):
            load_serving_bundle(**build_kwargs(root, runtime))
        assert runtime.loader_calls == []
