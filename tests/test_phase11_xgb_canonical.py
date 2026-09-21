import csv
import json
from dataclasses import replace
from pathlib import Path
import shutil

import numpy as np
import pytest
import torch

from data.synthetic.provenance import sha256_file
from data.xgb_canonical import (
    Phase10XGBData,
    StructuredModelInput,
    TaskMatrix,
    XGBCanonicalContractError,
    assert_exact_reconstruction,
    build_flat_feature_map,
    flatten_row,
    model_information_views,
    row_keys_hash,
    unflatten_row,
    validate_flat_feature_map,
)
from models.icu_time_postprocess import remaining_icu_hours_from_log_prediction
from models.xgb_canonical import (
    RUN_ENGINEERING_SMOKE,
    XGBCandidateConfig,
    XGBCandidateError,
    XGBLineage,
    config_sha256,
    load_xgb_bundle,
    train_xgb_candidate,
    xgb_registry_fields,
)
from preprocess.target_scaler import RecoveryTargetScaler
from training.class_weights import SupportClassWeight


ROOT = Path(__file__).resolve().parents[1]
MAP = ROOT / "artifacts/features/synthetic_xgb_flat_feature_map_v1.json"
MANIFEST = ROOT / "artifacts/manifests/synthetic_phase11_xgb_input_manifest_v1.json"


def gateway():
    return Phase10XGBData(ROOT, MAP)


def test_phase10_freeze_and_phase11_manifest_lineage_are_complete():
    payload = json.loads(MANIFEST.read_text())
    assert payload["phase10_status"] == "PHASE10_COMPLETE"
    assert payload["parity_result"] == "PASS_COMPLETE_TRAIN_AND_VALIDATION"
    assert payload["partitions"]["train"]["row_count"] == 8626
    assert payload["partitions"]["validation"]["row_count"] == 1754
    assert payload["scientific_xgb_candidates_executed"] == 0
    assert payload["test_matrix_constructed"] is False
    for path_field, hash_field in (
        ("split_path", "split_sha256"), ("feature_schema_path", "feature_schema_sha256"),
        ("preprocessor_path", "preprocessor_sha256"), ("flat_feature_map_path", "flat_feature_map_sha256"),
        ("target_contract_path", "target_contract_sha256"), ("search_space_path", "search_space_sha256"),
    ):
        assert sha256_file(ROOT / payload[path_field]) == payload[hash_field]


def test_flat_map_is_deterministic_named_complete_and_target_free():
    payload = json.loads(MAP.read_text())
    validate_flat_feature_map(payload)
    assert payload == build_flat_feature_map(ROOT)
    assert payload["flat_dimension"] == 8 * 21 * 3 + 8 + 8 == 520
    assert payload["block_order"] == ["value", "observation_mask", "tslo", "padding", "static"]
    names = [item["flat_name"] for item in payload["entries"]]
    assert names[0] == "bin0__pao2__latest__value"
    assert names[-1] == "static__cardiac_condition_group==__UNKNOWN__"
    assert not any(name in {"f{}".format(index) for index in range(520)} for name in names)
    forbidden = ("subject_id", "stay_id", "prediction_time", "grid_index", "eligible", "target", "outtime", "latent")
    assert not any(token in name.lower() for token in forbidden for name in names)
    support = {"vasopressor_on", "invasive_ventilation_on", "norepinephrine_rate", "epinephrine_rate", "dopamine_rate", "dobutamine_rate"}
    assert sum(item["canonical_feature_name"] in support for item in payload["entries"]) == 8 * 6 * 3


def test_final_row_exact_flatten_unflatten_and_identity_exclusion():
    flat_map = json.loads(MAP.read_text())
    row = json.loads((ROOT / "artifacts/data/synthetic/phase10/final/synthetic_phase10_v1/train.jsonl").open().readline())
    vector, structured = flatten_row(row, flat_map)
    assert vector.dtype == np.float32 and vector.shape == (520,)
    assert_exact_reconstruction(structured, unflatten_row(vector, flat_map))
    before = vector.copy()
    row["subject_id"] = "ATTACKED"
    row["stay_id"] = "ATTACKED"
    row["prediction_time"] = "2999-01-01T00:00:00Z"
    row["grid_index"] = 99
    attacked, _ = flatten_row(row, flat_map)
    assert np.array_equal(before, attacked)


def test_one_factory_derives_lossless_gru_and_xgboost_views():
    flat_map = json.loads(MAP.read_text())
    row = json.loads((ROOT / "artifacts/data/synthetic/phase10/final/synthetic_phase10_v1/train.jsonl").open().readline())
    views = model_information_views(row, flat_map)
    assert set(views) == {"gru", "xgboost"}
    assert isinstance(views["gru"], StructuredModelInput)
    assert_exact_reconstruction(views["gru"], unflatten_row(views["xgboost"], flat_map))


def test_feature_reorder_missing_and_extra_attacks_fail():
    payload = json.loads(MAP.read_text())
    changed = json.loads(json.dumps(payload))
    changed["entries"][0], changed["entries"][1] = changed["entries"][1], changed["entries"][0]
    with pytest.raises(XGBCanonicalContractError):
        validate_flat_feature_map(changed)
    changed = json.loads(json.dumps(payload)); changed["entries"].pop(); changed["flat_dimension"] -= 1
    with pytest.raises(XGBCanonicalContractError):
        validate_flat_feature_map(changed)
    changed = json.loads(json.dumps(payload)); changed["entries"].append(dict(changed["entries"][-1], flat_index=520, flat_name="xgb_only")); changed["flat_dimension"] += 1
    with pytest.raises(XGBCanonicalContractError):
        validate_flat_feature_map(changed)


def test_scientific_gateway_hard_rejects_test_and_unknown_partitions():
    data = gateway()
    for name in ("test", "sealed_test", "all"):
        with pytest.raises(XGBCanonicalContractError, match="TEST ACCESS FORBIDDEN"):
            data.load_partition(name)


def test_task_eligibility_targets_and_phase10_scaler_are_reused():
    data = gateway(); train = data.load_partition("train")
    matrices = {name: data.task_matrix(train, name) for name in ("recovery24", "recovery48", "icu_stay_time", "organ_support")}
    assert [len(matrices[name].y) for name in matrices] == [5005, 1868, 8626, 5559]
    assert matrices["recovery24"].row_keys != matrices["recovery48"].row_keys
    raw = train.targets["recovery24"][train.eligibility["recovery24"]]
    expected = data.recovery_scaler.transform_horizon(torch.from_numpy(raw), 0).numpy()
    assert np.array_equal(matrices["recovery24"].y, expected)
    assert np.array_equal(matrices["icu_stay_time"].y, train.targets["icu_stay_time"])
    assert data.support_class_weight.pos_weight == pytest.approx(1.2736196319018405)


def test_labels_and_eligibility_cannot_change_x_and_row_order_aligns_by_identity():
    flat_map = json.loads(MAP.read_text())
    path = ROOT / "artifacts/data/synthetic/phase10/final/synthetic_phase10_v1/validation.jsonl"
    rows = [json.loads(line) for line in path.open()][:4]
    baseline = {tuple(row[key] for key in ("subject_id", "stay_id", "prediction_time", "grid_index")): flatten_row(row, flat_map)[0] for row in rows}
    attacked = list(reversed(rows))
    for row in attacked:
        row.update(delta_sofa_24=1e30, delta_sofa_48=-1e30, organ_support_label=1, recovery24_eligible=not row["recovery24_eligible"], split="test")
    actual = {tuple(row[key] for key in ("subject_id", "stay_id", "prediction_time", "grid_index")): flatten_row(row, flat_map)[0] for row in attacked}
    assert baseline.keys() == actual.keys()
    assert all(np.array_equal(baseline[key], actual[key]) for key in baseline)


@pytest.mark.parametrize(
    "field,value",
    (("max_depth", 2), ("learning_rate", .21), ("subsample", .59), ("colsample_bytree", 1.01), ("min_child_weight", 11), ("reg_alpha", 1.1), ("reg_lambda", .09), ("n_estimators", 1001)),
)
def test_xgb_candidate_range_validation_rejects_outside_values(field, value):
    config = XGBCandidateConfig(3, .1, .8, .8, 1, 0, 1, 8)
    with pytest.raises(XGBCandidateError):
        replace(config, **{field: value}).validate()


def _task(view, partition, X, y):
    keys = tuple((f"s-{partition}-{view}-{index}", f"e-{index}", "2026-01-01T00:00:00Z", index) for index in range(len(y)))
    return TaskMatrix(view, partition, keys, np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32))


@pytest.fixture(scope="module")
def smoke_bundles(tmp_path_factory):
    root = tmp_path_factory.mktemp("phase11-xgb-smoke")
    rng = np.random.default_rng(20260921)
    feature_names = tuple(f"feature_{index}" for index in range(12))
    X_train = rng.normal(size=(32, 12)).astype(np.float32)
    X_validation = rng.normal(size=(12, 12)).astype(np.float32)
    config = XGBCandidateConfig(3, .1, .9, .9, 1, 0, 1, 8)
    scaler = RecoveryTargetScaler(1.0, 2.0, -1.0, 4.0, 20, 16, "train", "d" * 64, "contract", "schema")
    weight = SupportClassWeight(32, 16, 16, 1.0, "train", "d" * 64, "label", "event", "2" * 64)
    scaler_path, scaler_sha256 = scaler.save(root / "recovery_target_scaler.json")
    weight_path, weight_sha256 = weight.save(root / "support_class_weight.json")
    lineage = XGBLineage("schema", "a" * 64, "map", "b" * 64, "pre", "c" * 64, "split", "d" * 64, "targets", "e" * 64, scaler_sha256, weight_sha256, "2" * 64, "e9dd63a" * 5 + "e9dd6")
    recovery_train = {"recovery24": _task("recovery24", "train", X_train[:28], rng.normal(size=28)), "recovery48": _task("recovery48", "train", X_train[:20], rng.normal(size=20))}
    recovery_val = {"recovery24": _task("recovery24", "validation", X_validation[:10], rng.normal(size=10)), "recovery48": _task("recovery48", "validation", X_validation[:8], rng.normal(size=8))}
    recovery = train_xgb_candidate(task="recovery", candidate_id="engineering_xgb_smoke_recovery", config=config, train_matrices=recovery_train, validation_matrices=recovery_val, output_directory=root / "recovery", lineage=lineage, feature_names=feature_names, seed=17, run_type=RUN_ENGINEERING_SMOKE, recovery_scaler=scaler, recovery_scaler_path=scaler_path)
    icu_train = {"icu_stay_time": _task("icu_stay_time", "train", X_train, np.abs(rng.normal(size=32)))}
    icu_val = {"icu_stay_time": _task("icu_stay_time", "validation", X_validation, np.abs(rng.normal(size=12)))}
    icu = train_xgb_candidate(task="icu_stay_time", candidate_id="engineering_xgb_smoke_icu", config=config, train_matrices=icu_train, validation_matrices=icu_val, output_directory=root / "icu", lineage=lineage, feature_names=feature_names, seed=17, run_type=RUN_ENGINEERING_SMOKE)
    y_train = np.asarray([0, 1] * 16, dtype=np.float32); y_val = np.asarray([0, 1] * 6, dtype=np.float32)
    support = train_xgb_candidate(task="organ_support", candidate_id="engineering_xgb_smoke_support", config=config, train_matrices={"organ_support": _task("organ_support", "train", X_train, y_train)}, validation_matrices={"organ_support": _task("organ_support", "validation", X_validation, y_val)}, output_directory=root / "support", lineage=lineage, feature_names=feature_names, seed=17, run_type=RUN_ENGINEERING_SMOKE, support_class_weight=weight, support_class_weight_path=weight_path)
    return root, X_validation, feature_names, lineage, scaler, weight, recovery, icu, support, scaler_path, weight_path


def test_fixture_smoke_fit_native_serialize_reload_and_task_outputs(smoke_bundles):
    root, X, names, lineage, scaler, weight, recovery, icu, support, scaler_path, weight_path = smoke_bundles
    loaded_recovery = load_xgb_bundle(root / "recovery", expected_task="recovery", expected_lineage=lineage, expected_feature_names=names, recovery_scaler=scaler, recovery_scaler_path=scaler_path)
    expected = recovery.predict(X)
    assert expected.shape == (12, 2)
    assert np.allclose(expected, loaded_recovery.predict(X))
    standardized = recovery.predict_standardized(X)
    assert np.allclose(expected[:, 0], standardized[:, 0] * scaler.scale_24 + scaler.mean_24)
    assert np.allclose(expected[:, 1], standardized[:, 1] * scaler.scale_48 + scaler.mean_48)
    loaded_icu = load_xgb_bundle(root / "icu", expected_task="icu_stay_time", expected_lineage=lineage, expected_feature_names=names)
    assert np.array_equal(loaded_icu.predict_hours(X), remaining_icu_hours_from_log_prediction(torch.from_numpy(loaded_icu.predict_log(X))).numpy())
    loaded_support = load_xgb_bundle(root / "support", expected_task="organ_support", expected_lineage=lineage, expected_feature_names=names, support_class_weight=weight, support_class_weight_path=weight_path)
    probability = loaded_support.predict(X)
    assert probability.shape == (12,) and np.all((probability >= 0) & (probability <= 1))
    assert np.array_equal(probability, loaded_support.model.predict_proba(X)[:, 1].astype(np.float32))
    assert loaded_support.calibrator is None and loaded_support.threshold is None and loaded_support.probability_type == "raw_uncalibrated"


def test_recovery_is_one_candidate_config_with_two_horizon_models(smoke_bundles):
    root, _, _, _, _, _, _, _, _, _, _ = smoke_bundles
    metadata = json.loads((root / "recovery/bundle.metadata.json").read_text())
    assert set(metadata["models"]) == {"recovery24", "recovery48"}
    assert metadata["candidate_id"] == "engineering_xgb_smoke_recovery"
    assert metadata["counts_toward_30_config_budget"] is False
    assert metadata["metrics_computed"] is False and metadata["test_accessed"] is False
    assert metadata["calibration"] is None and metadata["threshold"] is None
    assert metadata["config_sha256"] == config_sha256(XGBCandidateConfig(3, .1, .9, .9, 1, 0, 1, 8), task="recovery", seed=17, run_type=RUN_ENGINEERING_SMOKE)
    registry = xgb_registry_fields(metadata, metadata_ref="artifacts/smoke/bundle.metadata.json")
    assert registry["model_family"] == "xgboost"
    assert registry["derived_feature_hash"] == metadata["lineage"]["flat_feature_map_sha256"]
    assert json.loads(registry["notes"])["best_iterations"].keys() == {"recovery24", "recovery48"}


@pytest.mark.parametrize("field", (
    "feature_schema_sha256", "flat_feature_map_sha256", "preprocessor_sha256",
    "split_sha256", "target_schema_sha256", "recovery_target_scaler_sha256",
    "support_class_weight_sha256", "event_dictionary_sha256", "code_commit",
))
def test_every_declared_lineage_identity_is_enforced(smoke_bundles, field):
    root, _, names, lineage, scaler, _, _, _, _, scaler_path, _ = smoke_bundles
    with pytest.raises(XGBCandidateError, match="lineage"):
        load_xgb_bundle(
            root / "recovery", expected_task="recovery",
            expected_lineage=replace(lineage, **{field: "9" * 64}),
            expected_feature_names=names, recovery_scaler=scaler,
            recovery_scaler_path=scaler_path,
        )


def test_artifact_lineage_feature_order_task_and_model_tampering_fail(smoke_bundles, tmp_path):
    root, _, names, lineage, scaler, _, _, _, _, scaler_path, _ = smoke_bundles
    with pytest.raises(XGBCandidateError, match="feature"):
        load_xgb_bundle(root / "recovery", expected_task="recovery", expected_lineage=lineage, expected_feature_names=tuple(reversed(names)), recovery_scaler=scaler, recovery_scaler_path=scaler_path)
    with pytest.raises(XGBCandidateError, match="family/task"):
        load_xgb_bundle(root / "recovery", expected_task="icu_stay_time", expected_lineage=lineage, expected_feature_names=names)
    attacked = tmp_path / "recovery"; shutil.copytree(root / "recovery", attacked)
    model = attacked / "recovery_24h.json"; model.write_text(model.read_text() + " ")
    with pytest.raises(XGBCandidateError, match="model hash"):
        load_xgb_bundle(attacked, expected_task="recovery", expected_lineage=lineage, expected_feature_names=names, recovery_scaler=scaler, recovery_scaler_path=scaler_path)


def test_wrong_scaler_and_class_weight_artifact_bytes_are_rejected(smoke_bundles, tmp_path):
    root, _, names, lineage, scaler, weight, _, _, _, scaler_path, weight_path = smoke_bundles
    wrong_scaler = tmp_path / "wrong_scaler.json"
    shutil.copy2(scaler_path, wrong_scaler)
    wrong_scaler.write_text(wrong_scaler.read_text() + " ")
    with pytest.raises(XGBCandidateError, match="scaler artifact hash"):
        load_xgb_bundle(root / "recovery", expected_task="recovery", expected_lineage=lineage, expected_feature_names=names, recovery_scaler=scaler, recovery_scaler_path=wrong_scaler)
    wrong_weight = tmp_path / "wrong_weight.json"
    shutil.copy2(weight_path, wrong_weight)
    wrong_weight.write_text(wrong_weight.read_text() + " ")
    with pytest.raises(XGBCandidateError, match="class weight artifact hash"):
        load_xgb_bundle(root / "support", expected_task="organ_support", expected_lineage=lineage, expected_feature_names=names, support_class_weight=weight, support_class_weight_path=wrong_weight)


def test_registry_has_zero_scientific_xgb_candidates_and_no_selected_manifest_change():
    rows = list(csv.DictReader((ROOT / "experiments/registry.csv").open()))
    assert not [row for row in rows if row["model_family"] == "xgboost" and row["run_type"] == "scientific"]
    assert not (ROOT / "artifacts/search/real").exists()
    phase11 = json.loads(MANIFEST.read_text())
    assert phase11["search_executed"] is False
    assert phase11["model_selection_performed"] is False
    assert phase11["calibration_performed"] is False
    assert phase11["production_shap_executed"] is False
    assert phase11["serving_routing_changed"] is False
