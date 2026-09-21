import csv
import json
from copy import deepcopy
from pathlib import Path

from data.synthetic.provenance import sha256_file
from preprocess.synthetic_phase10 import (
    FrozenSyntheticFeaturePreprocessor,
    Phase10Config,
    assign_subjects,
    fit_preprocessing,
    transform_feature_view,
)


ROOT = Path(__file__).resolve().parents[1]
SPLIT = ROOT / "artifacts/splits/synthetic_split_v2.csv"
PREPROCESSOR = ROOT / "artifacts/preprocessors/synthetic_feature_preprocessor_v1.json"
MANIFEST = ROOT / "artifacts/manifests/synthetic_phase10_manifest_v1.json"


def test_phase10_frozen_artifacts_have_integrity_and_sealed_test():
    manifest = json.loads(MANIFEST.read_text())
    assert manifest["status"] == "PHASE10_COMPLETE"
    assert manifest["subject_counts"] == {"train": 1400, "validation": 300, "test": 300}
    assert sum(manifest["row_counts"].values()) == 12222
    assert manifest["test_opened"] is False
    assert manifest["test_label_statistics_computed"] is False
    assert manifest["model_training_performed"] is False
    for item in manifest["artifacts"]:
        assert sha256_file(ROOT / item["path"]) == item["sha256"]
    assert not any(item["logical_name"] == "model_ready_test" for item in manifest["artifacts"])
    assert manifest["test_model_ready_artifact"] == "NOT_MATERIALIZED_UNTIL_G3_EQUIVALENT_FREEZE"
    assert not (ROOT / "artifacts/data/synthetic/phase10/final/synthetic_phase10_v1/test.jsonl").exists()


def test_subject_and_clone_isolation_and_train_fit_inventory():
    with SPLIT.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    groups = {name: {row["subject_id"] for row in rows if row["split"] == name} for name in ("train", "validation", "test")}
    assert tuple(map(len, groups.values())) == (1400, 300, 300)
    assert not groups["train"] & groups["validation"]
    assert not groups["train"] & groups["test"]
    assert not groups["validation"] & groups["test"]
    clone_partitions = {}
    for row in rows:
        clone_partitions.setdefault(row["clone_fingerprint_sha256"], set()).add(row["split"])
    assert all(len(partitions) == 1 for partitions in clone_partitions.values())
    fit_ids = set((ROOT / "artifacts/preprocessors/synthetic_train_fit_subjects_v1.txt").read_text().splitlines())
    assert fit_ids == groups["train"]


def test_assignment_is_order_invariant_label_blind_and_clone_safe():
    config = Phase10Config("v", 17, "test", {"train": .7, "validation": .15, "test": .15}, "UNK", 54.0, 0.0)
    subjects = [f"s{i}" for i in range(20)]
    fingerprints = {subject: f"fp-{subject}" for subject in subjects}
    fingerprints["s1"] = fingerprints["s0"]
    first, counts, clones = assign_subjects(subjects, fingerprints, config)
    second, _, _ = assign_subjects(reversed(subjects), fingerprints, config)
    assert first == second
    assert first["s0"] == first["s1"]
    assert clones == 1
    assert sum(counts.values()) == 20


def _toy_rows():
    common = {
        "temporal_feature_names": ["a", "b"], "static_feature_names": ["age_years", "sex_category", "cardiac_condition_group"],
        "feature_schema_sha256": "f" * 64, "split": None, "preprocessor": None,
        "padding_mask": [True, False], "observation_mask": [[False, False], [True, False]],
        "tslo_hours": [[54.0, 54.0], [0.0, 54.0]],
        "history_values": [[None, None], [1.0, None]], "static_features": [50, "F", "HF"],
        "delta_sofa_24": 1.0, "recovery24_eligible": True, "delta_sofa_48": 2.0, "recovery48_eligible": True,
        "organ_support_label": 1, "organ_support_eligible": True,
    }
    train2 = deepcopy(common); train2.update(subject_id="train2", history_values=[[None, None], [3.0, 5.0]], static_features=[70, "M", "ISC"], delta_sofa_24=3.0, delta_sofa_48=4.0, organ_support_label=0)
    train1 = deepcopy(common); train1["subject_id"] = "train1"
    validation = deepcopy(common); validation.update(subject_id="validation", history_values=[[None, None], [9999.0, 9999.0]], static_features=[999, "X", "X"], delta_sofa_24=999.0, delta_sofa_48=999.0)
    return [train1, train2, validation]


def test_validation_perturbation_cannot_change_fitted_statistics():
    rows = _toy_rows()
    assignments = {"train1": "train", "train2": "train", "validation": "validation"}
    config = Phase10Config("v", 1, "n", {"train": .7, "validation": .15, "test": .15}, "UNK", 54.0, 0.0)
    first = fit_preprocessing(rows, assignments, config, feature_schema_sha256="f" * 64, split_sha256="a" * 64, fit_subjects_sha256="b" * 64)
    attacked = deepcopy(rows)
    attacked[-1]["history_values"][1][0] = -1e30
    attacked[-1]["delta_sofa_24"] = -1e30
    attacked[-1]["organ_support_label"] = 0
    second = fit_preprocessing(attacked, assignments, config, feature_schema_sha256="f" * 64, split_sha256="a" * 64, fit_subjects_sha256="b" * 64)
    assert first == second


def test_training_and_serving_use_the_identical_frozen_transform():
    expected = json.loads((ROOT / "artifacts/data/synthetic/phase10/final/synthetic_phase10_v1/train.jsonl").open().readline())
    with (ROOT / "artifacts/data/synthetic/pre_split/final/phase9_final_v1/pre_split_scientific_package.jsonl").open() as handle:
        raw = None
        for line in handle:
            candidate = json.loads(line)
            if (candidate["stay_id"], candidate["prediction_time"]) == (expected["stay_id"], expected["prediction_time"]):
                raw = candidate
                break
    assert raw is not None
    view = {name: raw[name] for name in ("history_values", "observation_mask", "tslo_hours", "padding_mask", "static_features", "static_feature_names")}
    serving = FrozenSyntheticFeaturePreprocessor(PREPROCESSOR).transform(view)
    payload = json.loads(PREPROCESSOR.read_text())
    direct = transform_feature_view(view, payload)
    assert serving == direct
    assert serving["history_values"] == expected["sequence_values"]
    assert serving["static_features"] == expected["static_features"]
    assert all(value == 0.0 for index, row in enumerate(serving["history_values"]) if raw["padding_mask"][index] for value in row)


def test_preprocessor_records_training_only_provenance_and_no_fake_years():
    payload = json.loads(PREPROCESSOR.read_text())
    metadata = json.loads((ROOT / "artifacts/splits/synthetic_split_v2.metadata.json").read_text())
    assert payload["fit_partition"] == "train"
    assert payload["validation_or_test_statistics_used"] is False
    assert payload["fit_subject_count"] == 1400
    assert metadata["labels_or_performance_used"] is False
    assert metadata["fake_calendar_fields_used"] is False
    assert "anchor_year" not in SPLIT.read_text()
    contract = json.loads((ROOT / "configs/synthetic/model_input_contract_v1.json").read_text())
    assert contract["static_features"]["ordered_names"] == payload["encoded_static_feature_names"]
    assert contract["information_parity"]["xgboost_flat_dimension"] == 520
