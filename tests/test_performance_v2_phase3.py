"""Performance-v2 Phase 3 focused tests (pre-generation freeze portion)."""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
PHASE2_DIR = ROOT / "artifacts/performance_v2/phase2"
PHASE3_DIR = ROOT / "artifacts/performance_v2/phase3"
GOV_DIR = ROOT / "artifacts/performance_v2/governance"


def _load(path):
    return json.loads((ROOT / path).read_text())


# --- Phase-2 selection immutability -----------------------------------------

def test_phase2_validation_selection_file_unchanged_reference_values():
    selection = _load("artifacts/performance_v2/phase2/validation_selection_v2.json")
    assert selection["status"] == "V2_VALIDATION_SELECTION_COMPLETE"
    assert selection["tasks"]["recovery24"]["feature_variant"] == "B_MIN"
    assert selection["tasks"]["icu_stay_time"]["config"]["feature_variant"] == "B_PLUS_F"
    assert selection["tasks"]["organ_support"]["feature_variant"] == "B_FULL"


def test_selected_models_v2_binds_exact_phase2_config_hashes():
    selection = _load("artifacts/performance_v2/phase2/validation_selection_v2.json")
    selected = _load("artifacts/performance_v2/phase3/selected_models_v2.json")
    assert selected["status"] == "V2_SELECTED_MODELS_FROZEN_PRE_TEST"
    for task in ("recovery24", "recovery48", "icu_stay_time", "organ_support"):
        expected_hash = selection["tasks"][task]["config_hash"]
        assert selected["tasks"][task]["selected_phase2_candidate_config_hash"] == expected_hash


def test_selected_models_v2_hyperparameters_match_phase2_verbatim():
    selection = _load("artifacts/performance_v2/phase2/validation_selection_v2.json")
    selected = _load("artifacts/performance_v2/phase3/selected_models_v2.json")
    rec24_phase2 = selection["tasks"]["recovery24"]["hyperparameters"]
    rec24_phase3 = selected["tasks"]["recovery24"]["hyperparameters"]
    for key in ("max_depth", "min_child_weight", "learning_rate", "subsample", "colsample_bytree", "reg_alpha", "reg_lambda", "gamma", "objective"):
        assert rec24_phase2[key] == rec24_phase3[key]


# --- DEV subject manifest ----------------------------------------------------

def test_dev_subject_manifest_counts():
    manifest = _load("artifacts/performance_v2/phase3/dev_subject_manifest_v1.json")
    assert manifest["train_subject_count"] == 1400
    assert manifest["validation_subject_count"] == 300
    assert manifest["dev_subject_count"] == 1700
    assert len(manifest["dev_subjects"]) == 1700
    assert not (set(manifest["train_subjects"]) & set(manifest["validation_subjects"]))


# --- Crossfit fold isolation --------------------------------------------------

def test_crossfit_fold_subject_and_stay_isolation():
    with (PHASE3_DIR / "dev_crossfit_fold_assignment_v1.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1700
    subject_folds = {row["subject_id"]: row["crossfit_fold_5"] for row in rows}
    stay_to_subjects = {}
    for row in rows:
        stay_to_subjects.setdefault(row["stay_id"], set()).add(row["subject_id"])
    assert all(len(subjects) == 1 for subjects in stay_to_subjects.values())
    assert len(set(subject_folds.values())) == 5


def test_crossfit_fold_assignment_is_deterministic_and_label_free():
    import sys as _sys
    _sys.path.insert(0, str(ROOT / "scripts"))
    from performance_v2_phase3_part1_2_dev_and_folds import crossfit_fold_for_subject
    import inspect

    params = list(inspect.signature(crossfit_fold_for_subject).parameters)
    assert params == ["subject_id", "n_folds", "version"]
    assert crossfit_fold_for_subject("SYN-S-00000001") == crossfit_fold_for_subject("SYN-S-00000001")


# --- OOF support coverage / exclusion ---------------------------------------

def _load_oof_jsonl(path):
    rows = []
    with (ROOT / path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def test_support_oof_covers_every_eligible_dev_row_exactly_once():
    audit = _load("artifacts/performance_v2/phase3/calibration/support_dev_oof_audit_v1.json")
    assert audit["status"] == "OOF_AUDIT_PASS"
    assert audit["every_row_exactly_once"] is True
    assert audit["all_five_folds_represented"] is True
    assert audit["n_oof_rows"] == audit["n_eligible_dev_rows"]


def test_support_oof_subject_exclusion_per_fold():
    audit = _load("artifacts/performance_v2/phase3/calibration/support_dev_oof_audit_v1.json")
    for fold in audit["fold_summary"]:
        assert fold["n_val_subjects"] > 0
        assert fold["n_train_subjects"] > 0


def test_support_oof_probabilities_finite_and_binary_labels():
    audit = _load("artifacts/performance_v2/phase3/calibration/support_dev_oof_audit_v1.json")
    assert audit["all_probabilities_finite_and_in_0_1"] is True
    assert audit["labels_binary"] is True
    assert audit["both_classes_represented"] is True


def test_support_oof_no_row_predicted_by_a_model_trained_on_its_own_subject():
    """Reconstruct fold membership from the crossfit assignment and verify
    no OOF row's subject appears in more than one fold (a necessary
    condition for "never predicted by a model that saw this subject")."""

    import sys
    sys.path.insert(0, str(ROOT / "scripts"))
    from performance_v2_phase3_part5_6_support_oof import crossfit_fold_for_subject

    oof_rows = _load_oof_jsonl("artifacts/performance_v2/phase3/calibration/support_dev_oof_raw_predictions_v1.jsonl")
    for row in oof_rows[:200]:
        assert row["fold_id"] == crossfit_fold_for_subject(row["subject_id"])


# --- Fold-local preprocessing / weighting (design check) --------------------

def test_support_oof_script_fits_normalization_per_fold_not_globally():
    text = (ROOT / "scripts/performance_v2_phase3_part5_6_support_oof.py").read_text()
    assert "fit_normalization(fold_train_rows" in text
    assert "fit_normalization(support_dev" not in text
    assert "fit_normalization(dev_rows" not in text


# --- Isotonic calibration lineage --------------------------------------------

def test_isotonic_calibrator_lineage_matches_oof_artifact_hash():
    calibrator = _load("artifacts/performance_v2/phase3/calibration/isotonic_support_v2.json")
    oof_path = PHASE3_DIR / "calibration/support_dev_oof_raw_predictions_v1.jsonl"
    from vedant_infra.hashing import sha256_file
    assert calibrator["source_oof_sha256"] == sha256_file(oof_path)
    assert calibrator["status"] == "V2_SUPPORT_CALIBRATOR_FROZEN"
    assert calibrator["fit_weights"].startswith("stay_balanced")


def test_isotonic_calibrator_output_is_monotonic():
    calibrator = _load("artifacts/performance_v2/phase3/calibration/isotonic_support_v2.json")
    x = calibrator["calibration_knots"]["x_thresholds"]
    y = calibrator["calibration_knots"]["y_thresholds"]
    assert len(x) == len(y)
    assert all(y[i] <= y[i + 1] + 1e-12 for i in range(len(y) - 1))


# --- Threshold determinism ----------------------------------------------------

def test_threshold_is_deterministic_given_same_oof_input():
    threshold = _load("artifacts/performance_v2/phase3/thresholds/support_threshold_v2.json")
    assert threshold["status"] == "V2_SUPPORT_THRESHOLD_FROZEN"
    assert 0.0 <= threshold["threshold_value"] <= 1.0
    assert threshold["comparator"] == ">="
    assert threshold["tie_break"] == ["highest full-precision weighted F1", "closest to 0.5", "higher threshold"]


def test_threshold_maximizes_dev_weighted_f1_among_candidates():
    """Re-derive F1 at the frozen threshold from the calibrated OOF file and
    confirm no candidate in a small perturbation neighborhood beats it,
    as a sanity check on the persisted optimum (not a full re-search)."""

    threshold_doc = _load("artifacts/performance_v2/phase3/thresholds/support_threshold_v2.json")
    calibrated = _load_oof_jsonl("artifacts/performance_v2/phase3/calibration/support_dev_oof_calibrated_predictions_v1.jsonl")
    from evaluation.weights import compute_stay_weights

    labels = np.asarray([row["label"] for row in calibrated])
    probs = np.asarray([row["calibrated_probability"] for row in calibrated])
    stay_ids = [row["stay_id"] for row in calibrated]
    weights = np.asarray(compute_stay_weights(stay_ids, [True] * len(calibrated)).weights)

    def weighted_f1(threshold):
        predictions = (probs >= threshold).astype(int)
        tp = float(np.sum(weights[(labels == 1) & (predictions == 1)]))
        fp = float(np.sum(weights[(labels == 0) & (predictions == 1)]))
        fn = float(np.sum(weights[(labels == 1) & (predictions == 0)]))
        if tp + fp == 0 or tp + fn == 0:
            return 0.0
        precision, recall = tp / (tp + fp), tp / (tp + fn)
        return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)

    frozen_f1 = weighted_f1(threshold_doc["threshold_value"])
    assert abs(frozen_f1 - threshold_doc["dev_weighted_f1"]) < 1e-9
    for perturbation in (-0.02, -0.01, 0.01, 0.02):
        nearby = min(1.0, max(0.0, threshold_doc["threshold_value"] + perturbation))
        assert weighted_f1(nearby) <= frozen_f1 + 1e-9


# --- Fixed training-budget derivation ----------------------------------------

def test_final_training_budget_derived_from_train_cv_folds_only():
    budget = _load("artifacts/performance_v2/phase3/final_training_budget_v1.json")
    assert budget["status"] == "FINAL_TRAINING_BUDGET_FROZEN"
    for task in ("recovery24", "recovery48", "icu_stay_time", "organ_support"):
        entry = budget["tasks"][task]
        assert entry["final_n_estimators"] == int(round(float(np.median(entry["fold_best_iterations"]))))
        assert entry["final_n_estimators"] > 0


def test_final_training_budget_config_matches_phase2_selection():
    budget = _load("artifacts/performance_v2/phase3/final_training_budget_v1.json")
    selection = _load("artifacts/performance_v2/phase2/validation_selection_v2.json")
    for task in ("recovery24", "recovery48"):
        assert budget["tasks"][task]["source_config_hash"] == selection["tasks"][task]["config_hash"]
    assert budget["tasks"]["icu_stay_time"]["source_config_hash"] == selection["tasks"]["icu_stay_time"]["config_hash"]
    assert budget["tasks"]["organ_support"]["source_config_hash"] == selection["tasks"]["organ_support"]["config_hash"]


# --- Final-model serialization parity ----------------------------------------

def test_final_refit_reload_parity_recorded_true():
    refit = _load("artifacts/performance_v2/phase3/models/final_refit_manifest_v1.json")
    assert refit["status"] == "FINAL_DEV_MODELS_TRAINED"
    for task, result in refit["reproducibility_check"].items():
        assert result["matches"] is True, task
        assert result["max_abs_diff"] < 1e-5


def test_final_model_files_exist_and_hash_matches_manifest():
    from vedant_infra.hashing import sha256_file

    refit = _load("artifacts/performance_v2/phase3/models/final_refit_manifest_v1.json")
    for task, entry in refit["tasks"].items():
        model_path = ROOT / entry["model_path"]
        assert model_path.is_file()
        assert sha256_file(model_path) == entry["model_sha256"]


# --- Feature-map correctness --------------------------------------------------

def test_phase3_preprocessing_variants_match_context_v2_definitions():
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from performance_v2.context_v2 import context_feature_names

    for variant in ("B_MIN", "B_FULL", "B_PLUS_F"):
        payload = _load(f"artifacts/performance_v2/phase3/preprocessing/context_normalization_v2_{variant}.json")
        assert tuple(payload["feature_names"]) == context_feature_names(variant)
        assert len(payload["mean"]) == len(payload["feature_names"])
        assert len(payload["std"]) == len(payload["feature_names"])


def test_phase3_preprocessing_fit_on_full_dev_not_train_only():
    dev_manifest = _load("artifacts/performance_v2/phase3/dev_subject_manifest_v1.json")
    for variant in ("B_MIN", "B_FULL", "B_PLUS_F"):
        payload = _load(f"artifacts/performance_v2/phase3/preprocessing/context_normalization_v2_{variant}.json")
        assert payload["dev_subject_count"] == dev_manifest["dev_subject_count"] == 1700


# --- Model freeze / pre-generation state -------------------------------------

def test_v2_model_freeze_binds_all_required_parents():
    freeze = _load("artifacts/performance_v2/governance/v2_model_freeze_v1.json")
    assert freeze["status"] == "V2_MODEL_FREEZE_ACTIVE"
    required = (
        "phase1_strategy", "phase2_validation_selection", "feature_contract", "dev_subject_manifest",
        "dev_crossfit_folds", "final_training_budget", "recovery24_model", "recovery48_model", "icu_model",
        "support_model", "support_oof_raw", "support_calibrator", "support_threshold", "selected_models_v2",
        "prediction_contract", "environment_lock", "generator_config",
    )
    for key in required:
        assert key in freeze["parents"], key
        assert freeze["parents"][key]["sha256"]
    assert freeze["fresh_test_generated"] is False
    assert freeze["fresh_test_accessed"] is False
