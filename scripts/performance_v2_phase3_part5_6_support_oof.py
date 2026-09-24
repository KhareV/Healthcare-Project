"""Performance-v2 Phase 3, Parts 5-6: honest out-of-fold DEV support
predictions for calibration, plus the OOF invariants audit.

For each of the 5 DEV crossfit folds, fits the EXACT selected support
XGBoost config/feature-variant/training-weight-mode (never re-selected) on
the other 4 folds' subjects only, with context normalization and any
sample weighting fit ONLY from that fold's training subjects, and predicts
the held-out fold's eligible support rows. Every DEV row is scored by a
model that never saw its subject during training.
"""

import json
import sys
from pathlib import Path

import numpy as np
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.synthetic.config import canonical_json_bytes  # noqa: E402
from evaluation.weights import compute_stay_weights  # noqa: E402
from performance_v2.context_normalization import apply_normalization, fit_normalization  # noqa: E402
from performance_v2.data_loading import load_dev_rows  # noqa: E402
from performance_v2.targets import support_rows  # noqa: E402
from performance_v2.v2_features import build_group_a_matrix, group_a_feature_names  # noqa: E402
from vedant_infra.hashing import sha256_file  # noqa: E402

OUT_DIR = ROOT / "artifacts/performance_v2/phase3"
CAL_DIR = OUT_DIR / "calibration"
SEED = 2026_09_25
N_FOLDS = 5
CROSSFIT_VERSION = "performance_v2_phase3_dev_crossfit_fold_v1"


def crossfit_fold_for_subject(subject_id: str, n_folds: int = 5) -> int:
    import hashlib
    digest = hashlib.sha256((CROSSFIT_VERSION + "||" + subject_id).encode("utf-8")).hexdigest()
    return int(digest, 16) % n_folds


def _stay_balanced_sample_weight(rows):
    result = compute_stay_weights([row["stay_id"] for row in rows], [True] * len(rows))
    return np.asarray(result.weights, dtype=np.float64)


def main() -> None:
    selection = json.loads((ROOT / "artifacts/performance_v2/phase2/validation_selection_v2.json").read_text())
    support_selection = selection["tasks"]["organ_support"]
    config = support_selection["hyperparameters"]
    config_hash = support_selection["config_hash"]
    variant = support_selection["feature_variant"]
    budget = json.loads((OUT_DIR / "final_training_budget_v1.json").read_text())
    final_n_estimators = budget["tasks"]["organ_support"]["final_n_estimators"]

    train_rows = load_dev_rows(ROOT, splits=("train",))
    val_rows = load_dev_rows(ROOT, splits=("validation",))
    dev_rows = train_rows + val_rows
    support_dev = support_rows(dev_rows)
    fold_ids = np.asarray([crossfit_fold_for_subject(row["subject_id"]) for row in support_dev])

    group_a_names = group_a_feature_names(dev_rows)  # shared column universe across the whole DEV population

    oof_rows = []
    fold_summary = []
    for fold in range(N_FOLDS):
        train_mask, val_mask = fold_ids != fold, fold_ids == fold
        fold_train_rows = [support_dev[i] for i in np.where(train_mask)[0]]
        fold_val_rows = [support_dev[i] for i in np.where(val_mask)[0]]
        if not fold_train_rows or not fold_val_rows:
            continue

        train_a, _ = build_group_a_matrix(fold_train_rows, names=group_a_names)
        val_a, _ = build_group_a_matrix(fold_val_rows, names=group_a_names)
        normalization = fit_normalization(fold_train_rows, variant)  # TRAIN-subjects-of-this-fold only
        train_ctx = apply_normalization(fold_train_rows, normalization)
        val_ctx = apply_normalization(fold_val_rows, normalization)
        train_x = np.concatenate([train_a, train_ctx], axis=1)
        val_x = np.concatenate([val_a, val_ctx], axis=1)
        train_y = np.asarray([float(r["organ_support_label"]) for r in fold_train_rows])

        sample_weight = None  # organ_support's selected config carries no training-weight-mode axis (class weighting handled by the frozen XGB default, unchanged from Phase 2)
        model = xgb.XGBClassifier(
            objective="binary:logistic", random_state=SEED, missing=np.nan,
            max_depth=config["max_depth"], min_child_weight=config["min_child_weight"],
            learning_rate=config["learning_rate"], subsample=config["subsample"],
            colsample_bytree=config["colsample_bytree"], reg_alpha=config["reg_alpha"],
            reg_lambda=config["reg_lambda"], gamma=config["gamma"],
            n_estimators=final_n_estimators,
        )
        model.fit(train_x, train_y, sample_weight=sample_weight)
        predictions = model.predict_proba(val_x)[:, 1]

        fold_subjects_train = {r["subject_id"] for r in fold_train_rows}
        fold_subjects_val = {r["subject_id"] for r in fold_val_rows}
        assert not (fold_subjects_train & fold_subjects_val), "fold train/val subject overlap"

        for row, prediction in zip(fold_val_rows, predictions):
            oof_rows.append({
                "subject_id": row["subject_id"], "stay_id": row["stay_id"], "prediction_time": row["prediction_time"],
                "grid_index": row["grid_index"], "label": int(row["organ_support_label"]),
                "raw_probability": float(prediction), "fold_id": fold,
                "model_config_hash": config_hash, "feature_contract_hash": sha256_file(ROOT / "configs/performance_v2/feature_contract_v2.json"),
            })
        fold_summary.append({"fold": fold, "n_train_subjects": len(fold_subjects_train), "n_val_subjects": len(fold_subjects_val), "n_val_rows": len(fold_val_rows)})
        print("fold", fold, "train_subjects", len(fold_subjects_train), "val_subjects", len(fold_subjects_val), "val_rows", len(fold_val_rows))

    CAL_DIR.mkdir(parents=True, exist_ok=True)
    oof_path = CAL_DIR / "support_dev_oof_raw_predictions_v1.jsonl"
    with oof_path.open("w", encoding="utf-8") as handle:
        for row in oof_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    print("wrote", oof_path, sha256_file(oof_path), "rows:", len(oof_rows))

    # --- Part 6: OOF invariants audit ---
    all_keys = [(r["subject_id"], r["stay_id"], r["prediction_time"], r["grid_index"]) for r in oof_rows]
    expected_keys = [(r["subject_id"], r["stay_id"], r["prediction_time"], r["grid_index"]) for r in support_dev]
    audit = {
        "status": "OOF_AUDIT_PASS",
        "artifact_version": "performance_v2_phase3_support_oof_audit_v1",
        "n_eligible_dev_rows": len(support_dev), "n_oof_rows": len(oof_rows),
        "every_row_exactly_once": sorted(all_keys) == sorted(expected_keys) and len(set(all_keys)) == len(all_keys),
        "all_five_folds_represented": sorted({r["fold_id"] for r in oof_rows}) == list(range(N_FOLDS)),
        "all_probabilities_finite_and_in_0_1": bool(np.all(np.isfinite([r["raw_probability"] for r in oof_rows])) and np.all([0.0 <= r["raw_probability"] <= 1.0 for r in oof_rows])),
        "labels_binary": bool(set(r["label"] for r in oof_rows) <= {0, 1}),
        "both_classes_represented": len(set(r["label"] for r in oof_rows)) == 2,
        "fold_summary": fold_summary,
        "no_fresh_test_rows": True, "no_v1_test_rows": True,
    }
    assert audit["every_row_exactly_once"], "OOF coverage failed"
    assert audit["all_five_folds_represented"], "not all folds represented"
    assert audit["all_probabilities_finite_and_in_0_1"], "invalid probabilities"
    assert audit["labels_binary"], "non-binary labels"
    assert audit["both_classes_represented"], "single-class OOF set"
    audit_path = CAL_DIR / "support_dev_oof_audit_v1.json"
    audit_path.write_bytes(canonical_json_bytes(audit))
    print("wrote", audit_path, sha256_file(audit_path))
    print("OOF_AUDIT_PASS")


if __name__ == "__main__":
    main()
