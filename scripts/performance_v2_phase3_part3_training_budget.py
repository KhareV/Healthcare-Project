"""Performance-v2 Phase 3, Part 3: fixed final XGBoost training budget
(n_estimators) per selected task, derived ONLY from existing Phase-2 TRAIN
development evidence -- never from fresh test, never a new hyperparameter
search (every hyperparameter/objective/feature-variant/weight-mode below is
copied verbatim from artifacts/performance_v2/phase2/validation_selection_v2.json).

Phase 2's own CV results did not persist per-fold best_iteration (only the
aggregate fold metric), and its VALIDATION finalist refit used a fixed
n_estimators=400 with no early stopping at all -- so neither of the
preferred rule's literal data sources exists on disk verbatim. This
re-runs ONLY the frozen selected config across the exact Phase-2 TRAIN-CV
folds (train_cv_fold_assignment_v1.csv), with a generous early-stopping
ceiling (well above anything Phase 2's own tighter, compute-bounded search
caps could have reached), purely to capture each fold's genuine
best_iteration -- no hyperparameter is searched or changed.
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
from performance_v2.data_loading import load_dev_rows  # noqa: E402
from performance_v2.search_utils import fold_of, load_fold_assignment  # noqa: E402
from performance_v2.targets import icu_hours_target, icu_rows, recovery_targets, support_rows  # noqa: E402
from performance_v2.v2_features import build_xgb_matrix_single  # noqa: E402
from vedant_infra.hashing import sha256_file  # noqa: E402

OUT_DIR = ROOT / "artifacts/performance_v2/phase3"
SEED = 2026_09_25
N_FOLDS = 5
GENEROUS_N_ESTIMATORS = 4000
GENEROUS_EARLY_STOPPING = 60


def _stay_balanced_sample_weight(rows):
    result = compute_stay_weights([row["stay_id"] for row in rows], [True] * len(rows))
    return np.asarray(result.weights, dtype=np.float64)


def _load_selection():
    return json.loads((ROOT / "artifacts/performance_v2/phase2/validation_selection_v2.json").read_text())


def _fold_best_iterations_regression(rows, targets, config, fold_assignment, log_space=False):
    fold_ids = np.asarray([fold_of(row, fold_assignment, "xgb_fold_5") for row in rows])
    x_all, names, _norm = build_xgb_matrix_single(rows, config["feature_variant"])
    y_all = np.log1p(np.asarray(targets)) if log_space else np.asarray(targets)
    best_iterations = []
    for fold in range(N_FOLDS):
        train_mask, val_mask = fold_ids != fold, fold_ids == fold
        if val_mask.sum() < 10 or train_mask.sum() < 10:
            continue
        train_x, val_x = x_all[train_mask], x_all[val_mask]
        train_y, val_y = y_all[train_mask], y_all[val_mask]
        fold_train_rows = [rows[i] for i in np.where(train_mask)[0]]
        sample_weight = _stay_balanced_sample_weight(fold_train_rows) if config.get("weight_mode") == "stay_balanced" else None
        model = xgb.XGBRegressor(
            objective=config["objective"], random_state=SEED, missing=np.nan,
            max_depth=config["max_depth"], min_child_weight=config["min_child_weight"],
            learning_rate=config["learning_rate"], subsample=config["subsample"],
            colsample_bytree=config["colsample_bytree"], reg_alpha=config["reg_alpha"],
            reg_lambda=config["reg_lambda"], gamma=config["gamma"],
            n_estimators=GENEROUS_N_ESTIMATORS, early_stopping_rounds=GENEROUS_EARLY_STOPPING, eval_metric="mae",
        )
        model.fit(train_x, train_y, sample_weight=sample_weight, eval_set=[(val_x, val_y)], verbose=False)
        best_iterations.append(int(model.best_iteration) + 1)
    return best_iterations


def _fold_best_iterations_classification(rows, labels, config, fold_assignment):
    fold_ids = np.asarray([fold_of(row, fold_assignment, "xgb_fold_5") for row in rows])
    x_all, names, _norm = build_xgb_matrix_single(rows, config["feature_variant"])
    y_all = np.asarray(labels, dtype=np.float64)
    best_iterations = []
    for fold in range(N_FOLDS):
        train_mask, val_mask = fold_ids != fold, fold_ids == fold
        if val_mask.sum() < 10 or train_mask.sum() < 10:
            continue
        train_x, val_x = x_all[train_mask], x_all[val_mask]
        train_y, val_y = y_all[train_mask], y_all[val_mask]
        model = xgb.XGBClassifier(
            objective="binary:logistic", random_state=SEED, missing=np.nan, eval_metric="aucpr",
            max_depth=config["max_depth"], min_child_weight=config["min_child_weight"],
            learning_rate=config["learning_rate"], subsample=config["subsample"],
            colsample_bytree=config["colsample_bytree"], reg_alpha=config["reg_alpha"],
            reg_lambda=config["reg_lambda"], gamma=config["gamma"],
            n_estimators=GENEROUS_N_ESTIMATORS, early_stopping_rounds=GENEROUS_EARLY_STOPPING,
        )
        model.fit(train_x, train_y, eval_set=[(val_x, val_y)], verbose=False)
        best_iterations.append(int(model.best_iteration) + 1)
    return best_iterations


def _aggregate(fold_iterations):
    return int(round(float(np.median(fold_iterations))))


def main() -> None:
    selection = _load_selection()
    fold_assignment = load_fold_assignment()
    train_rows = load_dev_rows(ROOT, splits=("train",))

    budget = {
        "status": "FINAL_TRAINING_BUDGET_FROZEN",
        "artifact_version": "performance_v2_phase3_final_training_budget_v1",
        "aggregation_rule": "rounded median of per-fold best_iteration+1, re-measured on the exact Phase-2 TRAIN-CV folds (artifacts/performance_v2/phase2/train_cv_fold_assignment_v1.csv, xgb_fold_5) using the frozen selected hyperparameters/objective/weight_mode/feature_variant verbatim, with a generous early-stopping ceiling (n_estimators<=4000, early_stopping_rounds=60) so the measured best_iteration is a genuine convergence point rather than truncated by Phase 2's tighter compute-bounded search caps. No hyperparameter was changed or searched.",
        "generous_ceiling": {"n_estimators": GENEROUS_N_ESTIMATORS, "early_stopping_rounds": GENEROUS_EARLY_STOPPING},
        "tasks": {},
    }

    for horizon in ("24", "48"):
        task_key = "recovery" + horizon
        config = selection["tasks"][task_key]["hyperparameters"] if "hyperparameters" in selection["tasks"][task_key] else selection["tasks"][task_key]["config"]
        rows = recovery_targets(train_rows, horizon)
        targets = [float(r["delta_sofa_" + horizon]) for r in rows]
        fold_iterations = _fold_best_iterations_regression(rows, targets, config, fold_assignment, log_space=False)
        final_n = _aggregate(fold_iterations)
        budget["tasks"][task_key] = {
            "source_config_hash": selection["tasks"][task_key]["config_hash"], "source_config": config,
            "fold_best_iterations": fold_iterations, "final_n_estimators": final_n,
        }
        print(task_key, "fold best_iterations", fold_iterations, "-> final_n_estimators", final_n)

    icu_config = selection["tasks"]["icu_stay_time"]["config"]
    icu_train = icu_rows(train_rows)
    icu_hours = [icu_hours_target(r) for r in icu_train]
    icu_fold_iterations = _fold_best_iterations_regression(icu_train, icu_hours, icu_config, fold_assignment, log_space=True)
    icu_final_n = _aggregate(icu_fold_iterations)
    budget["tasks"]["icu_stay_time"] = {
        "source_config_hash": selection["tasks"]["icu_stay_time"]["config_hash"], "source_config": icu_config,
        "fold_best_iterations": icu_fold_iterations, "final_n_estimators": icu_final_n,
    }
    print("icu_stay_time fold best_iterations", icu_fold_iterations, "-> final_n_estimators", icu_final_n)

    support_config = selection["tasks"]["organ_support"]["hyperparameters"]
    support_train = support_rows(train_rows)
    support_labels = [float(r["organ_support_label"]) for r in support_train]
    support_fold_iterations = _fold_best_iterations_classification(support_train, support_labels, support_config, fold_assignment)
    support_final_n = _aggregate(support_fold_iterations)
    budget["tasks"]["organ_support"] = {
        "source_config_hash": selection["tasks"]["organ_support"]["config_hash"], "source_config": support_config,
        "fold_best_iterations": support_fold_iterations, "final_n_estimators": support_final_n,
    }
    print("organ_support fold best_iterations", support_fold_iterations, "-> final_n_estimators", support_final_n)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "final_training_budget_v1.json"
    out_path.write_bytes(canonical_json_bytes(budget))
    print("wrote", out_path, sha256_file(out_path))


if __name__ == "__main__":
    main()
