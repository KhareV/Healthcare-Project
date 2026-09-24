# Performance V2 — Phase 3: Final DEV Refit, Calibration, and Sealed Fresh Test

Status at pre-generation freeze: **V2_MODEL_FREEZE_ACTIVE**. This phase made
no new model-selection decisions — every hyperparameter, objective, feature
variant, and training-weight mode below is copied verbatim from Phase 2's
`artifacts/performance_v2/phase2/validation_selection_v2.json`. No v1 TEST
row was read. The generator is verified byte-identical to Phase 1/2
(`5d7138ed93f3062c8b9fcd1913db28f243dcf4727a4a094d5860757761d8c4eb`).

**Correction from the Phase-2 handoff summary**: recovery48's selected
feature variant is **B_MIN**, not B_FULL as my own prior verbal summary
stated. `validation_selection_v2.json` (config_hash `078bb71e8659a241`)
and its source `finalist_metrics_xgb.json` entry both show
`feature_variant="B_MIN"` for the actual winning config (MAE48=1.5373,
beating the B_FULL finalist's 1.5592). Phase 3 binds the disk artifact, not
the earlier prose summary.

## 1. Selected v2 systems (frozen, unchanged from Phase 2)

| Task | Family | Feature variant | Objective | Weight mode |
|---|---|---|---|---|
| recovery24 | xgboost | B_MIN | reg:pseudohubererror | uniform |
| recovery48 | xgboost | B_MIN | reg:pseudohubererror | stay_balanced |
| icu_stay_time | xgboost (log1p) | B_PLUS_F | reg:absoluteerror | stay_balanced |
| organ_support | xgboost | B_FULL | binary:logistic | (none searched in Phase 2) |

## 2. DEV population

`artifacts/performance_v2/phase3/dev_subject_manifest_v1.json`: TRAIN=1400
+ VALIDATION=300 = **1700 DEV subjects**, zero overlap (verified from disk,
matching the expected counts). DEV crossfit folds (5-fold, SHA-256 of
`subject_id`, no label): `dev_crossfit_fold_assignment_v1.csv`, sizes
{0:332, 1:339, 2:375, 3:320, 4:334}.

## 3. Fixed final training budget

Neither Phase 2's CV results nor its VALIDATION finalist refit persisted a
genuine `best_iteration` (Phase 2's finalist refit used a fixed
`n_estimators=400`, no early stopping). Per the Phase-3 fallback rule, this
phase re-measured `best_iteration+1` on the exact Phase-2 TRAIN-CV folds
using the frozen selected config verbatim, with a generous ceiling
(`n_estimators<=4000`, `early_stopping_rounds=60`) so the reading is a
genuine convergence point rather than truncated by Phase 2's tighter
compute-bounded search caps — no hyperparameter was searched or changed.

| Task | Fold best_iterations | Final n_estimators (rounded median) |
|---|---|---:|
| recovery24 | [181, 237, 169, 169, 121] | **169** |
| recovery48 | [352, 410, 316, 291, 106] | **316** |
| icu_stay_time | [586, 759, 417, 422, 428] | **428** |
| organ_support | [189, 56, 55, 41, 199] | **56** |

## 4. Final V2 feature preprocessing

`artifacts/performance_v2/phase3/preprocessing/context_normalization_v2_{B_MIN,B_FULL,B_PLUS_F}.json`:
the same `context_v2.py` variant definitions and Group-A raw-canonical
representation as Phase 2, refit on all 1700 DEV subjects (not Phase 2's
TRAIN-only 1400) — information-equivalent successors, not redefinitions.
Group-A feature count: 510 (unchanged from Phase 1/2).

## 5. Support cross-fitted OOF predictions

`artifacts/performance_v2/phase3/calibration/support_dev_oof_raw_predictions_v1.jsonl`:
5-fold DEV crossfit, exact selected support config/feature-variant/budget,
context normalization and (none, since support has no weight-mode axis)
sample weighting fit only from each fold's training subjects. 6675 rows
total, fold sizes (held-out rows) 1239/1366/1431/1218/1421.

**OOF audit** (`support_dev_oof_audit_v1.json`, `status=OOF_AUDIT_PASS`):
every eligible DEV row appears exactly once; all 5 folds represented; every
probability finite and in [0,1]; labels binary; both classes present; zero
fresh-test/v1-test rows.

## 6. V2 support calibration and threshold

`isotonic_support_v2.json`: `sklearn.isotonic.IsotonicRegression(y_min=0,
y_max=1, out_of_bounds='clip')`, fit on the 6675 cross-fitted OOF
raw-probability/label pairs with stay-balanced weights (each eligible stay
contributes total weight 1). No calibration method comparison — this is
the final, frozen v2 method per spec.

`support_threshold_v2.json`: threshold selected by maximizing stay-balanced
DEV weighted F1 over the calibrated OOF predictions (candidates: sorted
unique calibrated probabilities + {0.0, 1.0}; tie-break: highest F1 →
closest to 0.5 → higher threshold).

**Frozen threshold: 0.39781983118092895** (comparator `>=`), **DEV weighted
F1 = 0.7148**.

## 7. Final full-DEV model refit

All four models trained on ALL 1700 DEV subjects' eligible rows, exact
Phase-2 hyperparameters, Phase-3 fixed `n_estimators`, no early stopping
(no holdout at this stage by design):

| Task | n_estimators | n DEV rows | Model SHA-256 (first 16) |
|---|---:|---:|---|
| recovery24 | 169 | 5973 | `b355c26339271f0b` |
| recovery48 | 316 | 2200 | `5dda6bb0830428b4` |
| icu_stay_time | 428 | 10380 | `918c81eef86bd89e` |
| organ_support | 56 | 6675 | `2fa653a0d2c644d8` |

**Reproducibility check** (`final_refit_manifest_v1.json`): every model's
pre-save and post-reload predictions on a 20-row DEV fixture match exactly
(`max_abs_diff < 1e-5` for all four) — serialization only, not a quality
check.

## 8. Selected-models manifest and prediction contract

`selected_models_v2.json` (`V2_SELECTED_MODELS_FROZEN_PRE_TEST`) binds
every task's family/config-hash/feature-variant/hyperparameters/training
budget/preprocessor/model hash, plus support's calibrator/threshold hashes,
plus parent hashes (Phase-1 strategy, Phase-2 selection, feature contract,
DEV manifest, DEV folds, generator config).

`prediction_contract_v2.json`: recovery — raw ΔSOFA, no clipping; ICU —
`expm1(clamp_min(raw_log1p, 0))`; support — raw probability →
`isotonic_support_v2` → calibrated probability → `support_threshold_v2`
(`>=`).

## 9. Pre-generation freeze and commit

`artifacts/performance_v2/governance/v2_model_freeze_v1.json`
(`V2_MODEL_FREEZE_ACTIVE`) binds every artifact above plus the generator
config hash and code commit, with `fresh_test_generated=false` and
`fresh_test_accessed=false`. This is committed and pushed to
`origin/performance-v2` **before** any fresh-cohort generation — the
auditable proof point that the model was fully frozen before the new
evaluation data existed.

## 10. Fresh v2 test cohort (generated after the freeze commit)

See the freeze-then-generate governance artifacts under
`artifacts/performance_v2/governance/` for the cohort spec, seed
derivation, generator-equivalence audit, and sealing. No target/outcome
distribution from the fresh cohort is reported anywhere in this phase's
artifacts or documentation — only structural integrity (subject counts,
hashes, schema, non-overlap with v1/DEV) was inspected, per the explicit
prohibition on outcome inspection before Phase 4.

## No v1 TEST access, no fresh-test inference

No code path in this phase reads a v1 `split=="test"` row (enforced by the
same `performance_v2.data_loading` choke point Phases 1-2 used) or performs
inference with any Phase-3 model against the fresh cohort. Zero final v2
predictions or metrics exist from fresh-cohort data at the end of this
phase — only the sealed cohort itself.

## Phase-4 readiness

Phase 4 will: implement the guarded fresh-test accessor's state transition
(`SEALED_NOT_ACCESSED → AUTHORIZED_FOR_ONE_FINAL_RUN → FINAL_V2_TEST_ACCESS_CONSUMED
→ FINAL_V2_RUN_COMPLETED`), run the four frozen `selected_models_v2`
models exactly once against the sealed fresh cohort, compute grouped-stay
bootstrap CIs, and freeze the final v2 evidence package. Nothing in Phase 3
performs any part of that.
