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

**Seed derivation** (fixed before any generation; never retried):
`seed_material = "performance_v2_fresh_test_v1|" + SHA256(selected_models_v2.json) + "|1000"`
→ `fresh_test_seed = int(SHA256(seed_material).hexdigest()[:8], 16)` = **402995653**.
Recorded in `configs/performance_v2/fresh_test_cohort_v1.json` and
`artifacts/performance_v2/phase3/fresh_test_cohort_spec_v1.json`.

**Cohort**: 1000 subjects, evaluation-only (`fresh_v2_test`), no
train/validation subset. Distinct namespace `SYN-V2-S-*` / `SYN-V2-E-*`, so
subject/stay IDs cannot string-collide with v1's `SYN-S-*`/`SYN-E-*`.

**Frozen generator code, minimally extended, not shortcut**: the generator's
`config.py` originally hardcoded a single authorized final-mode identity
(seed 20260921, 2000 subjects). It now holds an explicit two-entry frozen
allow-list — v1's pair unchanged, plus one new pair for this cohort — each
bound 1:1 to a required `subject_id_namespace`. `generator.py`'s subject/
episode ID prefix is now read from that namespace (default `"SYN"`, so v1 is
byte-identical). No latent-process, SOFA, support-process, observation, or
label-building code was touched (see `generator_equivalence_audit_v1.json`
for the full unchanged-file hash list and the exact narrow diff). Re-running
the unmodified `final_benchmark_v1.json` config through the current code
reproduced `subjects.jsonl`/`episodes.jsonl`/`support_intervals.jsonl`
byte-for-byte; `raw_events.jsonl` matched 670298/670303 lines exactly, with
the remaining 5 differing only in the final microsecond digit of
`event_time` (environment float-rounding, verified deterministic across two
independent re-runs — not a scientific or code-logic change).

**Build pipeline**: the same accepted multi-stage pipeline as v1 —
`generate_synthetic_dataset.py` → `build_synthetic_cohort.py` →
`build_synthetic_timeline.py` → `build_synthetic_features.py` →
`build_phase9_final_package.py` — run mode-for-mode identical (`final`),
against the new additive `configs/performance_v2/fresh_test_cohort_v1.json`
(inherits the same scientific parameter file as v1; only seed, n_subjects,
namespace, and provenance metadata differ — see the generator-equivalence
audit for the full permitted-diff list).

**Structural sealing only**: `fresh_test_cohort_structural_audit_v1.json`
confirms 1000 unique namespaced subjects/stays, zero v1 subject-ID overlap,
zero clone-fingerprint overlap (via the same identity-free raw-trajectory
fingerprint used for v1's own split), unique canonical row keys, and
required eligibility fields present. No recovery/ICU/support target value,
distribution, or count from the fresh cohort was read, printed, or
inspected anywhere in Phase 3.

**Split contract**: `fresh_test_split_v1.csv` (`subject_id,split`, every row
`split=test`), `fresh_test_split_manifest_v1.json` status
`FROZEN_V2_EVALUATION_ONLY_COHORT`. No train/validation subset was created.

**Sealing**: `artifacts/performance_v2/governance/v2_fresh_test_access_state.json`
= `SEALED_NOT_ACCESSED`; `v2_fresh_test_freeze_v1.json` status
`V2_FRESH_TEST_FROZEN`, binding the cohort config, seed, generator-equivalence
audit, every pipeline-stage manifest, the structural audit, the split
contract, `selected_models_v2.json`, `v2_model_freeze_v1.json`, and the
pre-generation freeze commit `393414f987161a8f2f4cb9728329b2b21021badd`.

**Access choke point**: `src/performance_v2/fresh_test_access.py` is the
only function in the repository permitted to return fresh-test rows; it
refuses while the access state is `SEALED_NOT_ACCESSED` (verified by test).
Only a guarded, not-yet-implemented Phase-4 command may advance
`SEALED_NOT_ACCESSED → AUTHORIZED_FOR_ONE_FINAL_RUN → FINAL_V2_TEST_ACCESS_CONSUMED
→ FINAL_V2_RUN_COMPLETED`. Phase 3 never calls any transition function.

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
