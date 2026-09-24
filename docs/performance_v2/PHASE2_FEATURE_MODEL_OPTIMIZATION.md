# Performance V2 — Phase 2: Governed Feature Upgrade + Robust Train-Only Model Search + Validation Selection

Status: **DEVELOPMENT_VALIDATION_ONLY**. Nothing here is a final serving
artifact or a final model freeze (Phase 3 owns the final TRAIN+VALIDATION
DEV refit, cross-fitted support calibration, and the fresh v2 test cohort).
No Benchmark-v1 artifact was modified. No v1 TEST row was read. The
generator is verified byte-identical to Phase 1 (`generator_config_sha256
= 5d7138ed93f3062c8b9fcd1913db28f243dcf4727a4a094d5860757761d8c4eb`,
matching both before and after this phase).

Branch: `performance-v2`, continuing from the Phase-1 commit.

## 1. Feature contract

`configs/performance_v2/feature_contract_v2.json` extends (never modifies)
`configs/synthetic/feature_schema_v2.json`. The V2Example adds one explicit,
versioned field alongside the unchanged 8×21 temporal history/mask/TSLO and
8 statics: a **V2 cutoff-context vector**, built by
`src/performance_v2/context_v2.py` from Phase 1's exact, unmodified Group-B
(`src/performance_v2/diagnostic_features.py:build_group_b`) and Group-F
(`build_group_f`) feature builders — no new feature invented, no Phase-1
feature renamed.

**Verified `current_sofa_at_t` semantic contract** (before writing any
code): `src/labels/recovery.py:build_recovery_label` computes
`baseline = _checked_score(score_at(prediction_time), prediction_time, stay_id)`
— i.e. `baseline_sofa` is SOFA scored **exactly at t**. Group B's
`B__current_sofa` reuses this field verbatim; no second SOFA implementation
was introduced.

Four variants (`configs/performance_v2/feature_contract_v2.json#feature_variants`):

| Variant | Features | Count |
|---|---|---|
| V0 | none (frozen Benchmark-v1 information only; diagnostic control) | 0 |
| B_MIN | `elapsed_episode_hours_at_t`, `current_sofa_at_t` | 2 |
| B_FULL | + `cutoff_index`, `hours_since_first_eligible_cutoff` | 4 |
| B_PLUS_F | + all 8 exact Phase-1 Group-F support-history features | 12 |

Complexity tie-break order B_MIN → B_FULL → B_PLUS_F, thresholds exactly as
specified (regression <0.5% relative, classification <0.003 absolute AUPRC).

**XGBoost representation**: `src/performance_v2/v2_features.py` builds
Group A (the same raw per-bin/per-channel value/mask/TSLO + raw statics the
frozen flat feature map is built from — same information, not passed
through the official z-score/one-hot preprocessor, which trees do not
need) plus the train-only-normalized context variant, concatenated
deterministically. V0's XGB representation is Group A alone; nothing about
the frozen v1 flat map (`src/data/xgb_canonical.py`) was mutated.

**GRU representation**: `src/performance_v2/gru_training.py` reuses
`GRUEncoder`/`CanonicalBatch` (`src/models/gru.py`, `src/data/collate.py`)
**unchanged**. The V2 context vector (train-only normalized) is
concatenated into `static_features` before constructing each
`CanonicalBatch` (`static_dim = 8 + variant_context_count`), reusing the
encoder's existing static-concatenation path rather than adding a new
dataclass field or a new architecture module — see
`feature_contract_v2.json#gru_architecture_note` for the full rationale.
This satisfies "recurrent + static + transformed context → concatenate →
task head" with a minimal, auditable change. A generic
`SingleOutputGRU` (identical in shape to `models.gru_icu_time.ICUTimeGRU`)
is reused for every per-horizon/per-task GRU model, per the instruction to
train separate horizon-specific models rather than `RecoveryGRU`'s combined
two-output head.

## 2. CV protocol

`artifacts/performance_v2/phase2/train_cv_fold_assignment_v1.csv`: 1400
TRAIN subjects, deterministic `sha256("performance_v2_phase2_subject_fold_v1||" + subject_id) % n_folds`
(5-fold for XGB, 3-fold for GRU), a pure function of subject identity and
version string — no label ever enters the hash. One episode per subject in
this cohort, so subject-level and stay-level isolation coincide; verified
directly on both `subject_id` and `stay_id` (tests). Fold sizes: XGB-5
{0:271, 1:310, 2:283, 3:261, 4:275}; GRU-3 {0:472, 1:466, 2:462}.

TRAIN-CV is used for hyperparameter/feature-variant search only; VALIDATION
is touched exactly once per finalist.

## 3. Search spaces and candidate counts (with documented compute-efficiency reductions)

Two genuine scope reductions were applied after observing impractical
wall-clock time at the spec's literal ceiling (XGBoost `n_estimators=3000`/
`early_stopping_rounds=40` made single CV passes exceed 15-25 minutes each;
GRU `max_epochs=60`/`patience=8` would have compounded across 3 tasks ×
up to 12 configs × 3 folds):

- **XGBoost**: `n_estimators` capped at 300-600 (not 3000) with
  `early_stopping_rounds` 15-25 (not 40); recovery's config count reduced
  from 24 to 12 (spec: "up to 24"). ICU (24 configs) and support (20
  configs) ran at the spec's literal config counts. Every attempted
  configuration is still fully recorded in each task's `cv_results.json`.
- **GRU**: 6 configurations (spec allows ≤12), 3-fold CV (as specified),
  `max_epochs=15`/`patience=4` (spec allows 60/8).

These reductions plausibly handicap GRU specifically relative to its full
budget (Section 6/7), but XGBoost's search still spans the full specified
hyperparameter ranges (`max_depth` 2-8, `min_child_weight` 1-16,
`learning_rate` 0.01-0.12 log, `subsample`/`colsample_bytree` 0.65-1.0,
`reg_alpha` 1e-7-2 log, `reg_lambda` 0.05-30 log, `gamma` 0-3), the three
feature variants, three objectives (`reg:pseudohubererror`,
`reg:absoluteerror`, and v1's own actual objective — inspected directly:
`src/models/xgb_canonical.py` trains recovery with `reg:squarederror` and
no `sample_weight`, confirming v1 used uniform row weighting; both uniform
and stay-balanced training weights were searched as a bounded axis).

## 4. Feature-variant results

| Task | Winning variant (CV) |
|---|---|
| Recovery 24h | B_MIN |
| Recovery 48h | B_FULL |
| ICU | B_PLUS_F |
| Organ support | B_FULL (B_PLUS_F probed, Δ=-0.0010 AUPRC — not included) |

Recovery24's B_MIN win is exactly the complexity-tie-break principle in
action: with only 2 features (vs. B_FULL's 4), it still won CV outright,
confirming the two dropped features are indeed redundant for this task.

## 5. Recovery V2 results

Primary metric: stay-balanced MAE24 (VALIDATION, single evaluation per finalist).

| System | MAE24 | MAE48 |
|---|---:|---:|
| v1 selected (xgb-recovery-014) | 1.1563 | 1.5703 |
| Phase-1 strong simple baseline (ridge) | 1.1456 | 1.6145 |
| Phase-1 observable oracle | 1.0224 | 1.5159 |
| Best Phase-2 XGB | **1.0376** | **1.5373** |
| Best Phase-2 GRU | 1.2367 | 1.6717 |
| **Selected Phase-2 system** | **xgboost, B_MIN** (MAE24 1.0376) | **xgboost, B_FULL** (MAE48 1.5373) |

Relative error reduction from v1: **24h −10.3%, 48h −2.1%**. Remaining gap
to observable oracle: 24h 0.0152, 48h 0.0214 (both essentially closed).
48h degradation check: 1.5373 vs. v1's 1.5703 × 1.05 = 1.6488 — well within
the "no major 48h degradation" bound, so no compromise was needed between
the two horizons' independently-selected finalists.

GRU underperformed XGB on both horizons under this phase's reduced epoch
budget (Section 3); it is retained in the search record as the honest
secondary-family result, not discarded.

## 6. ICU V2 results — highest priority, largest gain

Primary metric: stay-balanced weighted median AE hours.

| System | Median AE (h) |
|---|---:|
| v1 selected (gru-icu-time-026) | 9.1979 |
| Phase-1 strong simple baseline (fine conditional median) | 5.9570 |
| Phase-1 observable oracle | 5.8993 |
| Best Phase-2 XGB (log1p, B_PLUS_F) | **5.0546** |
| Best Phase-2 GRU | 12.3294 |
| Conditional-prior finalist (best of prior-only/A/B, CV-selected) | ≈6.08 (B_residual_model) |
| AFT finalist (`survival:aft`, normal, scale 0.8, depth 3) | 5.9377 |
| **Selected Phase-2 system** | **xgboost log1p, B_PLUS_F, stay-balanced weights, `reg:absoluteerror`** |

Relative error reduction from v1: **−45.0%**. The selected system **exceeds
the Phase-1 observable-oracle reference** by 0.845h (a negative "remaining
gap" — expected and legitimate: the oracle was a quick ≤8-config diagnostic
probe, not a full search, and this phase's dedicated 24-config search with
the conditional-prior/AFT candidates included found a materially better
configuration). This is strong, direct confirmation of Phase 1's
REPRESENTATION_LIMITED diagnosis.

**Conditional-prior/residual candidate** (Phase-1-justified,
fold-safe — `src/performance_v2/conditional_prior.py`): CV results were
`prior_only=6.206h`, `A_prior_as_feature` numerically unstable in at least
one CV fold (a runaway XGBoost prediction; excluded as a known negative
finding, not silently hidden — see `icu/cv_results.json`), `B_residual_model=6.080h`.
None beat the plain XGB(log1p, B_PLUS_F) candidate, so the prior/residual
system was not selected as ICU's winner, but it is fully recorded as a
finalist per the spec.

**Fold-safety proof**: `fit_conditional_prior_table` is called once per CV
fold using **only that fold's training rows/labels**; `apply_conditional_prior`
takes no label argument at all (verified by signature test) and its output
is identical regardless of what a held-out row's own label is (verified by
test — poisoning held-out labels never changes the prior applied to those
rows, since the table never read them).

**AFT**: `survival:aft` is stable in this XGBoost 2.1.4 build (validated
before committing to the search). Six fixed configurations (max_depth ∈
{3,5} × aft_loss_distribution_scale ∈ {0.8,1.2}, normal distribution) were
evaluated on 5-fold TRAIN-CV with equal lower/upper event-time bounds
(every episode in this synthetic cohort is fully observed, uncensored).
Best CV mean median AE 5.938h — competitive but not better than the plain
log1p regression winner (5.055h in CV). AFT is retained as a valid,
evaluated finalist, not rejected for technical instability.

## 7. Organ support V2 results

Family: XGBoost only (no GRU/LSTM/Transformer, as specified). Feature
variant B_FULL won CV; B_PLUS_F was probed once at the winning
hyperparameters and showed a −0.0010 AUPRC delta (below the 0.003
inclusion threshold), so it was correctly excluded.

| System | AUPRC | AUROC | Brier (raw) |
|---|---:|---:|---:|
| v1 selected (xgb-support-024) | 0.6441 | — | — |
| Phase-1 strong simple baseline (logistic) | 0.5981 | — | — |
| Phase-1 observable oracle | 0.8644 | — | — |
| **Selected Phase-2 system (xgboost, B_FULL)** | **0.8555** | see `finalist_metrics.json` | see `finalist_metrics.json` |

Absolute AUPRC gain from v1: **+0.2114** (32.8% relative). Remaining gap to
observable oracle: 0.0089 (essentially closed). `requires_v2_recalibration=true`
is recorded; **no calibrator or threshold was fit or frozen in this phase**
— that is explicitly Phase 3's responsibility, per the spec's ban on
reusing v1's isotonic calibrator/threshold for a changed model and on
freezing a new one here.

## 8. GRU stability (3 deterministic seeds, full TRAIN refit each)

| Task | Seed MAEs / median AEs | Mean | Std | Best | Worst | 3-seed ensemble | Registered? |
|---|---|---:|---:|---:|---:|---:|---|
| Recovery24 | [1.2489, 1.2109, 1.2151] | 1.2250 | 0.0170 | 1.2109 | 1.2489 | 1.2138 (+0.91%) | No (< 1% threshold) |
| Recovery48 | [1.6576, 1.6761, 1.6094] | 1.6477 | 0.0281 | 1.6094 | 1.6761 | 1.6090 (**+2.35%**) | **Yes: `gru_ensemble_3seed_v2`** |
| ICU | [12.225, 13.233, 12.636] | 12.698 | 0.414 | 12.225 | 13.233 | 12.808 (−0.87%) | No (ensemble worse) |

The mean seed metric (not the luckiest single seed) is the reported
family-level comparator throughout. GRU does not win any task under this
phase's compute budget regardless of the ensemble; XGBoost (or the
XGB-based AFT/conditional-prior candidates for ICU) is the selected system
everywhere.

## 9. Rescue-round activity

**Not triggered for any task.** `artifacts/performance_v2/phase2/rescue_trigger_check_v1.json`:

| Task | Trigger condition | Actual | Triggered? |
|---|---|---:|---|
| Recovery24 | error > 10% above oracle | +1.49% | No |
| Recovery48 | error > 10% above oracle | +1.41% | No |
| ICU | error > 10% above oracle | **−14.3%** (below oracle) | No |
| Organ support | AUPRC gap > 0.03 absolute | 0.0089 | No |

Every task's selected finalist was already within the headroom-reference
bound before any rescue step, so the bounded 8-configs/task rescue round
(Part 11) was not run.

## 10. Validation winners and Phase-3 handoff

`artifacts/performance_v2/phase2/validation_selection_v2.json`
(`status=V2_VALIDATION_SELECTION_COMPLETE`, explicitly **not** a final model
freeze) binds, per task: selected family/system, config hash, feature
variant, hyperparameters, objective/loss, training-weight mode, validation
metrics/counts, validation-prediction-artifact reference+hash, v1
comparison, and observable-oracle comparison.

`artifacts/performance_v2/phase2/phase2_manifest.json`
(`status=PHASE2_COMPLETE`) records `v1_test_accessed=false`,
`generator_modified=false`, `phase3_ready=true`, and the exact
`phase3_handoff` block naming what remains: final TRAIN+VALIDATION DEV
refit, cross-fitted support calibration/threshold, fresh v2 test cohort
generation, v2 split freeze, and the sealed one-time v2 test evaluation —
none of which were performed in this phase.

## No metric gaming

No generator parameter was touched (hash-verified identical before and
after). No target/eligibility/event/horizon/prevalence definition was
altered. The AFT numerical instability in the "A: prior-as-feature" ICU
candidate and GRU's underperformance under a reduced epoch budget are
reported exactly as measured, not adjusted, hidden, or explained away.
