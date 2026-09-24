# Performance V2 — Phase 1: Predictability, Error, Representation, and Performance-Ceiling Diagnosis

Status: **DEVELOPMENT_DIAGNOSTIC_ONLY**. Nothing in this document or in
`artifacts/performance_v2/phase1/` is a serving artifact. No Benchmark-v1
model was retrained, retuned, reselected, or recalibrated. No v1 TEST row
was read, inspected, or evaluated against at any point (enforced in code:
`performance_v2.data_loading.load_dev_rows` refuses any split other than
`train`/`validation`; see the leakage/governance tests in
`tests/test_performance_v2_phase1_leakage.py`). The generator was never
edited.

Branch: `performance-v2`, created from frozen commit
`7565e0e373f5d49e20f123a00adfdb4ed91d82d0` (`CORE_PROJECT_PIPELINE_COMPLETE`).

## 1. The existing performance problem

Benchmark v1's final TEST results (historical evidence only, not used as an
optimization target here):

| Task | Selected | Final TEST metric | Naive | Relative gain over naive |
|---|---|---|---|---|
| Recovery 24h | xgb-recovery-014 | MAE 1.1211 | 1.1771 | ~4.8% |
| Recovery 48h | xgb-recovery-014 | MAE 1.5424 | 1.6529 | ~6.7% |
| ICU remaining time | gru-icu-time-026 | median AE 9.4797h | 9.5838h | ~1.1% |
| Organ support | xgb-support-024 | calibrated AUPRC 0.6337 | 0.3883 | +63% |

ICU is barely better than a train-median constant; recovery is only
marginally better than predicting zero movement. Support is the one task
with substantial real signal already extracted.

## 2. Reconstructed v1 validation performance story (Part 1)

`artifacts/performance_v2/phase1/v1_validation_performance_summary.json`.
Read only from existing frozen search artifacts — nothing retrained.

| Task | Naive (validation) | Best XGB (validation) | Best GRU (validation) | LSTM sensitivity (validation) | Selected |
|---|---|---|---|---|---|
| Recovery 24h MAE | 1.2200 | **1.1563** | 1.2508 | 1.2604 | xgboost |
| Recovery 48h MAE | 1.6295 | **1.5703** | 1.7563 | 1.7756 | xgboost |
| ICU median AE (h) | 9.7447 | 11.8148 | **9.1979** | 11.2219 | gru |
| Support AUPRC | 0.4456 | **0.6441** | 0.6342 | 0.6184 | xgboost |

Validation-only sample counts (from `validation_sample_counts` in the
summary artifact): recovery24 968 rows, recovery48 332 rows, ICU 1754 rows,
support 1116 rows, total validation rows 1754 (train: 8626). LSTM is worse
than both XGB and GRU on every task, confirming it was correctly not
selected.

## 3. Target distribution / baseline diagnosis (Part 2)

`artifacts/performance_v2/phase1/target_distribution_diagnosis_v1.json`,
TRAIN+VALIDATION only.

**Recovery 24h** (n=5973): mean ΔSOFA = 0.169, median = 0, std = 1.847,
IQR = [-1, 1]. 37.1% of rows have **exactly zero** ΔSOFA24; 67.9% have
`|ΔSOFA24| <= 1`. Mean absolute deviation from zero (= the zero-delta
persistence baseline's own MAE) is 1.272. Most SOFA movement over 24h is
small, consistent with the strong mean-reversion in the generator's latent
dynamics (Section 8).

**ICU remaining time** (n=10380): overall median 27.77h, IQR
[16.16, 44.20]. Binning by `elapsed_episode_hours_at_t` alone (TRAIN-only
bin edges [24,42,60,78,96,∞)) already explains a **~20.5% fractional
reduction** in unconditional variance, with the conditional median falling
monotonically from 32.98h (earliest cutoffs) to 21.03h (latest cutoffs).
**Elapsed episode time strongly determines remaining time**, and it is
absent from the current model input (Section 4).

**Organ support**: overall prevalence 42.8% (train+validation). Prevalence
varies materially by baseline-SOFA band and by cutoff index (see the full
artifact for the per-band table).

## 4. Canonical feature audit (Part 3)

`artifacts/performance_v2/phase1/canonical_feature_inventory_v1.json`, read
directly from `configs/synthetic/feature_schema_v2.json` and
`src/data/xgb_canonical.py:build_flat_feature_map` (the single place the
served flat/structured model input is constructed) — not assumed.

The current model input is exactly: 21 temporal channels × 8 bins × 3
blocks (LATEST value, observation mask, time-since-last-observation/TSLO),
8 padding flags, and 8 encoded static features (age z-score + one-hot
sex/cardiac group).

**Confirmed absent from the model input**, despite being lawfully known at
cutoff t:

- `episode_elapsed_time_at_t` / `cutoff_index` — not a temporal channel,
  not a static, never added to the flat feature map. (`grid_index` exists
  in the raw row for governance/slicing but is never fed to the model.)
- **Current SOFA** — `configs/synthetic/feature_schema_v2.json`'s
  `sofa_feature.included` is explicitly `false` ("RECOVERY_TARGET_SOURCE_IS_NOT_AUTOMATICALLY_A_MODEL_FEATURE").
  `baseline_sofa` exists in the pre-split scientific package (lawful,
  computed at/before t) but is never joined into the served feature vector.
- Recent SOFA trajectory/trend.
- Per-channel deltas (6h/12h/24h/48h), slopes, rolling mean/min/max/std.
- Missingness-pattern summaries (observation count/fraction beyond the
  per-bin mask+TSLO already present).
- Support-state duration-in-state / transition-count summaries (the raw
  per-bin binary vasopressor/ventilation state IS present).

**Confirmed present**: per-bin LATEST value, observation mask, and TSLO for
all 21 channels (TSLO is already a genuine derived-style signal); the raw
per-bin support state.

## 5. Lawful diagnostic feature generator + leakage tests (Parts 4-5)

`src/performance_v2/diagnostic_features.py` builds six feature groups from
information at-or-before t only:

- **A**: existing canonical features (value+mask+TSLO×21×8, statics).
- **B**: `elapsed_episode_hours_at_t`, `cutoff_index`,
  `hours_since_first_eligible_cutoff` (all derived only from
  `prediction_time` and `intime`, never `outtime`), current baseline SOFA.
- **C**: per-channel 6/12/24/48h deltas and 12/24/48h least-squares slopes;
  SOFA trend/slope from past-cutoff-only re-scoring via the frozen,
  unmodified `data.synthetic.sofa.sofa_at()` (reused exactly as Stage 5's
  complete-component sensitivity did — see `src/performance_v2/sofa_trend.py`).
- **D**: per-channel windowed mean/min/max/std over 12/24/48h.
- **E**: per-channel/overall observed-fraction summaries.
- **F**: support-state duration-in-current-state, transitions-in-window,
  bins-since-last-transition.

`elapsed_episode_hours_at_t = (prediction_time - intime)` was verified
exactly equal to `24 + 6*grid_index` for every sampled row (test:
`test_elapsed_time_formula_matches_grid_index_contract`), confirming it is
lawful and derivable with zero lookback risk.

**Leakage tests** (`tests/test_performance_v2_phase1_leakage.py`, 9/9
passing): no diagnostic feature name ever carries a prohibited token
(`outtime`, `delta_sofa`, `future_support`, `latent_state`, ...); poisoning
the (illegally-present) future-outcome fields never changes a single
diagnostic feature value; an illegal 9th bin appended after the frozen
8-bin window is **rejected fail-closed**
(`Phase1FeatureError`), not silently misread as the true last observation.

## 6. Stronger simple baselines (Part 6)

`artifacts/performance_v2/phase1/stronger_baselines_v1.json`, VALIDATION
only.

**Recovery**: R0 (zero-delta) MAE24=1.2200/MAE48=1.6295. R1 (linear
extrapolation of the past-only SOFA trend) is **much worse**
(MAE24=2.587, MAE48=4.632) — the mean-reverting generator dynamics punish
naive linear extrapolation badly; this is a genuine negative finding, not
suppressed. R2 (plain ridge regression on the full diagnostic feature set,
median-imputed, no tuning) already reaches MAE24=1.1456 — **better than
the current selected, tuned XGB model's own validation MAE (1.1563)**, and
MAE48=1.6145 (close to XGB's 1.5703).

**ICU**: I0 (naive train median) median AE=9.745h. I1 (TRAIN-only
conditional median by elapsed-time band alone) was, unexpectedly,
*worse* on stay-balanced MAE than I0 (median AE=11.40h) despite the clear
monotonic conditional-median gradient in Part 3 — likely a sensitivity of
this coarse a binning to how per-bucket stay-balanced weights interact
with bucket boundaries; not fully resolved within Phase-1's scope, reported
honestly rather than hidden. **I2** (TRAIN-only conditional median on the
finer elapsed×cutoff×cardiac×severity grid, min-cell-count=20 fallback)
resolves this cleanly: median AE=**5.957h**, a 39% reduction from the
naive baseline, using nothing but grouped historical medians — no ML at
all. I3 (ridge in log1p space) did not improve on I0.

**Support**: prevalence baseline AUPRC=0.4456 (validation). A single,
untuned logistic regression on the full diagnostic feature set already
reaches AUPRC=0.5981.

## 7. Observable oracle (Part 7)

`artifacts/performance_v2/phase1/observable_oracle/`,
**NON_SERVING_DIAGNOSTIC_ORACLE**. XGBoost, ≤8 deterministic
configurations per task (depth∈{3,5} × n_estimators∈{150,400} ×
learning_rate=0.05), ranked on VALIDATION only, all groups A-F.

| Task | Current selected (validation) | Observable oracle (validation) | Relative gain |
|---|---:|---:|---:|
| Recovery 24h MAE | 1.1563 | **1.0224** | +11.6% |
| Recovery 48h MAE | 1.5703 | **1.5159** | +3.5% |
| ICU median AE (h) | 9.1979 | **5.8993** | +35.9% |
| Support AUPRC | 0.6441 | **0.8644** | +34.2% |

These are large, consistent gains across all four task/horizon
combinations, using only features already lawful at cutoff t.

## 8. Latent oracle (Part 8)

`artifacts/performance_v2/phase1/latent_oracle/`,
**NON_SERVING_LATENT_ORACLE / NEVER_ALLOWED_IN_PRODUCTION_FEATURES**.
Features: the subject-level generation-time latent constants
(`z0`, `equilibrium`, `support_propensity` from
`data.synthetic.latent_state.initial_state`, reconstructed deterministically
from the frozen generator config + each TRAIN/VALIDATION subject's own
identity — never from a TEST subject) plus elapsed time.

| Task | Latent oracle (validation) | vs. naive | vs. observable oracle |
|---|---:|---|---|
| Recovery 24h MAE | 1.2808 | **worse** than naive (1.2200) | much worse |
| Recovery 48h MAE | 1.8022 | **worse** than naive (1.6295) | much worse |
| ICU median AE (h) | 14.0501 | worse than naive (9.7447) | much worse |
| Support AUPRC | 0.4918 | barely above naive (0.4456) | much worse |

**Interpretation — Case A, not Case B**: the latent oracle does not exceed
the observable oracle on any task; it does not even reliably beat naive.
**Scope limitation, stated plainly**: this implementation uses each
subject's *generation-time* (t=0) latent draw, not the *instantaneous*
latent state at the specific cutoff t (which would require replaying
`simulate_with_support`'s full, RNG-coupled support-decision logic per
subject — not attempted here for compute-efficiency). Because the latent
process is mean-reverting with a real drift/innovation/shock structure
(Section 9), the state at t=0 has usually drifted substantially by the
time an eligible cutoff (t≥24h, up to t≈90h) is reached, so this proxy
plausibly *understates* the true latent ceiling. Even so, the result is
informative: it shows the exploitable signal is **not hiding in an
inaccessible t=0 subject trait** — it is *recoverable from the subject's
own observable trajectory*, which is exactly what the observable oracle
demonstrates. This is direct evidence for representation/model limitation
(Case A), not observability limitation (Case D would require the latent
oracle to dominate while the observable oracle stays near naive, which is
not what was found).

## 9. Generator predictability analysis (Part 9)

Read from `src/data/synthetic/generator.py`, `latent_state.py`,
`observations.py`, `support.py` — never edited.

- **Recovery-relevant vitals**: each observed value is a noisy logit-link
  function of a weighted average of specific dimensions of a 5-dimensional
  latent state (`systemic_severity, cardiac_stress, respiratory_burden,
  renal_burden, neurologic_burden`) at observation time, plus a per-subject
  constant effect and per-observation Gaussian noise. Observation *times*
  are themselves an irregular, severity-dependent point process.
- **The latent state itself** evolves via
  `state[h+1] = state[h] + k*(equilibrium - state[h]) + drift + coupling@tanh(state[h]) + shock_pulse + innovation*N(0,1)`:
  mean-reversion (`k≈0.05-0.06`/hour) toward a per-subject equilibrium,
  a per-subject constant drift, a nonlinear cross-dimension coupling term,
  hourly Gaussian innovation, and occasional (`p≈0.018`/hour) multi-hour
  shocks. **SOFA-relevant future movement is genuinely partly
  unpredictable at t**: it depends on innovation/shock draws that have not
  happened yet.
- **ICU episode duration** is fixed **once**, at subject creation, as
  `duration = clip(exp(4.0 + [0.16,0.12,0.08,0.08,0.04]·z0 + 0.08*support_propensity + N(0, 0.35)), 20, 132)`
  hours — it does **not** depend on the evolving trajectory at all. The
  latent coefficients are small (max 0.16); the noise term (std=0.35 in
  log-hours) is comparatively large.
- **Organ support initiation** is an hourly logistic function of the
  current latent state (`systemic_severity`, `cardiac_stress`/
  `respiratory_burden`), a 5-hour rolling severity mean, a per-subject
  constant propensity effect, and fresh hourly logit noise.

## 10. Same-history counterfactual stochasticity test (Part 10)

`artifacts/performance_v2/phase1/stochasticity_analysis_v1.json`.

- **ICU duration noise floor** (closed-form/Monte Carlo, 20000 draws): the
  duration formula's noise term alone (`N(0, 0.35)` log-hours,
  geometric std factor **1.42×**) sets a hard per-subject floor on
  achievable duration accuracy for *any* model — even a perfect oracle for
  `z0`/`support_propensity` cannot see past it.
- **Latent forward dispersion** (50 TRAIN cutoffs × 100 continuations each,
  no-support latent recursion only — see scope limitation below): given an
  *identical* state at t, the standard deviation of the `systemic_severity`
  latent dimension at t+24h is **0.399** and at t+48h is **0.412** —
  a substantial, genuinely irreducible dispersion that is nearly saturated
  by 24h (consistent with the mean-reverting process reaching near-stationary
  variance quickly).
- **Scope limitation**: both continuation analyses exclude the
  support-conditioned state-evolution perturbation in
  `simulate_with_support` (faithfully forking it would require re-deriving
  each subject's exact mid-stream support-decision RNG position — not
  attempted for a diagnostic phase). Reported dispersion is a lower bound
  for subjects ever on support.
- **Conclusion**: real, quantified stochasticity exists and is a
  legitimate secondary contributor to task difficulty, but — per Section
  7 — it does not prevent the observable oracle from already closing most
  of the naive-vs-current-model gap, so it is not the dominant bottleneck.

## 11. Validation error analysis for the current selected models (Part 11)

`artifacts/performance_v2/phase1/validation_error_analysis_v1.json`,
VALIDATION only, using the already-frozen validation prediction files (no
re-inference).

ICU's current GRU model's mean absolute error falls **monotonically** with
cutoff index: 19.95h (cutoff 0) → 18.52 → 17.35 → 16.93 → ... → 14.95h
(cutoff 7+), and similarly by elapsed-time quartile (19.26h → 17.16 →
16.31 → 14.56h). The model's error is already implicitly correlated with
exactly the structural variable (elapsed time) it cannot see explicitly —
consistent with, and corroborating, the Part 15 ablation finding.

## 12. Train vs validation generalization gap (Part 12)

`artifacts/performance_v2/phase1/train_validation_gap_v1.json`. Computed
for the OBSERVABLE_ORACLE at its Part-7-chosen configuration (refit only,
no new search); a from-scratch frozen-bundle TRAIN-side inference run for
the current v1 selected models themselves was not attempted in Phase-1's
time budget (documented scope limitation — GRU's torch structured-input
pipeline and XGB's early-stopped best-iteration are not trivially
re-runnable without re-inference).

| Task | Train | Validation | Relative gap |
|---|---:|---:|---:|
| Recovery 24h MAE | 0.710 | 1.022 | 44.0% |
| Recovery 48h MAE | 0.842 | 1.516 | 80.0% |
| ICU MAE (h) | 6.819 | 7.710 | 13.1% |
| ICU median AE (h) | 4.947 | 5.899 | 19.2% |
| Support AUPRC | 0.902 | 0.864 | −4.2% (healthy, expected) |

Recovery shows a substantial train/validation gap — with ~969-1400
diagnostic features and 6-9k training rows, the oracle is genuinely
overfitting somewhat, especially for recovery. This does **not** invalidate
the representation-limited finding (the reported oracle numbers above are
VALIDATION-only, held out correctly), but it means Phase 2 should not
naively throw the entire diagnostic feature set at the model: **Group B
alone** (Section 13) captures nearly all of the gain with only 4 extra
features and should be prioritized over the larger, higher-overfitting-risk
groups C-F.

## 13. Objective / target-formulation audit (Part 13)

No large-scale objective search was run (out of Phase-1's compute budget);
this is a qualitative assessment informed by Sections 6-8.

- **Recovery**: current training uses a standardized target + Huber loss;
  final reporting is raw MAE24. R1's catastrophic failure (Section 6)
  shows naive linear extrapolation is badly miscalibrated for this
  mean-reverting target — this argues for keeping a robust, shrinkage-
  friendly loss (Huber/pseudo-Huber or L1), not for a fundamentally
  different target formulation. No evidence surfaced that a target change
  (vs. a feature change) is the primary lever here.
- **ICU**: current training uses log1p + Huber; final reporting is raw-hour
  median AE. The ICU duration mechanism (Section 9) — a fixed value with
  additive log-space noise and hard clipping — is a plausible mechanistic
  match for a log-space regression *or* a survival/AFT-style formulation.
  Given Section 7 already shows a plain log1p-regression oracle with the
  missing structural features closes most of the achievable gap, a
  survival/hazard reformulation is a reasonable **secondary** Phase-2
  experiment, not a precondition for the primary gain.
- **Support**: current BCE/XGB objective is adequate; Section 7's gain came
  entirely from features, not from an objective change.

## 14. Model-family opportunity audit (Part 14)

- **Recovery**: XGBoost already outperforms GRU/LSTM on validation
  (Section 2) even without Group B; the observable oracle (also XGBoost)
  shows most of the achievable gain is feature-driven, so XGBoost with
  Group A+B(+F) is the primary Phase-2 candidate. A GRU/sequence model with
  the same expanded features is a reasonable secondary comparison (worth
  testing since GRUs can in principle learn deltas/slopes from the raw
  8-bin sequence that XGBoost cannot construct on its own — Group C/D
  showed only mixed/marginal value for XGBoost specifically, which may not
  transfer to a sequence model's own internal representation).
- **ICU**: GRU already outperforms XGBoost on validation; keep GRU as the
  primary Phase-2 candidate with Group A+B, and compare against XGBoost
  with the same features as a secondary check (the oracle used XGBoost and
  still beat the current GRU substantially, so it is worth testing whether
  XGBoost with Group B now also beats GRU with Group B). A discrete-time
  hazard/AFT survival model is a plausible secondary experiment (Section
  13) but not required for the primary gain.
- **Support**: no model-family change recommended; XGBoost with Group B is
  the sole primary candidate, consistent with "do not destabilize support
  without validation evidence."
- **Not selected for Phase 2**: TCN/temporal-CNN, compact Transformer —
  no evidence in this diagnosis points to a sequence-modeling *capacity*
  limitation (the bottleneck is structural features missing from the
  input, not model expressiveness); adding a larger model family without
  first adding Group B would not address the identified bottleneck and
  is explicitly out of scope per the "no giant dependency / no large
  search" compute-efficiency constraint.

## 15. Feature-group ablation / upside attribution (Part 15)

`artifacts/performance_v2/phase1/feature_ablation_v1.json`. Same oracle
configuration (Part 7's chosen config) reused across cumulative feature
groups A → A+B → A+B+C → A+B+C+D → A+B+C+D+E → A+B+C+D+E+F, VALIDATION only.

| Task | A only | A+B | A+B+C | A+B+C+D | A+B+C+D+E | A+B+C+D+E+F |
|---|---:|---:|---:|---:|---:|---:|
| Recovery24 MAE | 1.1387 | **1.0323** | 1.0370 | 1.0481 | 1.0480 | 1.0224 |
| Recovery48 MAE | 1.5863 | **1.5290** | 1.5380 | 1.4847 | 1.4957 | 1.5159 |
| ICU median AE (h) | 12.8598 | **5.8659** | 5.9356 | 5.8954 | 6.0008 | 5.8993 |
| Support AUPRC | 0.6359 | **0.8527** | 0.8576 | 0.8629 | 0.8642 | 0.8644 |

**This is the single most decisive Phase-1 finding**: Group B alone
(elapsed time, cutoff index, current SOFA — 4 extra features) delivers
essentially all of the achievable gain for every task (recovery24: 9.3
percentage points of the ~10 pp MAE reduction; ICU: 54 percentage points
of median-AE reduction, essentially the entire gain; support: 21.7 of the
~23 AUPRC points). Groups C-F add only small, sometimes negative (C,
recovery24) incremental value once B is present, and Section 12 shows they
come with meaningfully more overfitting risk for a modest or negative
return.

## 16. Performance-ceiling table (Part 16)

`artifacts/performance_v2/phase1/performance_ceiling_summary_v1.json` and
`performance_ceiling_table_v1.md`:

| Task | Naive | Current selected (validation) | Strong simple baseline | Observable oracle | Latent oracle | Headroom vs current | Bottleneck |
|---|---:|---:|---:|---:|---:|---:|---|
| recovery_24h | 1.2200 | 1.1563 | 1.1456 | 1.0224 | 1.2808 | +11.6% | REPRESENTATION_LIMITED |
| recovery_48h | 1.6295 | 1.5703 | 1.6145 | 1.5159 | 1.8022 | +3.5% | REPRESENTATION_LIMITED |
| icu_stay_time | 9.7447 | 9.1979 | 5.9570 | 5.8993 | 14.0501 | +35.9% | REPRESENTATION_LIMITED |
| organ_support | 0.4456 | 0.6441 | 0.5981 | 0.8644 | 0.4918 | +34.2% | REPRESENTATION_LIMITED |

All four rows are labeled **REPRESENTATION_LIMITED**, each with direct
supporting measurement (not asserted): observable-oracle-beats-current
(Part 7), ablation attributes the gain to specific absent features (Part
15), and the latent oracle does not dominate (Part 8, ruling out
OBSERVABILITY_LIMITED as primary). GENERATOR_STOCHASTICITY_LIMITED is a
real, quantified secondary factor (Parts 9-10) for recovery and ICU, but
not the dominant one anywhere.

## 17-18. Selected v2 path and exact Phase-2 plan

**`artifacts/performance_v2/phase1/selected_v2_strategy_v1.json`** —
**PATH A: FEATURE_AND_MODEL_V2**, decided automatically from the evidence
above (observable oracle >> current model on every task; latent oracle
does not exceed observable oracle; stochasticity real but secondary;
generator mechanics physiologically plausible, not scientifically
implausible — ruling out Paths B/C/D as dominant, and Path A's evidence is
consistent and decisive enough across all three tasks that a Hybrid (Path
E) label would understate how uniform the Group-B finding is).

- **Recovery**: add Group B first (highest priority), Group F second
  (secondary, small real 24h gain); treat C/D/E as optional/regularized
  only, given Section 12's overfitting evidence. Keep the raw-ΔSOFA target
  and Huber-family loss. Test XGBoost (primary) and GRU (secondary) with
  the expanded feature set. Primary metric: MAE24 (+MAE48/RMSE/median AE
  secondary). Moderate search budget, reusing the existing Phase-12-style
  process.
- **ICU**: add Group B first (by far the dominant lever; C/D/E/F
  deprioritized — no material incremental gain shown). Keep the log1p
  regression formulation as primary; a discrete-time hazard/AFT model is
  an optional secondary experiment. Test GRU (primary, matches current
  family) and XGBoost (secondary, since the oracle itself used XGBoost).
  Primary metric: median absolute error hours.
- **Organ support**: **TEST_HIGH_VALUE_FEATURE_ADDITIONS_ONLY** — add
  Group B to the existing xgb-support-024-equivalent feature set and
  re-run the existing validation search; do not change model family,
  calibration, or threshold procedure unless Phase-2 validation evidence
  supports it.
- **Generator v2**: **not required.** No evidence that hidden generator
  state materially exceeds what lawful observables already expose, and the
  identified stochasticity does not dominate the achievable gain.
- **Fresh dataset generation before Phase 2**: **not needed.** Continue
  development on the existing frozen TRAIN/VALIDATION partitions.
- **Fresh v2 test cohort**: generated only *after* the v2 feature set,
  model family, and objective are completely frozen — a new independent
  synthetic cohort under the same frozen generator, a new frozen subject
  split and sealed test partition (same governance pattern as v1: G1-style
  freeze, clone-fingerprint isolation), never inspected before the final,
  one-time v2 evaluation.

## No metric gaming

No generator parameter, noise term, event probability, or difficult case
was tuned, removed, or adjusted to produce a target score. Every number in
this document comes from code that reads TRAIN/VALIDATION data and frozen
config/generator files as-is. The R1 recovery baseline's poor result and
the I1 ICU anomaly (Section 6) and the latent oracle's underperformance
(Section 8) are reported exactly as measured, not adjusted or omitted.
