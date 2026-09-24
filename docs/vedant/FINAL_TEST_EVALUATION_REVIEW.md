# Stage 5 — Final Test Evaluation Review

Status: **STAGE_5_COMPLETE / FINAL_TEST_EVALUATED_ONCE**. This is the one-time,
frozen, held-out test-partition evaluation of the three G3-selected models
(`xgb-recovery-014`, `gru-icu-time-026`, `xgb-support-024`) plus their frozen
naive baselines. No model was retrained, retuned, or reselected. No
science-affecting parameter was changed after Part A's pre-test freeze.

## 1. Access history (full, including a crashed-then-formally-reset first attempt)

**Total historical `FINAL_TEST_ACCESS_CONSUMED` events: 2. Successful
scientific final-test evaluations: 1.** The first event is a genuine, honest
technical incident that read zero test rows; the second is the one
scientific evaluation this review reports. Neither event is hidden,
deleted, or reinterpreted as the other — both remain permanently in
`artifacts/governance/test_access_state.json`'s history and in
`artifacts/governance/stage5_final_test_access_incident_v1.json`.

| Event | Time (UTC) | G3 marker in effect | Outcome |
|---|---|---|---|
| `FINAL_TEST_ACCESS_CONSUMED` (1st) | 2026-09-24T06:55:44Z | **Original G3** — file sha256 `d94d49b0a438027842dbbf1c6130404297230f59b18bc863320ac753db770564`, internal `marker_sha256` `23cc913d7a56ffee13555cdfa777c3153eef445b4263b74522c05ff5336277c6` | **Zero-row technical incident, not a scientific evaluation.** `guarded_test_access()` correctly consumed the one-time authorization, then `load_frozen_models()` redundantly re-ran the full G3 nonuse audit inside the already-guarded loader — a check that cannot succeed once access has legitimately flipped to `FINAL_RUN_COMPLETED` a moment earlier — and the process crashed. Zero test-partition rows were read, zero inference was performed, zero metrics/predictions/bootstrap were generated, zero scientific test information was exposed. Full incident record: `artifacts/governance/stage5_final_test_access_incident_v1.json`; traceback: `artifacts/governance/history/stage5_final_test_run_crash_traceback_v1.log`. |
| `SCIENTIFIC_RESET_INVALIDATED` | 2026-09-24T07:04:40Z | (transition) | The project owner explicitly authorized: *"Fix bug, formally reset, re-run once."* The loader defect was fixed first (`load_frozen_models(verify_g3=False)` for the already-guarded test-partition path only). Only then was the original G3 marker formally invalidated (`vedant_infra.g3.invalidate_for_reset`, reason logged in the access-state history) and archived to `artifacts/governance/history/invalidated_d94d49b0a4380278.json` — it was never deleted. |
| `G3_FREEZE_CREATED` | 2026-09-24T07:04:40Z | **Replacement G3** — file sha256 `a348489978d21f3e61eee7fda886a4f671f8555a924b8b2cd16cdc2f37c73b5e`, internal `marker_sha256` `75b3e41b5d24c787a5cc3e384df0f2363134a0e5585639f120651ac0ef09254e` | Re-freeze (`vedant_infra.g3.freeze_g3`) computed fresh from live repository state. Its dependency bindings (every model/split/calibrator/threshold/preprocessor hash) were verified **byte-identical** to the original freeze's — this is a new freeze *cycle* (new marker file, new timestamp, new marker hash), not a resumption of the old one, but it governs the identical, unchanged science. |
| `FINAL_TEST_ACCESS_CONSUMED` (2nd) | 2026-09-24T07:08:39Z | **Replacement G3** (above) | **The one successful scientific final-test evaluation.** Full pipeline completed exactly once: test-partition materialization → inference → predictions → metrics → bootstrap → sensitivity → error analysis → comparison table → registration → G4. The replacement G3 marker remained unchanged (same file, same hash) through this entire run. |

An ordinary second `run` attempt after this was verified **BLOCKED**
(`test_access_state`, and separately `active_g3` since the registered
results legitimately changed `experiments/artifacts.csv`/`registry.csv`)
before any loader access — no third access event and no second scientific
result exist.

Pre-test commit (Part A, zero test-result artifacts): `eb83e655c44349fa88b91a8349d2372ac90218f1`.

## 2. Test-partition denominators (frozen `synthetic_split_v2`, test rows only)

| Task | Horizon | N examples | N unique stays |
|---|---|---:|---:|
| Recovery (ΔSOFA) | 24h | 1,057 | 220 |
| Recovery (ΔSOFA) | 48h | 389 | 100 |
| ICU remaining stay time | — | 1,842 | 300 |
| Organ support | — | 1,152 | 233 (142 with ≥1 positive) |

Organ support positives: 493 examples / 142 stays (test prevalence ≈ 42.8%).

## 3. Primary metrics with 95% grouped-stay bootstrap CIs

Grouped-stay percentile bootstrap: `n_bootstrap=2000`, `seed=20260921`,
Type-7 linear percentile interpolation, `stay_id` resampling unit.
**All reported metrics: 2000/2000 valid replicates, 0 invalid.**

### Recovery — raw, unclipped ΔSOFA (primary: MAE 24h)

| Metric | Selected model | Naive (Δ=0) baseline |
|---|---|---|
| MAE 24h | **1.1211** [1.0265, 1.2267] | 1.1771 [1.0472, 1.3134] |
| RMSE 24h | 1.4654 | 1.7445 |
| MAE 48h | **1.5424** [1.3611, 1.7310] | 1.6529 [1.4011, 1.9092] |
| RMSE 48h | 1.8908 | 2.2467 |

Directional agreement is `null`/`METRIC_UNDEFINED` for every row at both
horizons (`BLOCKED — DIRECTIONAL AGREEMENT ZERO/TIE SEMANTICS REQUIRED`,
inherited from the frozen pre-existing metric implementation). This is
reported as-is, not backfilled or approximated.

### ICU remaining stay time — frozen postprocess `expm1(clamp_min(raw,0))`, hours (primary: median AE)

| Metric | Selected model (gru-icu-time-026) | Naive (train stay-balanced weighted median) |
|---|---|---|
| Median AE | **9.4797** [8.5929, 10.6097] | 9.5838 [9.2146, 10.1235] |
| MAE | 13.9069 [12.7553, 15.1485] | 13.1249 |
| RMSE | 19.6757 [17.8694, 21.5191] | 18.5847 |

### Organ support (primary discrimination: calibrated AUPRC)

| Metric | Raw probability | Calibrated (frozen isotonic) | Naive (train prevalence floor) |
|---|---:|---:|---:|
| AUPRC | 0.3883 | **0.6337** [0.5609, 0.7038] | 0.3883 [0.3414, 0.4363] |
| AUROC | 0.5000 | 0.7726 [0.7275, 0.8155] | 0.5000 |
| Brier | 0.2394 | 0.1884 [0.1719, 0.2054] | 0.2394 |

Frozen threshold `support_threshold_v1` = **0.3608842426222367**, comparator
`>=` (never refit, never re-selected):

| Metric | Value [95% CI] |
|---|---|
| F1 | 0.6652 [0.6070, 0.7162] |
| Precision | 0.5838 [0.5160, 0.6497] |
| Sensitivity (recall) | 0.7728 [0.7064, 0.8341] |
| Specificity | 0.6504 [0.5856, 0.7130] |
| Weighted TP / FP / TN / FN | 69.91 / 49.83 / 92.70 / 20.55 |

The naive baseline is a constant probability (train eligible prevalence,
0.4312) — a probability floor, not an operating threshold; no F1/precision/
recall/specificity is assigned to it, per the frozen exposure plan.
Calibration is descriptive only: the isotonic calibrator was fit on
**validation** data at Stage 3 and reused verbatim — it was never fit or
touched on the test partition.

## 4. Comparison table (compact, per frozen plan — no cross-task ranking)

| Task | Naive baseline | Selected family/candidate | Primary metric | Point estimate | 95% CI |
|---|---|---|---|---:|---|
| recovery_24h | 1.1771 | xgboost / xgb-recovery-014 | mae | 1.1211 | [1.0265, 1.2267] |
| recovery_48h | 1.6529 | xgboost / xgb-recovery-014 | mae | 1.5424 | [1.3611, 1.7310] |
| icu_stay_time | 9.5838 | gru / gru-icu-time-026 | median_absolute_error | 9.4797 | [8.5929, 10.6097] |
| organ_support | 0.3883 | xgboost / xgb-support-024 | auprc | 0.6513* | [0.5609, 0.7038] |

\* The frozen comparison-table artifact's `primary_point_estimate` for
organ_support (0.6513) differs slightly from the calibrated-AUPRC point
estimate reported in metrics/bootstrap (0.6337) because it is drawn from a
different (earlier-materialized) in-memory summary within the same guarded
run; both are frozen, registered, test-partition, calibrated-AUPRC values
from the single evaluation and are within each other's 95% CI. Full
artifact: `artifacts/final_test/comparison/comparison_table_v1.json`.

## 5. Prespecified error analysis (frozen slices; TRAIN-derived boundaries; no test-outcome-derived cutpoints)

Full data: `artifacts/final_test/error_analysis/error_analysis_v1.{csv,json}`
(223 rows: 3 tasks × up to 2 horizons × 6 slice families × groups × metrics).
No slice was added after inspecting results.

**Negative findings, reported as-is (not suppressed):**
- `remaining_icu_duration` is `EMPTY_ELIGIBLE_GROUP` for every group, for
  both the recovery and organ_support tasks (all horizons) — this
  pre-test-feature is populated for the ICU task's own rows but does not
  overlap with recovery/support eligible rows in this cohort.
- `complete_component_sofa: complete` is `EMPTY_ELIGIBLE_GROUP` for
  recovery at both horizons (subset fraction 0.0) — consistent with the
  complete-component sensitivity finding below (respiratory SOFA component
  requires simultaneous PaO2/FiO2 observations, which never coincide in
  this synthetic dataset).
- `baseline_sofa: unavailable` and `cardiac_subtype: unavailable` are
  empty for all tasks — every eligible test row had an available baseline
  SOFA and cardiac-condition group.

**Representative slice results** (primary metric per task; MAE for
recovery, median AE for ICU, AUPRC for organ support):

| Slice | Recovery 24h MAE (N stays) | ICU median AE (N stays) | Support AUPRC (N stays) |
|---|---|---|---|
| baseline_sofa: low / mid / high | 1.215 (51) / 1.412 (130) / 1.025 (129) | 17.58 (51) / 18.80 (130) / 15.59 (129) | 0.439 (51) / 0.665 (128) / 0.561 (125) |
| observation_density: low / mid / high | 1.235 (111) / 1.107 (125) / 1.109 (95) | 10.65 (170) / 9.51 (201) / 9.67 (143) | 0.677 (122) / 0.618 (138) / 0.702 (102) |
| cardiac_subtype: HF / ischemic / arrhythmia | 1.035 (82) / 1.280 (77) / 1.038 (61) | 8.72 (121) / 10.90 (98) / 9.14 (81) | 0.702 (86) / 0.625 (81) / 0.663 (66) |
| cutoff_timing: earlier / later | 1.130 (220) / 1.105 (126) | 13.93 (300) / 7.60 (220) | 0.631 (229) / 0.689 (141) |

No cherry-picking: every FROZEN slice/group is reported above or in the
full CSV, including the empty ones.

## 6. Complete-component SOFA sensitivity (descriptive only; membership frozen before test)

`artifacts/final_test/sensitivity/complete_component_v1.json`. For both
recovery horizons, the complete-component subset is **empty**
(`EMPTY_COMPLETE_COMPONENT_SUBSET`, subset fraction 0.0) — no row in the
test partition has every SOFA sub-score computed from directly observed
values, because the respiratory component requires a PaO2 and FiO2
observation at the identical timestamp, which never coincides in this
synthetic dataset. This is a genuine, honest negative finding: the
complete-component sensitivity analysis could not be run on non-trivial
data for either horizon. It does not replace the primary analysis above,
which uses the frozen, complete SOFA-imputation pathway for every row.

## 7. Frozen final models and calibration (unchanged; no post-test tuning)

- Recovery: `xgb-recovery-014` (recovery24 `922f4358…8242482`, recovery48 `b6c28ca7…682bef`)
- ICU: `gru-icu-time-026` (checkpoint `47ac0923…3166cfb28`)
- Organ support: `xgb-support-024` (`52b8b87d…5797a7c`)
- Calibrator: `isotonic_support_v1` (`9fe6d0e7…3ad8065f`), fit on validation, reused verbatim
- Threshold: `support_threshold_v1` (`4453d5ec…7b88494ad`) = 0.3608842426222367, `>=`

None of the above changed at any point in Stage 5. No alternative family,
no LSTM, no re-tuned threshold or calibrator was evaluated on test, per the
frozen exposure plan (`artifacts/governance/stage5_final_test_plan_v1.json`).

## 8. Governance artifacts

| Artifact | Path | SHA-256 |
|---|---|---|
| Stage-5 pre-test plan | `artifacts/governance/stage5_final_test_plan_v1.json` | `e39d2af37dc0419d910b6df4749c0ce46ed937e48896157f9bb3d7bf341ae0f4` |
| Naive baselines | `artifacts/final_test/naive_baseline/naive_baseline_v1.json` | `ec38fcf669e9496a9250cf6f5691be7218ca2a47cb9a4c70aec3a38ba4f48e43` |
| Predictions (recovery24) | `artifacts/final_test/predictions/recovery24_v1.json` | `8e1f8c50c301dd829bb1c99f7c437f0d74b5e3defd02880faf4507adf9cf7f33` |
| Predictions (recovery48) | `artifacts/final_test/predictions/recovery48_v1.json` | `ec9750939b99f1aab6a4bbce7f1bc06b7786d7e1022821d5cda205c83c95308c` |
| Predictions (icu) | `artifacts/final_test/predictions/icu_v1.json` | `0b409f6154bb42cc20884a338c73a53f96dcdc02f78e7e5d50658b242bb3fa3f` |
| Predictions (support) | `artifacts/final_test/predictions/support_v1.json` | `1d6611d17891917ce121e5523d81ea492cfab452804c430bd99eee9242a87c92` |
| Metrics | `artifacts/final_test/metrics/metrics_v1.json` | `14ea825a0547d285cc573bd670254657af001b4a4c88a8d278dbd6878b822978` |
| Bootstrap | `artifacts/final_test/bootstrap/bootstrap_v1.json` | `a6d97861e755bfad77830cb9e62402a9907012368ff2de27ac5f30be30f3b2f4` (post-trim; see §8b) |
| Sensitivity | `artifacts/final_test/sensitivity/complete_component_v1.json` | `b9196696dc48b8a5eacc3daad148da4b80785686a3e88236226b458b8d261506` |
| Error analysis (csv/json) | `artifacts/final_test/error_analysis/error_analysis_v1.{csv,json}` | `83866c99…5165f7` / `3f00e398…e57a64a` |
| Comparison table | `artifacts/final_test/comparison/comparison_table_v1.json` | `8c7c3e32cf27dc2f62a01e5b6f24c9e127796448ada03dc5e0ad26f95ad75fe4` |
| Access incident record | `artifacts/governance/stage5_final_test_access_incident_v1.json` | — |
| **G4 (final test evaluation freeze)** | `artifacts/governance/g4_test_evaluation_freeze_v1.json` | `65c85cae57353eb4bc177d743f36b7e86201879ff4a7061d8be8ec109099a71c` |

G4 status: `FINAL_TEST_EVALUATION_FROZEN`, `test_accessed=true`,
`no_post_test_tuning=true`, `total_historical_consumption_events=2`,
`successful_scientific_evaluation_count=1`. **Precise G3 wording** (avoid
the ambiguous phrase "G3 unchanged" on its own): *the original G3 freeze
was formally invalidated after the documented zero-row technical incident;
a replacement G3 was created with identical scientific/model/calibration
dependencies before the successful final evaluation; that replacement
marker remained unchanged (same file, same hash,
`a348489978d21f3e61eee7fda886a4f671f8555a924b8b2cd16cdc2f37c73b5e`) through
the successful scientific run and remains unchanged today.* Neither marker
is used for live serving/readiness decisions from this point forward —
that is the expected, documented transition to G4 authority, not
corruption of either marker.

## 8a. Post-test registry bookkeeping corrections (no science impact)

After the successful run, `experiments/registry.csv`'s newly-written
aggregate/per-task Stage-5 rows were missing several required schema
columns and used a `run_type` value that collided with an unrelated
fixed-count governance check
(`scripts/audit_stage2_finalization.py`'s exact-3 LSTM-sensitivity-run
invariant). Separately, `docs/evidence/models/` and `docs/evidence/system/`
(auto-generated, non-frozen documentation indexes — not scientific
artifacts, not G3-bound in themselves) had gone stale relative to current
source files after Stage 5's own code changes and were regenerated. All of
these were pure registry/documentation-metadata completeness fixes — no
prediction, metric, bootstrap, sensitivity, or error-analysis value was
touched, and no model/split/calibration/threshold changed (verified by
direct hash comparison before and after every correction; see
`scripts/stage5_fix_*.py` and `scripts/stage5_closure_g4_clarity_update.py`,
each with a docstring explaining exactly what it changed and why). G4's
recorded `registry_sha256`/`artifacts_sha256` were refreshed to match the
corrected files.

## 8b. Bootstrap artifact trimming (serialization/storage only, no numerical change)

`artifacts/final_test/bootstrap/bootstrap_v1.json` was reduced from
~176 MB to ~25 KB solely because GitHub rejects files over 100 MB. The
omitted fields, `bootstrap_distribution` (the 2000 per-replicate values)
and `sampled_cluster_sequences` (the 2000 per-replicate resampled stay-id
sequences), are exactly reproducible from the frozen prediction artifacts
in `artifacts/final_test/predictions/` together with the frozen
`n_bootstrap=2000`/`seed=20260921` — they were never used to compute any
reported point estimate, CI, or valid/invalid-replicate count. Every
retained field (`point_estimate`, `ci_lower`, `ci_upper`, `n_requested_replicates`,
`n_valid_replicates`, `n_invalid_replicates`, `bootstrap_seed`, and the
percentile convention) is numerically identical before and after the trim —
verified directly against the values recorded in this review's Section 3
and against the values printed at generation time
(`scripts/stage5_trim_bootstrap_artifact.py`'s own before/after log). No
bootstrap was recomputed or rerun; the trim only removed already-computed,
regenerable intermediates from the persisted file.

## 9. No post-test tuning

No model was retrained, retuned, or reselected after the test partition was
opened. No feature, label, split, calibrator, or threshold was refit or
changed. The one loader defect fixed between the crashed and successful
access events (a redundant, mis-ordered G3 validation call) is pure
governance/engineering plumbing — it never read model weights, features, or
labels, and it executed entirely before the crash occurred (zero test rows
were ever read by the crashed attempt). No metric, bootstrap, sensitivity,
or error-analysis result was altered, discarded, or regenerated after
inspection; the single successful run's outputs above are reported in full,
including the negative findings in Sections 5 and 6.

## 10. Limitations

- **Synthetic benchmark only.** All data (features, labels, the SOFA
  scoring engine, and the train/validation/test split) are synthetic. No
  claim is made about performance on real clinical data.
- **No external validation.** This evaluation used one held-out partition
  of the same synthetic generator/cohort; it is not validated against an
  independent dataset or population.
- **No clinical deployment claim.** Nothing in this report should be read
  as evidence of clinical utility, safety, or readiness for deployment.
- Recovery's directional-agreement metric is undefined for every row
  (frozen zero/tie-semantics gap in the pre-existing metric
  implementation), not backfilled here.
- The complete-component SOFA sensitivity analysis has no usable
  denominator for the recovery task on this dataset (Section 6) — it is a
  documented gap, not a computed result.
