# Performance V2 — Phase 4: Final Fresh-Test Evaluation

**Status: `V2_FINAL_EVALUATION_FROZEN`.** This is the one-time, scientific
fresh-test evaluation of the four Performance-V2 models frozen at the end of
Phase 3. Every number below comes from a single governed access event and a
single coordinated inference pass; no model was retrained, retuned, or
re-selected after seeing any fresh-test result.

## 1. Frozen model identities

| Task | Family | Feature variant | n_estimators | Model SHA-256 (first 16) |
|---|---|---|---:|---|
| recovery24 | XGBoost | B_MIN | 169 | `b355c26339271f0b` |
| recovery48 | XGBoost | **B_MIN** (correction — Phase-2 prose once said B_FULL; the disk-authoritative `selected_models_v2.json` says B_MIN, re-verified again in Phase 4) | 316 | `5dda6bb0830428b4` |
| icu_stay_time | XGBoost (log1p target) | B_PLUS_F | 428 | `918c81eef86bd89e` |
| organ_support | XGBoost | B_FULL | 56 | `2fa653a0d2c644d8` |

Support postprocessing: raw XGBoost probability → `isotonic_support_v2`
(frozen, no refit) → calibrated probability → compare to threshold
**0.39781983118092895** (`>=`). ICU postprocessing:
`expm1(clamp_min(raw_log_prediction, 0))`.

## 2. Fresh cohort independence

1000 subjects, namespace `SYN-V2-S-*`/`SYN-V2-E-*` (zero string overlap and
zero clone-fingerprint overlap with v1's `SYN-S-*` cohort, verified in
Phase 3), generated from a seed derived deterministically from
`SHA256(selected_models_v2.json)` before any generation occurred
(`fresh_test_seed = 402995653`), through the same accepted, scientifically
unmodified generation pipeline as Benchmark v1.

## 3. Access event

Exactly one governed access event, through `src/performance_v2/fresh_test_access.py`:

```
SEALED_NOT_ACCESSED -> AUTHORIZED_FOR_ONE_FINAL_RUN -> FINAL_V2_TEST_ACCESS_CONSUMED -> FINAL_V2_RUN_COMPLETED
```

All four models were run in this single pass against the same loaded
fresh-cohort rows. A structured preflight (20/20 checks) validated every
frozen dependency, hash binding, and code path *before* authorization,
without ever reading a fresh-test row. A second-run guard, run after
completion, confirms a repeat attempt is rejected
(`BLOCKED_ALREADY_CONSUMED`) before any fresh-row loading could occur.

## 4. Denominators (legitimate only after the single access event)

| Task | Eligible rows | Unique stays |
|---|---:|---:|
| recovery24 | 3515 | 730 |
| recovery48 | 1331 | 319 |
| icu_stay_time | 6083 | 1000 |
| organ_support | 3953 | 778 (513 positive stays, 1784 positive rows, row-level prevalence 0.4513) |

## 5. Final metrics (stay-balanced, from saved predictions only)

### Recovery +24h — target: raw, unclipped ΔSOFA24

| Metric | Fresh-test value | 95% CI |
|---|---:|---:|
| **MAE (primary)** | 1.0744 | [1.0303, 1.1199] |
| RMSE | 1.3705 | [1.3155, 1.4257] |
| Weighted median AE | 0.8596 | [0.8067, 0.9038] |
| Naive (constant 0) MAE | 1.3017 | [1.2287, 1.3738] |
| **Relative improvement over naive** | **17.46%** | |

### Recovery +48h — target: raw, unclipped ΔSOFA48 (independent of +24h)

| Metric | Fresh-test value | 95% CI |
|---|---:|---:|
| **MAE (primary)** | 1.3379 | [1.2585, 1.4250] |
| RMSE | 1.6228 | [1.5377, 1.7139] |
| Weighted median AE | 1.1921 | [1.1166, 1.3056] |
| Naive (constant 0) MAE | 1.5761 | [1.4424, 1.7123] |
| **Relative improvement over naive** | **15.11%** | |

### ICU remaining time — postprocessed hours

| Metric | Fresh-test value | 95% CI |
|---|---:|---:|
| **Weighted median AE (primary)** | 5.2060 h | [5.0602, 5.3947] |
| MAE | 7.8095 h | [7.3590, 8.2571] |
| RMSE | 12.5667 h | [11.6382, 13.4608] |
| Naive (DEV median = 20.6147h) median AE | 9.8058 h | [9.6187, 9.9762] |
| **Relative improvement over naive** | **46.91%** | |

### Organ support — raw probability

| Metric | Fresh-test value | 95% CI |
|---|---:|---:|
| AUPRC | 0.7637 | [0.7274, 0.7960] |
| AUROC | 0.7869 | [0.7614, 0.8110] |
| Brier | 0.2016 | [0.1959, 0.2076] |

### Organ support — calibrated probability (primary discrimination metric = calibrated AUPRC)

| Metric | Fresh-test value | 95% CI |
|---|---:|---:|
| **AUPRC (primary)** | 0.7042 | [0.6612, 0.7437] |
| AUROC | 0.7772 | [0.7515, 0.8021] |
| Brier | 0.2024 | [0.1876, 0.2177] |
| F1 @ threshold | 0.6928 | [0.6633, 0.7202] |
| Precision | 0.6140 | [0.5788, 0.6496] |
| Recall / sensitivity | 0.7948 | [0.7635, 0.8240] |
| Specificity | 0.6011 | [0.5626, 0.6372] |
| Naive (DEV prevalence = 0.4337) AUPRC | 0.4439 | [0.4167, 0.4711] |
| **Relative improvement over naive** | **58.63%** | |

Calibrated AUPRC (0.7042) is somewhat lower than raw AUPRC (0.7637). Isotonic
regression is monotonic non-decreasing, so it cannot invert ranking — this
drop reflects ties introduced where the isotonic map flattens (visible in
§6's reliability table, where the top bin groups many distinct raw
probabilities into one plateau), not a ranking reversal. It is reported
plainly, not adjusted for.

## 6. Calibration evidence

Raw Brier = 0.2016, calibrated Brier = 0.2024 (a slight, not a material,
change). 10-equal-width-bin, stay-balanced reliability table (raw
probabilities cluster in [0.2, 0.7]; calibrated probabilities spread across
the full range and track observed frequency closely except for some
overconfidence in the top bin: mean predicted 0.997 vs. observed frequency
0.784, n=856). Full table and ROC/precision-recall/reliability plots:
`artifacts/performance_v2/phase4/metrics/calibration_evidence_v2.json`,
`artifacts/performance_v2/phase4/plots/`.

## 7. Validation-to-test generalization (descriptive only)

| Task | Phase-2 validation | Fresh V2 test | Difference |
|---|---:|---:|---:|
| recovery24 MAE | 1.0376 | 1.0744 | +0.0368 (slightly worse) |
| recovery48 MAE | 1.5373 | 1.3379 | −0.1994 (better) |
| icu_stay_time median AE | 5.0546 h | 5.2060 h | +0.1514 h (slightly worse) |
| organ_support AUPRC | 0.8555 (raw) | 0.7042 (calibrated) / 0.7637 (raw) | notably lower |

Recovery and ICU generalize closely (small, mixed-direction differences,
consistent with ordinary sampling variation between two independent
cohorts). Support shows a more substantial validation-to-test gap on AUPRC,
in both the raw and calibrated forms — reported here exactly as measured,
with no retraining or re-selection in response. No test result was used to
alter any model, feature, threshold, or calibration.

## 8. Prespecified, compact error analysis

Boundaries were frozen from DEV *before* fresh-test access
(`final_evaluation_plan_v1.json`); only these predeclared slices are
reported (no post-hoc subgroup search):

- **Cutoff timing** (EARLY/MID/LATE grid-index bands): ICU MAE is highest
  EARLY (11.22h) and lowest LATE (6.00h) — expected, since early cutoffs
  carry the least observed history. Recovery MAE is comparatively flat
  across timing bands (24h: 0.99–1.10; 48h: 1.11–1.36).
- **True-delta sign** (recovery only): both horizons have their lowest MAE
  on truly-zero deltas (24h: 0.633; 48h: 0.989) and higher MAE on truly
  positive deltas (24h: 1.394; 48h: 1.268) — the model under-reacts to
  larger true changes, a directionally sensible and common regression-to-
  the-mean pattern.
- **True remaining-time quartile** (ICU only, DEV-derived boundaries
  12.02h/20.61h/33.95h): MAE rises from Q2 (2.66h) to Q4 (11.54h) — the
  model is most accurate for short-to-typical stays and loses precision for
  the longest ones, again an expected pattern for a bounded feature history.

Cardiac-subtype and observation-density slices were frozen in the plan but
are reported in the machine-readable artifact only, for compactness; see
`artifacts/performance_v2/phase4/metrics/error_analysis_v2.json`.

## 9. No post-test tuning

`post_test_tuning: false` in `v2_final_evaluation_freeze_v1.json`. After the
Phase-4A pre-access commit, no model, feature, hyperparameter, training
budget, calibration, threshold, prediction rule, bootstrap seed/count, metric
definition, baseline definition, or prespecified slice was changed. The only
work performed after fresh-test access was deterministic computation of
metrics/bootstrap/calibration/error-analysis/plots from the immutable saved
prediction files.

## 10. Limitations

- **Synthetic benchmark.** All data — Benchmark v1 and this fresh V2 cohort
  — is generated by a documented synthetic process. No real patient data was
  used anywhere in this project.
- **No clinical validation.** The SOFA-style scoring, support process, and
  outcome definitions are project-designed synthetic constructs, not
  clinically validated instruments.
- **No external real-patient validation.** "Fresh" here means an
  independently-seeded synthetic cohort from the same generator, not an
  external or real-world dataset.
- **No deployment-efficacy claim.** These are offline, retrospective,
  synthetic-benchmark metrics. Nothing here demonstrates or implies efficacy,
  safety, or readiness for any clinical or deployment use.
- **Cohort-to-cohort variation.** The validation-to-test differences in §7
  reflect ordinary sampling variation between two disjoint synthetic
  cohorts, not necessarily a stable population-level effect; the 95% CIs in
  §5 quantify this uncertainty for the fresh-test point estimates themselves.

## Phase-4 completion

`PERFORMANCE_V2_PHASE4_COMPLETE / PERFORMANCE_V2_COMPLETE /
FRESH_V2_TEST_EVALUATED_ONCE / FINAL_V2_RUN_COMPLETED /
V2_FINAL_EVALUATION_FROZEN / NO_POST_TEST_TUNING`. No Phase 5.
