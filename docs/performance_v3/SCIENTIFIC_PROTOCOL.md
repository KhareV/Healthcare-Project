# Performance V3 — Scientific Protocol (narrative)

Authoritative machine-readable form: `configs/performance_v3/scientific_protocol_v1.yaml`
(hash recorded in `artifacts/performance_v3/governance/v3_protocol_freeze_v1.json`).
This file explains the design; if the two disagree, the YAML governs.

## Why these experiments (evidence already in the repo)

* V2 Phase-1 diagnosis labelled every task `REPRESENTATION_LIMITED` and found four cheap context
  features (elapsed time, cutoff index, hours since first cutoff, current SOFA) gave nearly all
  achievable gain; groups C–F added little and overfit on ~1,700 development subjects.
* V2 final support model used 56 trees (`final_training_budget`) and its validation AUPRC (0.8555)
  fell to 0.7042 (calibrated) / 0.7637 (raw) on the fresh test — a large optimism gap that a bigger
  development pool can directly address.
* Isotonic calibration lowered AUPRC (0.7637 → 0.7042) and AUROC (0.7869 → 0.7772) without improving
  Brier (0.2016 → 0.2024): ties from a step function. V3 compares ranking-preserving calibrators.
* The generator has an irreducible noise floor (ICU duration noise N(0, 0.35) log-hours; latent
  innovation/shock). V3 therefore expects modest gains and treats null results as valid.

## Partitions (all sizes fixed before generation)

| Role | Subjects | Namespace | Used for |
|---|---|---|---|
| V1 train+validation (historical, unconsumed) | 1,700 | `SYN` | development pool |
| New development chunks D1–D3 | 3 × 2,100 | `SYN-V3-D1..3` | development pool |
| **Development pool** | **8,000** | — | cross-fitted selection, final refit |
| Learning-curve evaluation (LCE) | 1,500 | `SYN-V3-LCE` | learning curves only |
| Sealed V3 test | 1,500 | `SYN-V3-T` | one paired V2/V3 evaluation |
| V1 test (300), V2 fresh test (1,000) | — | `SYN`, `SYN-V2` | exposed, never used |

Seeds come from SHA-256 of the frozen protocol (dev, LCE) or of `selected_models_v3.json` (test).

## Stages

1. **Learning curve** (RQ1): frozen V2 recipe at 425/850/1,700/3,800/5,900/8,000 subjects, evaluated on LCE.
2. **S1 family screen**: XGBoost, LightGBM, CatBoost (10 random configs each), ridge/logistic (6), 3 folds, V2 feature variant.
3. **S2 feature ablation**: best config per task × 8 feature variants, 3 folds × 2 seeds.
4. **S3 GRU**: 8 configs, 3 folds, context concatenated to statics.
5. **S4 focused search**: top-3 (family, variant) per task × 20 configs, 5 folds.
6. **Calibration**: nested stay-grouped cross-fit of raw / Platt / isotonic / beta by log loss, simplicity tie-break.
7. **Final refit** on the 8,000-subject pool; calibrator + threshold from full-pool OOF scores.
8. **Pre-test freeze** → generate and seal SYN-V3-T → one paired evaluation.

## Decision rules fixed in advance

* Selection: lowest mean cross-fitted primary metric per task (highest AUPRC for support, on raw scores);
  ties within 0.5 % relative (regression) or 0.003 AUPRC are resolved by lower complexity, then lower fold SD, then latency.
* V2 recipe is replaced only if beaten on the same folds by more than the tie tolerance.
* Promotion (per task): paired Δ favours V3 and the 98.75 % (Bonferroni over four tasks) paired
  stay-cluster CI excludes 0; support additionally requires Δ Brier ≤ +0.003. Otherwise V2 is retained.
* A secondary control — the V2 recipe refit on the 8,000-subject pool — separates data-size from modelling effects.
* Everything is reported, including regressions and failed trials.

## Information parity and leakage

All V3 features are deterministic functions of the canonical row at cutoff `t`
(`history_values`, `observation_mask`, `tslo_hours`, statics, `prediction_time − intime`, support-state bins,
SOFA at cutoffs ≤ t via the frozen `sofa_at`). Labels, eligibility, `outtime` and generator latents are
forbidden inputs and are covered by perturbation tests. Sequence models receive the identical tensor plus the
same context vector.
