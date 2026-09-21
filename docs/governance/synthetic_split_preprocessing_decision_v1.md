# Synthetic Split and Preprocessing Decision v1

Status: **FROZEN_SYNTHETIC_AUTHORIZED**

Decision authority: **USER_DELEGATED_AI_PROJECT_DECISION**

Scope: authorized final synthetic benchmark only

This versioned decision resolves the `SYNTHETIC SUBJECT SPLIT POLICY` that
Project Scope v2 deliberately left unlocked before Phase 10. It does not edit
or erase the historical MIMIC-era `split_spec_v1`; that contract remains a
superseded record and must not be applied to synthetic subjects.

## Frozen split decision

- Split unit: `subject_id`.
- Fractions: 70% train, 15% validation, 15% sealed final test.
- Assignment: rank exact raw-trajectory clone groups by SHA-256 using root
  seed `20260921` and namespace
  `healthcare-project/synthetic-subject-split/v2`, then allocate exact counts
  where clone-group sizes permit.
- Current 2,000-subject artifact: 1,400 train, 300 validation, 300 test.
- No labels, outcomes, eligibility, prevalence, model performance,
  stratification, or invented calendar/anchor year participates in assignment.
- Exact identity-free raw-trajectory clones are grouped before assignment and
  may never cross partitions. The final Phase-9 cohort contains zero such
  clones.
- The assignment CSV and metadata are immutable hash-bound artifacts.

## Frozen preprocessing decision

Only training-subject rows fit learned values. Dynamic missing cells use the
per-channel median of genuine training observations. Active (non-padding)
cells are then standardized with population mean and standard deviation after
imputation; a zero standard deviation maps to unit scale. Structural padding
maps to numeric zero after scaling and remains separately identified by the
unchanged padding mask. Genuine-observation masks remain unchanged. TSLO stays
in frozen hours, including the `54.0` no-observation sentinel.

Age uses training-only population standardization. Sex and cardiac-condition
categories use full one-hot vocabularies learned from training subjects, with
an explicit `__UNKNOWN__` bucket for later unseen inputs. Recovery targets use
independent training-eligible 24-hour and 48-hour population statistics. ICU
time remains in its already-defined `log1p(hours)` domain. The support positive
class weight is `N_negative/N_positive` over eligible training rows only.

## Final-test seal

Phase 10 assigns final-test subjects but does not materialize a transformed
test dataset, compute test label summaries, train a model, tune a
hyperparameter, fit calibration, choose a threshold, or run evaluation. A
model-ready test artifact may be produced only after a G3-equivalent freeze
authorizes the one-time final-test path.

## Authoritative artifacts

- `configs/synthetic/split_preprocessing_v1.yaml`
- `configs/synthetic/model_input_contract_v1.json`
- `artifacts/splits/synthetic_split_v2.csv`
- `artifacts/splits/synthetic_split_v2.metadata.json`
- `artifacts/preprocessors/synthetic_feature_preprocessor_v1.json`
- `artifacts/preprocessors/recovery_target_scaler_synthetic_v1.json`
- `artifacts/preprocessors/support_class_weight_synthetic_v1.json`
- `artifacts/manifests/synthetic_phase10_manifest_v1.json`

Any change to a fraction, seed, namespace, clone definition, imputation,
normalization, encoding, target scaling, or class-weight rule requires a new
version and invalidates downstream fitted/model artifacts.
