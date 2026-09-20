# Phase 15 Prespecified Error Analysis — Sanskruti Review Package

## Scope and authority

This phase implements Vedant-owned error-analysis tooling only. The Final
Implementation Master Plan defines the eight slice families and forbids
cherry-picking; the Final Work Division assigns error analysis to Vedant with
Sanskruti as reviewer. Neither document freezes numeric band boundaries,
category mappings, missing-value handling, or a minimum sample-size threshold.
No Phase 16 registry implementation or Pulkit-owned serving work was added.

## Repository state and execution decision

The pre-change Phase 0–14 suite passed (`318 passed`). There is no real
`artifacts/models/selected_models_v1.json`, no real frozen validation prediction
artifact, and no approved complete slice specification. Consequently no real
validation results were generated and no test artifact was opened.

- Real validation analysis: **BLOCKED — frozen slice definitions, selected
  models, and compatible validation predictions required**.
- Final test: **FINAL TEST ERROR ANALYSIS — DEFERRED UNTIL PHASE 20**.
- Synthetic output only:
  `artifacts/error_analysis/synthetic/error_analysis_table.csv`.

## Review targets

1. `configs/error_analysis/error_analysis_v1.json` names only the eight
   prespecified families, gives explicit task applicability, and records every
   unresolved definition rather than inventing it.
2. `src/evaluation/slices.py` rejects prediction/error-derived definitions,
   requires all groups to be declared, and assigns deterministically.
3. `src/evaluation/error_analysis.py` accepts prediction records only, validates
   selected-model and split/feature/label lineage, rejects mixed splits and
   pre-Phase-20 test access, and emits all declared groups including empty ones.
4. Recovery horizons are separate; ICU-time predictions are evaluated in
   hours; support uses one supplied global probability stage and threshold.
5. Each group calls the Phase 9 evaluator after subsetting, which recomputes
   `1/n_i` within the group. No row weights are accepted from callers.
6. Complete-component assignment directly calls Phase 14
   `complete_component_included`; there is no duplicate completeness formula.
7. The table includes model/data lineage, denominators, support positive counts,
   undefined-metric status, probability type, and threshold ID.

## Frozen config identity

- Version: `prespecified_error_analysis_slice_spec_v1`
- File SHA-256 at review-package creation:
  `6d3fd2742bce88f9bf81b582418eb41b45f3cb0d803cacc1a3cfa03d373630fe`
- Status: `FRAMEWORK_ONLY_REAL_SLICE_EXECUTION_BLOCKED`

Any reviewed edit changes the file hash and must be propagated into generated
error-analysis lineage.

## Unresolved reporting/scientific inputs

- `BLOCKED — BASELINE SOFA BAND DEFINITIONS REQUIRED`
- `BLOCKED — REMAINING ICU-STAY SLICE BOUNDARIES REQUIRED`
- `BLOCKED — MISSINGNESS / OBSERVATION-DENSITY SLICE DEFINITION REQUIRED`
- `BLOCKED — ICU-TYPE MAPPING AND MISSING-VALUE POLICY REQUIRED`
- `BLOCKED — EARLY/LATER CUTOFF DEFINITION REQUIRED`
- `NOT REPORTED — RELIABLE MEDICAL/SURGICAL MAPPING UNAVAILABLE`
- `BLOCKED — BASELINE SUPPORT SLICE TASK-APPLICABILITY RULE REQUIRED`
- `UNLOCKED REPORTING PARAMETER — SLICE MISSING-VALUE POLICY`
- `UNLOCKED REPORTING PARAMETER — MINIMUM SLICE SAMPLE SIZE`
- ICU-time error-percentile levels remain the Phase 9 unlocked reporting
  parameter and were not invented here.

## Test evidence expected for sign-off

The Phase 15 tests cover deterministic/baseline-only assignment, cutoff and
support-at-t behavior, padding/missingness separation, conditional category
mapping, Phase 14 completeness reuse, within-group stay weights, task
eligibility, separated recovery horizons, hours-domain ICU metrics, one-class
support behavior, denominators/empty groups, selected-model binding, global
calibration/threshold lineage, blocked definitions, and test isolation.

This is descriptive infrastructure. It performs no hypothesis test and makes no
causal or statistical-significance claim.
