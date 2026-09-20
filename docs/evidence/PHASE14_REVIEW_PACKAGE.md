# Phase 14 review package

Status: **recovery-ablation and SOFA-completeness framework implemented with
synthetic evidence only**.

Real feature-ablation results are
`BLOCKED — FROZEN REAL RECOVERY MODEL REQUIRED FOR SCIENTIFIC ABLATION RESULTS`.
Final-test complete-component sensitivity is
`DEFERRED UNTIL AUTHORIZED FINAL TEST PHASE`.

## Framework boundaries

- Recovery task only; ICU-time and support are not included.
- Feature ablations remove declared model-facing feature groups without changing
  canonical values, imputation, structural padding, targets, eligibility, split,
  or non-ablated inputs.
- Hyperparameter-search budget is fixed to zero. The parent family/config/run and
  artifact hash remain bound to every variant.
- Complete-component sensitivity does not retrain. It independently requires all
  six baseline and relevant-future SOFA component flags for 24h and 48h, then
  calls the Phase-9 evaluator on the subset so stay weights are recomputed.
- Tables are generated from evaluator output; no performance direction is required.

## Required Sanskruti review/freeze before real execution

- Supply the production `feature_schema_v1` and its hash.
- Identify exactly which model-facing channels “mask ablation” removes. Task
  eligibility is forbidden; padding semantics remain structurally enforced.
- Supply/freeze the production TSLO encoding and feature group.
- Approve the exact variant matrix.
- Approve the no-retuning retraining policy and project seed.
- Supply the six authoritative SOFA component-observed flags at baseline and
  future cutoffs, with recovery prediction/label lineage.
- Verify real XGBoost/GRU parent artifact compatibility after selection exists.

No Phase 15, test evaluation, serving, API, dashboard, or model-selection change
is included.
