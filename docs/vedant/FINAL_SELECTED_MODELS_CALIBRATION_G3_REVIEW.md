# Vedant Finalization — Stage 3 Review

Selection / calibration freeze (G3) for the synthetic adult cardiac retrospective
sequential-forecasting benchmark. Test data remains untouched.

## Frozen scientific parents

| Parent | Path | SHA-256 |
|---|---|---|
| G1 (data/label freeze) | `artifacts/acceptance/g1_synthetic_data_freeze_v1.json` | `1fbcc1b0859c0052eb867aec15e89bd3f7df503ac318b30f92dd98ff9eeccf90` |
| Phase-14 handoff | `artifacts/handoffs/sanskruti_phase14_handoff_v1.json` | `3c4fa9a9efd25145ab1daa23c03632aa0cfd0b04d3ee0b4254c1cb180fff63d4` |
| Final-v2 GRU search | `artifacts/search/gru/final_v2/best_gru_candidates_v1.json` | `aea37e8d85b6894d5efd33b1a7d2e1969f683f02b8cef222d3504783d5917254` |
| Stage-2 family selection | `artifacts/selection/validation_family_selection_v1.json` | `0ccc7296618be5c6db1f93bc8b9f22f48c664f6b7abc5c80072828eedb8c5016` |
| LSTM sensitivity (sensitivity-only) | `artifacts/models/lstm_sensitivity_v1.json` | `0040a296876613cb714dac96c9a57d930750484e359be4d393d899ebee84ad8e` |
| Split | `artifacts/splits/synthetic_split_v2.csv` | `3d6af5298f219f4608e5318ca09856642a1918a1154563ed0e8aec1785563cf6` |

## Selected task models (frozen from Stage 2, unchanged in Stage 3)

| Task | Family | Candidate | Explanation method |
|---|---|---|---|
| Recovery (ΔSOFA24 / ΔSOFA48) | xgboost | `xgb-recovery-014` | tree_shap |
| ICU-like episode remaining time | gru | `gru-icu-time-026` | integrated_gradients |
| Organ-support initiation | xgboost | `xgb-support-024` | tree_shap |

No LSTM candidate is selected for serving in any task.

## Support calibration (`xgb-support-024` only)

- Method: isotonic regression, `y_min=0`, `y_max=1`, `out_of_bounds=clip`, fit on
  validation-only, organ-support-eligible rows, using stay-balanced weights
  (`1 / eligible_rows_for_stay`).
- Input: frozen Phase-12 validation raw uncalibrated probabilities for
  `xgb-support-024` — no re-inference.
- Model identity bound to the concrete `organ_support.json` model file
  (`sha256 52b8b87d691e420e0109d54819ab01b377e926ff2e577328adf05450d5797a7c`),
  not the bundle descriptor.
- Validation eligible population: 233 ICU-like stays, 1116 rows (521 positive
  rows, 146 positive stays).
- Artifact: `artifacts/calibration/isotonic_support_v1.json`
  (`sha256 9fe6d0e7c22d15d3683d11182e6a33a4692fd145fb8bd6a572f5af3a5ad8065f`)
- Calibrated validation predictions:
  `artifacts/calibration/support_validation_calibrated_predictions_v1.jsonl`

## Support threshold

- Chosen after calibration, maximizing stay-balanced validation F1 over sorted
  unique calibrated validation probabilities (plus 0.0/1.0), with deterministic
  tie-break (highest weighted F1 → closest to 0.5 → higher threshold).
- Value: `0.3608842426222367`
- Validation weighted F1: `0.6702135852717455` (precision `0.5900`, sensitivity
  `0.7756`, specificity `0.5669`)
- Artifact: `artifacts/thresholds/support_threshold_v1.json`
  (`sha256 4453d5eccc8bc0ec1be74a391dc4ac203bd4a01d5bb30019763707e7b88494ad`)

## Selected models manifest

- `artifacts/models/selected_models_v1.json`
  (`sha256 3d4fb0b1e0341b0ebe29a54e9581bc47d3b80c536d3ee22f830276b4631fbb0b`)
- `status = SELECTED_MODELS_FROZEN_PRE_TEST`, `serving_ready = false`,
  `test_accessed = false`.

## Explanation routing (method only, not computed)

- Recovery (xgboost) → tree_shap
- ICU time (gru) → integrated_gradients
- Organ support (xgboost) → tree_shap

Production explanation generation is deferred to Stage 4.

## G3 (selection / calibration freeze)

- `artifacts/governance/g3_freeze.json`
  (`sha256 d94d49b0a438027842dbbf1c6130404297230f59b18bc863320ac753db770564`)
- `status = G3_ACTIVE`; `all_model_choices_frozen`, `calibration_frozen`,
  `threshold_frozen`, `explanation_routing_frozen` all `true`;
  `test_accessed = false`.
- Governance authority: `USER_DELEGATED_AI_PROJECT_DECISION`,
  authorization `STAGE3_G3_FREEZE_REQUESTED`, `human_member_signoff_claimed = false`.

### Pre-test correction history

An initial G3 marker (`sha256 2ca3a9cbe6616785...`) recorded a governance
dependency path pointing at the obsolete `split_v1.metadata.json` while
carrying the correct `synthetic_split_v2` hash. Validation rejected it before
any test access occurred. The invalid marker was archived to
`artifacts/governance/history/invalid_pretest_2ca3a9cbe6616785.json`, the
correction event was recorded in
`artifacts/governance/test_access_state.json`, and a corrected marker
(current `g3_freeze.json`) was created and validated. No
`FINAL_TEST_ACCESS_CONSUMED` event exists at any point in this history.

## Final-test governance

- State: `AUTHORIZED_NOT_RUN` (Stage 5 may later consume test access; it has
  not been opened).
- `test_accessed = false`; `FINAL_TEST_ACCESS_CONSUMED` absent.
- No final-test metric, prediction, or error-analysis artifact was created.

## Audits / tests

- `scripts/audit_phase12_xgb_search.py`: `PASS` (corrected to allow a
  downstream-legitimate `selected_models_v1` created after Phase 12, while
  still failing closed on any change to an artifact Phase 12 recorded as
  present).
- `scripts/audit_final_gru_search.py`: `PASS`
- `scripts/audit_stage2_finalization.py`: `PASS`
- `scripts/audit_phase14_handoff.py`: `SANSKRUTI_TRACK_COMPLETE`
- `experiments.g3_cli audit`: all per-item checks `PASS` except the
  freeze-readiness `test_nonuse` check, which reports `BLOCKED` post-freeze
  by design (test access is now authorized, so re-freezing is correctly
  barred).
- Registry lineage audit: no hash mismatches, no missing/broken artifacts.
- Focused Stage-3/governance suite: 96 passed.
- Full repository regression: 1295 passed, 2 skipped, 0 failed, 1297
  collected.

## Stage-4 readiness

Zero new model-training runs occurred in Stage 3. Serving integration,
production explanation generation, and final-test evaluation remain
unimplemented and are explicitly out of scope until Stages 4 and 5.
