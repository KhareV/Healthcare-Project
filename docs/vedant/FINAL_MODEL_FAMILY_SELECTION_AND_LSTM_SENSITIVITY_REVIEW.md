# Final Model-Family Selection and LSTM Sensitivity Review

Status: **VEDANT FINALIZATION STAGE 2 COMPLETE**. Family selection is frozen from validation evidence only. LSTM is descriptive sensitivity evidence and is not serving-eligible.

## Frozen parents

- G1: `1fbcc1b0859c0052eb867aec15e89bd3f7df503ac318b30f92dd98ff9eeccf90`
- Sanskruti Phase-14 handoff: `3c4fa9a9efd25145ab1daa23c03632aa0cfd0b04d3ee0b4254c1cb180fff63d4`
- Phase-12 best XGBoost manifest: `c67af97b321d844e0bcb8f9777bc73fff4882cabda7fe7b78b21c4f8dc8d804a`
- Final-v2 best GRU manifest: `aea37e8d85b6894d5efd33b1a7d2e1969f683f02b8cef222d3504783d5917254`
- Stage-2 science implementation commit: `495ae7e3a4c1e595b4191a300f54865483084ee0`

The Phase-12 and final-v2 sources have identical split, feature-schema, and preprocessor hashes. Their validation subject/row keys, eligibility, and targets match. The only representational difference is the already-governed float32 flattening of XGBoost ICU-time targets; parity was verified within `1e-4` hours and `1e-6` log1p units. The failed Stage-1 v1 GRU history did not participate.

## Frozen XGBoost-versus-GRU selection

| Task | Selection metric | Best XGBoost | Best GRU | Frozen family/candidate |
|---|---:|---:|---:|---|
| Recovery | MAE24, lower | `1.1562660049418954` (`xgb-recovery-014`) | `1.2508119881095108` (`gru-recovery-024`) | **XGBoost / `xgb-recovery-014`** |
| ICU time | weighted median AE hours, lower | `11.814799558602587` (`xgb-icu-time-028`) | `9.197912417358399` (`gru-icu-time-026`) | **GRU / `gru-icu-time-026`** |
| Organ support | raw uncalibrated AUPRC, higher | `0.6441122590894012` (`xgb-support-024`) | `0.6342497969245816` (`gru-support-001`) | **XGBoost / `xgb-support-024`** |

Every decision was resolved by its primary metric; no tie-break was needed. Support probabilities remain raw and uncalibrated, and no operating threshold was selected.

## Fixed LSTM sensitivity

Exactly one LSTM was run per task using the corresponding best-GRU seed and hyperparameters, changing only the recurrent cell. No LSTM search occurred and these results did not alter the frozen family selection.

| Task | Best GRU primary | Fixed LSTM primary | LSTM − GRU | GRU params | LSTM params | Best epoch |
|---|---:|---:|---:|---:|---:|---:|
| Recovery MAE24 | `1.2508119881095108` | `1.2604288439943399` | `+0.009616855884829079` | 74,386 | 99,090 | 7 |
| ICU weighted median AE hours | `9.197912417358399` | `11.221936060791014` | `+2.0240236434326153` | 173,321 | 231,049 | 6 |
| Support AUPRC | `0.6342497969245816` | `0.6183570179847763` | `-0.01589277893980534` | 74,249 | 98,953 | 1 |

Checkpoint SHA-256 values:

- Recovery: `3ae246f59a2fd9f35662db10bc19c8630487289c6ae31ddaa9167f556f4caa2e`
- ICU time: `47cb1f9457716dd7f43f039619386b28f4444f704473589274173a70f3324ce2`
- Organ support: `c4f6b33cf60e74cef081992cc6b0be6e673b1686a8d8c123dc3cae56082a2a28`

## Governed artifacts and verification

- Validation family selection: `artifacts/selection/validation_family_selection_v1.json`, SHA-256 `0ccc7296618be5c6db1f93bc8b9f22f48c664f6b7abc5c80072828eedb8c5016`
- LSTM sensitivity manifest: `artifacts/models/lstm_sensitivity_v1.json`, SHA-256 `0040a296876613cb714dac96c9a57d930750484e359be4d393d899ebee84ad8e`
- Focused post-run suite: **59 passed**
- Corrected full repository regression: **1,289 passed, 2 skipped**
- Stage-2, registry/lineage, G1, Phase-14, Phase-12, final-GRU, and system-evidence audits: **PASS**

The recovery training completed before a post-training registry indexing bug was found. Its exact checkpoint and predictions were preserved and registered without retraining; ICU and support then ran once. Thus exactly three scientific LSTM training runs occurred.

## Test lock and Stage-3 readiness

`test_accessed=false` throughout. No test labels, predictions, metrics, error inspection, or bootstrap CIs were used. `selected_models_v1.json` and G3 remain absent. Support calibration and threshold selection remain unset. Pulkit serving routing and production explanation generation were not changed.

Stage 3 may now freeze the selected models, fit validation-only isotonic calibration for the selected XGBoost support classifier, choose its validation F1 threshold, and create G3 after all Stage-3 audits pass.
