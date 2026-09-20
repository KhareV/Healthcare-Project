# Pulkit Phase 12 — Recovery Reconstruction Review

## 1. PULKIT PHASE 12 SUMMARY

The existing canonical independent-horizon reconstruction is hardened and integrated behind `PredictionPipeline`. Synthetic serving now obtains explicit SOFA(t), validates original-unit delta provenance, reconstructs both horizons from the same baseline, clips only display values, and preserves raw deltas. A pure dashboard data transform binds validated results without repeating arithmetic. No UI or replay control was added.

## 2. REPOSITORY STATE BEFORE CHANGES

Phase 11 had a recovery-provider seam but delegated reconstruction to fixtures. The canonical model utility existed, while a production current-SOFA serving interface did not. Pre-change suite: 808 passed, 2 dependency skips.

## 3. EXISTING RECOVERY-RECONSTRUCTION AUDIT

`src/models/recovery_output.py::reconstruct_absolute_sofa` already implemented same-baseline addition and optional display clipping. It was reused rather than copied. No reconstruction import exists in raw-delta metrics, training tasks, or selection code.

## 4. CURRENT-SOFA INTERFACE STATUS

`CurrentSOFAProvider` and `CurrentSOFAState` define the handoff: exact stay, exact prediction cutoff, finite `[0,24]` value, SOFA version, source version/hash, and optional six component-observed flags. No SOFA component calculation is implemented. Production `sofa_at(t)` remains absent.

## 5. FILES CREATED

- `src/serving/recovery.py`
- `src/serving/recovery_display.py`
- `tests/recovery_helpers.py`
- `tests/test_recovery_reconstruction.py`
- `tests/test_recovery_display_transform.py`
- `tests/test_api_recovery_reconstruction.py`
- `docs/pulkit/PHASE12_RECOVERY_RECONSTRUCTION_REVIEW.md`

## 6. FILES MODIFIED

- `src/models/recovery_output.py`: finite/current-range validation around existing arithmetic.
- `src/serving/interfaces.py`: recovery postprocessor receives stay, cutoff, and selected identity.
- `src/serving/pipeline.py`: passes those exact bindings to postprocessing.
- `src/serving/postprocessing.py`: forwards recovery context while retaining canonical ICU behavior.
- `api/main.py`: maps recovery contract failures to safe 503 responses.
- Synthetic test postprocessors were updated for the context signature; shared fixtures now reuse the canonical reconstruction utility.

## 7. CANONICAL RECONSTRUCTION FUNCTION

The sole formula remains `reconstruct_absolute_sofa`. It adds the two-column delta tensor to one baseline tensor and optionally clamps the absolute result to `[0,24]`. No second serving or dashboard formula exists. No new transform-version identifier was invented because the existing utility had none.

## 8. CURRENT-SOFA PROVENANCE

The serving layer verifies cutoff/stay equality, SOFA spec version, source version/hash, value range, and optional component-flag shape. Flags are audit/data-quality information only; all-false flags do not create a completeness threshold.

## 9. DELTA24 OUTPUT-DOMAIN VALIDATION

The normalized contract requires `original_sofa_delta_points`, explicit horizon order, finite Delta24, and either model-native original units or one frozen inverse transform. Untyped/standardized values are rejected by the explicit adapter.

## 10. DELTA48 OUTPUT-DOMAIN VALIDATION

Delta48 has the same original-unit requirements and must be named as the second independent horizon. Swapped order, nonfinite values, and invalid/double-inverse provenance fail closed.

## 11. 24H RECONSTRUCTION RESULT

The canonical function computes `SOFA(t) + Delta24`. Tests cover lower/upper clipping, exact boundaries, in-range results, large deltas, and continuous values without rounding.

## 12. 48H RECONSTRUCTION RESULT

The canonical function computes `SOFA(t) + Delta48` using the same SOFA(t). No 24-hour result or delta enters this calculation.

## 13. DOUBLE-COUNTING REGRESSION TEST

For current SOFA 10, Delta24 +6, and Delta48 -3, results are 16 and 7. The test explicitly rejects 13, which would reveal illegal `10 + 6 - 3` chaining.

## 14. HORIZON-INDEPENDENCE TEST

Changing Delta24 leaves the 48-hour reconstruction unchanged; changing Delta48 leaves the 24-hour reconstruction unchanged. Swapping deltas swaps reconstructed horizons correspondingly.

## 15. DISPLAY CLIPPING RESULT

For current 22 and deltas +8/-30, unbounded absolute values are 30/-8 and displayed values are 24/0. Exact 0 and 24 remain unchanged. Invalid baseline SOFA is rejected rather than sanitized.

## 16. RAW-DELTA PRESERVATION RESULT

The raw delta tensor is unchanged by canonical display reconstruction. Pipeline responses retain +8/-30 or any other finite deltas even when displayed SOFA clips. No integer rounding occurs.

## 17. METRIC/TRAINING NONINTERFERENCE

Evaluation continues to declare `raw_sofa_delta_unclipped`; metric, selection, loss, optimizer, and target-scaler paths do not import serving reconstruction. No labels, training artifacts, calibration, thresholds, or selection rankings were changed.

## 18. MODEL-FAMILY PARITY

The recovery postprocessor has no GRU/XGBoost branch. Identical normalized current SOFA and deltas produce identical recovery payloads for either valid selected-family identity.

## 19. FUTURE-SOFA NONACCESS

The current provider exposes only `current_sofa(stay_id, prediction_time)`. Call-spy tests show exactly one request at t and none at t+24/t+48. Phase 7 future-row mutation/deletion/addition leaves Phase 12 API output unchanged at t.

## 20. SOFA VERSION COMPATIBILITY

The postprocessor is configured with an expected SOFA version and rejects mismatch. This expected version cannot yet be bound to a real selected artifact because the production handoff is absent.

## 21. API / DIRECT PIPELINE RECOVERY EQUALITY

Direct pipeline and HTTP recovery objects are JSON-identical, including both raw deltas and both reconstructed values. FastAPI performs no recovery arithmetic.

## 22. PHASE-5 SCHEMA INTEGRATION

The existing field names remain `delta_24h`, `delta_48h`, `reconstructed_sofa_24h`, and `reconstructed_sofa_48h`. Successful Phase 12 outputs validate through `prediction_schema_v1`; no aliases or direction/confidence fields were introduced.

## 23. DASHBOARD DATA-TRANSFORM CONTRACT

`RecoveryDisplayData` distinguishes observed current SOFA from predicted +24/+48 SOFA and preserves raw deltas/cutoff/display range. Its pure builder validates and binds API fields; it performs no arithmetic, clipping, files, network, models, Streamlit, Plotly, or future-actual binding. Rounding and clipping indicators remain unlocked presentation parameters.

## 24. TESTS ADDED

Tests cover canonical reuse, double counting, horizon independence/swap, clipping boundaries, raw preservation, continuous values, nonfinite/range failures, output domain/order/provenance, one inverse adapter call, exact SOFA(t) call, version/cutoff/stay attacks, component flags, later cutoff state, family parity, metric/training isolation, API equality, future perturbation, and pure display binding.

## 25. EXACT COMMANDS EXECUTED

```text
PYTHONPATH=src:tests python3 -m pytest -q
rg -n -i "reconstruct(_sofa)?|sofa_hat|clip.*sofa|display.*sofa|delta[_ ]?sofa24|delta[_ ]?sofa48|reconstructed_sofa" src tests configs docs
rg -n "def sofa_at|sofa_at\\(|current_sofa|SOFA_SPEC|sofa.*version" src tests configs artifacts
PYTHONPATH=src:tests python3 -m pytest tests/test_recovery_reconstruction.py tests/test_recovery_display_transform.py tests/test_api_recovery_reconstruction.py -q
PYTHONPATH=src:tests python3 -m pytest tests/test_prediction_schema.py tests/test_prediction_pipeline.py tests/test_preprocessor_serving.py tests/test_training_serving_equivalence.py tests/test_history_truncation.py tests/test_serving_feature_leakage.py tests/test_api_predict.py tests/test_api_errors.py tests/test_api_metadata.py tests/test_api_health.py tests/test_api_pipeline_equality.py tests/test_icu_time_serving_postprocess.py tests/test_gru_recovery.py tests/test_lstm_recovery.py tests/test_regression_metrics.py tests/test_metric_model_parity.py -q
python3 -m compileall -q api src tests
PYTHONPATH=src:tests python3 -m pytest -q
PYTHONPATH=src python3 -c "from pathlib import Path; from vedant_infra.g3 import audit_g3; print(audit_g3(Path('.'), scope='real'))"
```

## 26. TEST RESULTS

Pre-change: 808 passed, 2 skipped. Final Phase 12 focused: 29 passed; combined Phase 12/API verification: 42 passed. Relevant Phase 5–11 regression: 141 passed. Compilation passed. Final repository result: **837 passed, 2 skipped** in 5.25 seconds. The read-only G3 audit remained blocked with `test_data_accessed=False` and no freeze marker. Existing dependency skips remain Captum and genuine SHAP/XGBoost execution. No formatter, linter, or type-checker configuration exists, so none was invented or run.

## 27. REAL CURRENT-SOFA STATUS

**BLOCKED — VERIFIED PRODUCTION `sofa_at(t)` OR EQUIVALENT CURRENT-SOFA ARTIFACT INTERFACE REQUIRED.** Synthetic current state is explicit and cannot authorize real serving.

## 28. REAL SELECTED-RECOVERY-MODEL STATUS

**BLOCKED — REAL SELECTED RECOVERY ARTIFACT, LOADER, ORIGINAL-UNIT OUTPUT HANDOFF, AND TARGET-TRANSFORM PROVENANCE REQUIRED.** No production inverse transform or XGBoost structure was guessed.

## 29. BLOCKED / UNLOCKED PARAMETERS

Blocked: real current SOFA, SOFA-version binding, real selected recovery model, original-unit normalization implementation, real Phase 7 input, and owner approvals. Unlocked: dashboard numeric rounding and clipping-indicator presentation.

## 30. SANSKRUTI REVIEW NOTES

Review current-SOFA source/version/cutoff, absence of future SOFA access, same-baseline formulas, exact `[0,24]` display clip, raw-delta preservation, lack of SOFA reimplementation, and component flags as quality rather than confidence.

## 31. VEDANT REVIEW NOTES

Review exactly-once original-unit normalization, explicit horizon order, unchanged raw deltas, absence from metrics/training/selection, family-independent transform, Phase 5/API equality, and continued test nonaccess.

## 32. DEFINITION-OF-DONE CHECKLIST

Framework items are complete: canonical reuse, current-SOFA and delta contracts, independent horizons, display-only clipping, raw preservation, family parity, hostile tests, pipeline/API integration, pure view data, no UI, and full regression. Real artifacts/interfaces and owner reviews remain incomplete.

## 33. REAL PHASE-12 STATUS

**BLOCKED.** Synthetic framework completion does not satisfy the real checklist because production SOFA and selected recovery-model handoffs are absent.

## 34. READINESS FOR PULKIT PHASE 13

The reconstruction boundary and pure display data are ready for UI composition. Phase 13 may consume the validated API/view model, but must not duplicate arithmetic or expose future actual SOFA. No dashboard or replay controls were implemented here.
