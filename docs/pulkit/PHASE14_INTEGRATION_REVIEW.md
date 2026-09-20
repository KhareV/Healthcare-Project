# Pulkit Phase 14 Integration and Artifact-Compatibility Review

## 1. PULKIT PHASE 14 SUMMARY

The synthetic full-stack integration and hostile artifact-compatibility framework are complete. Phase 14 reuses the Phase-6 resolver, the Phase-7 canonical history boundary, the existing prediction pipeline, FastAPI, and the Phase-13 dashboard; it introduces no second scientific or artifact path.

## 2. REPOSITORY STATE BEFORE CHANGES

The pre-change suite passed 864 tests with 2 skips. Real `artifacts/models/selected_models_v1.json` and its frozen real task/calibration dependencies were absent. The final test remained unopened and no G3 marker existed.

## 3. EXISTING ARTIFACT-COMPATIBILITY AUDIT

`src/serving/artifacts.py` already performed hash-first manifest resolution, exact task/family/model/preprocessor binding, train-fit provenance validation, support calibrator/threshold binding, real-registry checks, safe relative paths, and synthetic/real scope isolation. Phase 14 extended coverage rather than duplicating it.

## 4. FILES CREATED

Created `tests/integration_helpers.py`, `tests/test_integration.py`, `tests/test_artifact_compatibility_e2e.py`, `tests/test_serving_lineage_e2e.py`, and this review.

## 5. FILES MODIFIED

Modified `src/serving/preprocessing.py` to reconcile earliest-cutoff padded feature positions with the frozen Phase-5 data-quality identity while retaining a separate padding-bin count. No model, artifact, registry, or governance state was changed.

## 6. END-TO-END INTEGRATION ARCHITECTURE

The exercised path is synthetic raw timeline → legal cutoff → canonical truncator/builder → family-specific canonical view → validated serialized-preprocessor identity → selected mixed-family predictors → recovery/ICU/support postprocessing → explanation router → prediction schema → FastAPI → dashboard replay/view/render.

## 7. INTEGRATION FIXTURE DESCRIPTION

The existing explicitly synthetic Phase-7 timeline and Phase-6 serving bundle are reused from repository-relative paths and copied to temporary directories. They include two stays, multiple bins, missing values, masks, TSLO, static data, future rows, three legal cutoffs, mixed families, current SOFA, calibration, and explanations. They are not the Phase-15 final demo fixture.

## 8. MIXED-FAMILY CONFIGURATION

The selected synthetic manifest specifies recovery GRU, ICU-time XGBoost, and organ-support GRU. Loader calls, response task identities, and dashboard explanation labels are independently asserted.

## 9. RAW HISTORY -> CUTOFF RESULT

At every tested cutoff, the existing canonical truncator retains only rows available at or before the exact legal timestamp. Unknown stays map to 404 and off-grid cutoffs to 422 before model inference.

## 10. FEATURE-BUILDER / PREPROCESSOR RESULT

The existing `CanonicalHistoryInputProvider` calls the shared synthetic canonical builder and produces GRU/XGBoost views from identical canonical information. Selected preprocessors are hash-verified, marked train-fit, and transform-only. Full-history and explicitly truncated inputs yield equal prepared model inputs.

## 11. TASK MODEL ROUTING RESULT

All task loaders are resolved solely from the selected manifest. No latest-file discovery, runtime family choice, model switcher, search, or fallback occurs.

## 12. ICU POSTPROCESS RESULT

The integrated path uses `remaining_icu_hours_from_log_prediction` through `CanonicalICUTimeServingPostprocessor`; its output is compared directly with the shared utility.

## 13. SUPPORT CALIBRATION / THRESHOLD RESULT

The frozen synthetic calibrator performs transform-only probability handling and the manifest-bound threshold is usable. Calibrator fit calls remain zero; the public v1 schema correctly omits an alert/class field.

## 14. EXPLANATION ROUTING RESULT

GRU tasks route to Integrated Gradients and the XGBoost task routes to TreeSHAP through the existing explanation router. Context is bound to task, model object/hash, manifest, feature version, support calibration identity, and cutoff. Existing adapter integration suites exercise the actual IG and TreeSHAP implementations.

## 15. RECOVERY RECONSTRUCTION RESULT

The integrated path uses `RecoveryServingPostprocessor`, verified current SOFA, typed original-unit deltas, exact horizon order, and the shared reconstruction transform. Results prove independent +24/+48 reconstruction with no chaining or double inverse transform.

## 16. PREDICTION-SCHEMA RESULT

Every integrated output passes both the Phase-5 validator and Pydantic `PredictionResponse`. Missing mandatory fields, wrong schema version, invalid routing, and unreconciled quality counts fail closed.

## 17. API == DIRECT PIPELINE RESULT

For the same request, the JSON-normalized direct pipeline response equals the HTTP `/predict` response exactly across scientific values, data quality, task identities, hashes, versions, explanations, and replay mode.

## 18. DASHBOARD == API CONTRACT RESULT

The dashboard controller consumes the actual in-process FastAPI response through a dashboard protocol adapter. Recovery, ICU time, support probability, quality, explanations, and metadata match API values without local recomputation.

## 19. REPLAY THREE-CUTOFF RESULT

Three legal cutoffs are selected in order and each produces one exact API request. The test deliberately permits identical predictions while asserting correct request timestamps and cutoff-safe timeline exposure.

## 20. FUTURE-ROW PERTURBATION RESULT

Deleting, changing, or adding rows after t leaves the canonical prepared inputs, direct prediction, API-compatible payload, dashboard snapshot, and t data-quality values unchanged.

## 21. TRAINING-SERVING EQUIVALENCE RESULT

Full stored history with internal truncation and explicitly cutoff-truncated storage produce identical task inputs and predictions. Existing Phase-7 parity, strict lookback, boundary, mask, padding, and TSLO tests are rerun in the critical upstream suite.

## 22. MODEL HASH MISMATCH TEST

Mutating recovery model bytes produces an explicit hash mismatch before any loader, preprocessor, predictor, calibrator, or explanation call.

## 23. FEATURE-SCHEMA MISMATCH TEST

Feature-version metadata drift and same-width feature-order mismatch fail explicitly through the existing artifact and canonical provider contracts.

## 24. PREPROCESSOR MISMATCH TEST

Preprocessor byte drift, identity drift, and contradictory `fit_partition` provenance fail before loader/inference calls.

## 25. SPLIT / LABEL MISMATCH TEST

Per-task split mismatch and model/manifest label-version mismatch are rejected before inference.

## 26. MANIFEST MISMATCH TEST

Outer manifest hash drift, internal content-hash drift, stale metadata, unknown/missing tasks, and unsupported families fail closed. Exact manifest references are mandatory.

## 27. CALIBRATOR / THRESHOLD MISMATCH TEST

Calibrator byte/hash/model-binding drift and threshold byte/hash/calibrator/model-binding drift are rejected before a response can be generated.

## 28. EXPLANATION MISMATCH TEST

Wrong family/method, model hash, task, feature version, manifest, and cutoff bindings are rejected by the existing router and its rerun suites. IG baseline/config and TreeSHAP flattened mapping incompatibilities remain covered by their dedicated integration tests.

## 29. HASH-DRIFT TEST

A temporary registered artifact is mutated after indexing; the existing registry audit reports its artifact ID in `hash_mismatches`. Drift is never adopted silently.

## 30. REGISTRY / LINEAGE AUDIT

The read-only repository audit reports no missing registered artifacts, hash mismatches, scientific orphan artifacts, broken parents, unknown producers, or incompatible dependencies. It also transparently reports 26 non-scientific/unregistered historical files and seven legacy-incomplete result artifacts; Phase 14 did not fabricate producers or silently register them. Existing forward/reverse traceability and calibration lineage tests are reused; no parallel registry was created.

## 31. SYNTHETIC-SCOPE ISOLATION

The integration manifest is explicitly non-scientific, non-serving, and non-G3-authorizing. Synthetic runs are tested as excluded from final-scientific queries, and restricted row-identity metadata is rejected.

## 32. FINAL-TEST NONACCESS EVIDENCE

No final-test entrypoint or loader is imported or invoked. Registry and G3 audit outputs report `final_test_accessed = false`; the effective state remains `NEVER_OPENED`.

## 33. G3 AUDIT STATUS

The existing real G3 audit was called directly in read-only, non-authorizing mode. It remains `BLOCKED`, including missing real split/selected-manifest prerequisites, and no `g3_freeze.json` marker was created.

## 34. NO-TRAINING / NO-SELECTION EVIDENCE

Serving preprocessor operations are transform-only, support calibrator fit count is zero, and existing pipeline guards prove training, threshold-search, and selection entrypoints are not invoked. The selected manifest remains the only family authority.

## 35. TESTS ADDED

Added 23 Phase-14 tests covering full-stack equality, three cutoffs, truncation/future perturbation, isolation, canonical transforms, dashboard binding, clean-process execution, byte/metadata mutation attacks, registry drift, synthetic exclusion, restricted metadata, and non-authorizing G3 state.

## 36. EXACT COMMANDS EXECUTED

Executed `PYTHONPATH=src:tests:. python3 -m pytest -q tests/test_integration.py tests/test_artifact_compatibility_e2e.py tests/test_serving_lineage_e2e.py`; the explicit Phase 6–13 and registry/G3 upstream pytest selection; `PYTHONPATH=src:tests:. python3 -m experiments.registry_cli audit --root .`; direct read-only `audit_g3(Path('.'), scope='real')`; `PYTHONPATH=src:tests:. python3 -m compileall -q api dashboard src tests`; and `PYTHONPATH=src:tests:. python3 -m pytest -q`.

## 37. TEST RESULTS

Phase-14 focused tests passed 23/23. The explicit Phase 6–13 critical regression set passed 193 tests with 2 skips. Compilation passed. Registry critical-integrity fields were empty, final-test access was false, and no G3 marker existed. The non-authorizing real G3 audit remained blocked with eight blockers. The complete repository suite passed 887 tests with 2 skips. No performance metric was an acceptance criterion.

## 38. REAL END-TO-END INTEGRATION STATUS

`BLOCKED — REAL SELECTED/FROZEN ARTIFACTS REQUIRED`. The real selected manifest, exact selected task artifacts, real frozen preprocessors, support calibration chain, verified current-SOFA handoff, approved real timeline, and complete real registry lineage are unavailable.

## 39. BLOCKED / UNLOCKED PARAMETERS

The Phase-14 integration fixture and clean-process harness are synthetic engineering choices only. Real feature schema, preprocessors, explanation configuration/baseline, current-SOFA source, and approved demo timeline remain blocked upstream. No scientific value was guessed.

## 40. VEDANT REVIEW NOTES

Review selected-manifest authority, hashes, feature/split/label compatibility, train-fit preprocessing, canonical ICU transform, calibration lineage, exact API equality, zero training/search, registry integrity, and final-test nonaccess.

## 41. SANSKRUTI REVIEW NOTES

Review canonical builder reuse, cutoff/history semantics, feature order/units, mask/TSLO/padding behavior, current SOFA compatibility, independent recovery reconstruction, XGBoost/GRU information parity, and dashboard quality terminology.

## 42. DEFINITION-OF-DONE CHECKLIST

Synthetic framework requirements are complete: existing artifact/registry layers reused, raw-to-dashboard integration tested, mixed-family routing proven, hostile mutations fail closed, clean process passes, no notebook/manual/absolute path dependency exists, test/G3 remain untouched, and review documentation exists. Phase 15 and Phase 16 are intentionally absent.

## 43. READINESS FOR PULKIT PHASE 15

The validated synthetic full-stack harness, mutation suite, equality evidence, read-only lineage/governance results, and real blocker inventory are ready for Phase 15 packaging and environment-freeze work. Phase 15 was not implemented.
