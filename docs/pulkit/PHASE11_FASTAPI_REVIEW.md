# Pulkit Phase 11 — FastAPI Review

## 1. PULKIT PHASE 11 SUMMARY

The FastAPI framework, structured HTTP errors, bundle-derived metadata, and canonical ICU-time serving adapter are implemented. Synthetic `/predict` is verified against direct `PredictionPipeline` output. The production app is truthfully unavailable without real artifacts. Recovery reconstruction remains delegated and deferred to Phase 12.

## 2. REPOSITORY STATE BEFORE CHANGES

The Phase 5 response validator, Phase 6 pipeline/artifact bundle, Phase 7 cutoff-safe provider, Phase 8 router, and Phase 9/10 explanation adapters existed. No `api` package or concrete canonical ICU serving postprocessor existed. Pre-change suite: 773 passed, 2 dependency skips.

## 3. FASTAPI / PYDANTIC ENVIRONMENT AUDIT

Installed: FastAPI 0.128.8, Pydantic 2.13.5, Starlette 0.49.3, httpx 0.28.1, and Uvicorn 0.39.0. No project dependency/lock/environment file was found, so installed packages were used without an unrecorded installation or invented lock.

## 4. FILES CREATED

- `api/__init__.py`
- `api/main.py`
- `api/schemas.py`
- `src/serving/postprocessing.py`
- `tests/api_helpers.py`
- `tests/test_api_predict.py`
- `tests/test_api_errors.py`
- `tests/test_api_metadata.py`
- `tests/test_api_health.py`
- `tests/test_api_pipeline_equality.py`
- `tests/test_icu_time_serving_postprocess.py`
- `docs/pulkit/PHASE11_FASTAPI_REVIEW.md`

## 5. FILES MODIFIED

Phase-boundary assertions in prediction-schema, pipeline, router, and TreeSHAP tests now assert that Phase 11 exists while Phase 12/dashboard code remains absent. No scientific feature, model, calibration, selection, or explanation implementation was changed.

## 6. APPLICATION CONSTRUCTION / DEPENDENCY INJECTION

`create_app(pipeline=..., metadata_provider=...)` accepts an already constructed pipeline and does not load artifacts or patient data at import. Tests inject the validated synthetic pipeline. Module-level `app` has no bundle, reports not ready, and returns 503 for prediction/metadata rather than loading fixtures.

## 7. POST /predict CONTRACT

The public body is exactly `stay_id` plus `prediction_time`; extra fields are rejected under the Phase 5 proposed policy. The route validates bundle metadata, calls `pipeline.predict` once, revalidates the Phase 5 exchange, checks metadata coherence, and serializes the existing payload.

## 8. HEALTH ENDPOINT CONTRACT

`GET /health` returns minimal liveness, readiness, serving scope, and replay mode without patient access. `ready` is false for unavailable and synthetic scopes. Exact health payload semantics remain an **UNLOCKED ENGINEERING PARAMETER** pending review.

## 9. MODEL-METADATA ENDPOINT CONTRACT

`GET /model-metadata` reports the active bundle’s manifest version/hash, split version, per-task identities, explanation routes, and support calibration/threshold hashes and value. The standalone wrapper is an **UNLOCKED ENGINEERING PARAMETER**; its scientific identities are not independently invented.

## 10. REQUEST VALIDATION

Pydantic v2 strict request models reuse `validate_request`. Missing fields, malformed or timezone-naive timestamps, invalid identifiers, and extra fields produce structured 422 responses.

## 11. UNKNOWN-STAY 404 RESULT

`UnknownStayError` maps to 404 with a concise message. The actual Phase 7 provider and a hostile exception test prove no model call and no path/PHI disclosure.

## 12. ILLEGAL-TIMESTAMP 422 RESULT

`IllegalCutoffError` maps to 422. Tests exercise before-first, off-grid, and after-bound cutoffs through the actual Phase 7 timestamp/truncation path. Timestamp arithmetic is not duplicated in API code.

## 13. ARTIFACT/LOAD 503 RESULT

Serving artifact, hash, compatibility, pipeline, feature-contract, explanation-router, metadata, history-contract, schema, and serving-postprocess failures map to a safe 503. No fallback model or null-success response exists.

## 14. STRUCTURED ERROR MODEL

Errors use `{error: {code, message}}`. Known errors are typed rather than classified by message. Client responses omit exception text, paths, stack traces, patient data, and artifact internals. Unexpected errors return a generic 500 and log only the exception type.

## 15. PREDICTIONPIPELINE DELEGATION

The route performs one `pipeline.predict` call. History retrieval, cutoff validation, features, preprocessing, models, calibration, explanations, and data quality remain behind the pipeline.

## 16. API VS DIRECT PIPELINE EQUALITY

For the identical request and bundle, JSON-normalized HTTP output equals direct pipeline output exactly. Repeated requests are deterministic and do not reload models.

## 17. PREDICTION-SCHEMA / PYDANTIC PARITY

Pydantic request/response models reject extra fields and have the same top-level required fields as `prediction_schema_v1.json`. Successful responses also pass the canonical explicit validator. Probability, replay mode, finiteness, task maps, and per-task model versions are represented.

## 18. PER-TASK MODEL METADATA

Recovery, ICU-time, and organ-support each report family, model version, artifact hash, preprocessor hash, feature version, label version, split hash, and explanation method. Both valid mixed-family arrangements are tested. No generic-only model version exists.

## 19. FEATURE/LABEL/SPLIT/MANIFEST METADATA

All values come from the same validated bundle used by prediction. No disk scan or “latest” lookup occurs. Stale manifest version/hash and task identity attacks fail before prediction.

## 20. EXPLANATION-METHOD METADATA

Metadata checks the frozen family mapping: XGBoost to TreeSHAP and GRU to Integrated Gradients. A wrong-route attack returns 503 before model prediction. Routes never invoke attribution algorithms directly.

## 21. SUPPORT CALIBRATOR/THRESHOLD METADATA

The metadata endpoint distinguishes the support model from calibrator and threshold identities. It exposes calibrator hash, threshold hash/value, and comparator from the validated bundle. Phase 5 excludes alert/class from the v1 prediction response, so the API does not add it.

## 22. REPLAY-MODE RESULT

Prediction, health, and metadata contracts use exactly `RETROSPECTIVE_SEQUENTIAL_REPLAY`. OpenAPI describes a local retrospective research prototype, not live clinical inference.

## 23. ICU-TIME SHARED POSTPROCESS INTEGRATION

`CanonicalICUTimeServingPostprocessor` imports and calls `remaining_icu_hours_from_log_prediction`. It contains no duplicate clamp or exponential equation. Recovery is delegated unchanged to a supplied provider.

## 24. ICU SERVING/EVALUATION EQUALITY

Raw values -1, 0, log(2), and log(11) produce exactly the same results through the canonical function and serving adapter. A function spy proves the canonical function is invoked. NaN and infinities fail and become safe HTTP 503 responses.

## 25. ICU TERMINOLOGY AUDIT

API/source text uses remaining current ICU stay time in hours. Hospital-discharge, recovery-time, and survival-time wording is absent.

## 26. SUPPORT-PROBABILITY VALIDATION

The API passes through the pipeline’s calibrated field. Canonical response validation enforces finite `[0,1]`; invalid/raw out-of-range values fail with 503. No sigmoid, calibrator fit, or threshold search exists in API code.

## 27. NO-DUPLICATE-PREPROCESSING EVIDENCE

Source scans exclude feature builders, imputation, scaling, tensorization, model loading, training, and selection. Pipeline-owned transform calls remain the only preprocessing path, and calibrator fit calls remain zero.

## 28. NO-PRECOMPUTED-PREDICTION EVIDENCE

API and serving source scans exclude prediction tables/lookups/caches. End-to-end tests use stored synthetic history through the Phase 7 builder and pipeline.

## 29. FINAL-TEST NONACCESS

All tests use synthetic fixtures. Health and metadata invoke neither history nor models. No final-test loader, G3 marker, test metric, or protected result is accessed or exposed.

## 30. TESTS ADDED

Tests cover all endpoints, request errors, 404/422/503 mapping, metadata attacks, mixed families, schema/OpenAPI parity, exact direct equality, canonical ICU processing, finite probability, no reload/refit, future-row invariance, legal later cutoffs, nonleaking errors, and unavailable production defaults.

## 31. EXACT COMMANDS EXECUTED

```text
PYTHONPATH=src:tests python3 -m pytest -q
python3 -m compileall -q api src/serving
PYTHONPATH=src:tests python3 -m pytest tests/test_api_predict.py tests/test_api_errors.py tests/test_api_metadata.py tests/test_api_health.py tests/test_api_pipeline_equality.py tests/test_icu_time_serving_postprocess.py -q
PYTHONPATH=src:tests python3 -m pytest tests/test_serving_artifacts.py tests/test_prediction_pipeline.py tests/test_history_truncation.py tests/test_serving_feature_leakage.py tests/test_preprocessor_serving.py tests/test_training_serving_equivalence.py tests/test_explanation_router.py tests/test_explanation_router_integration.py tests/test_integrated_gradients.py tests/test_integrated_gradients_baseline.py tests/test_integrated_gradients_targets.py tests/test_integrated_gradients_integration.py tests/test_tree_shap.py tests/test_tree_shap_feature_mapping.py tests/test_tree_shap_targets.py tests/test_tree_shap_integration.py -q
python3 -m compileall -q api src tests
PYTHONPATH=src:tests python3 -m pytest -q
PYTHONPATH=src python3 -c "from pathlib import Path; from vedant_infra.g3 import audit_g3; print(audit_g3(Path('.'), scope='real'))"
```

## 32. TEST RESULTS

Pre-change: 773 passed, 2 skipped. Phase 11 focused: 35 passed. Upstream Phase 6–10 focused regression: 133 passed, 2 skipped. Compilation passed. Final repository suite: **808 passed, 2 skipped** in 4.92 seconds. The read-only G3 audit remained blocked, reported `test_data_accessed=False`, and created no freeze marker. Skips remain the previously documented missing Captum and SHAP/XGBoost runtimes. No formatter, linter, or type-checker configuration exists, so none was invented or run.

## 33. REAL SERVING-ARTIFACT STATUS

**BLOCKED — REAL SELECTED MANIFEST, MODELS, PREPROCESSORS, SUPPORT CALIBRATOR/THRESHOLD, PRODUCTION HISTORY/FEATURE PATH, AND OPERATIONAL EXPLANATION ADAPTERS REQUIRED.** The default production app does not substitute synthetic artifacts.

## 34. PHASE-12 RECOVERY RECONSTRUCTION STATUS

**DEFERRED — PHASE 12 REQUIRED.** API schemas and synthetic fixtures pass through the existing Phase 5-compatible reconstruction fields. No current-SOFA-plus-delta arithmetic or clipping was added to API or the new serving postprocessor.

## 35. BLOCKED / UNLOCKED PARAMETERS

Blocked: all real serving artifacts and real history composition, plus operational IG/TreeSHAP dependencies. Unlocked: health/readiness payload freeze, unavailable metadata wrapper policy, standalone metadata wrapper, request identifier logging, and Phase 5 extra-field policy final approval.

## 36. VEDANT REVIEW NOTES

Review the one-call pipeline delegation, canonical timestamp path, manifest authority, per-task identity/provenance, support chain, shared ICU function, no refitting, exact direct equality, test nonaccess, and synthetic/real separation.

## 37. SANSKRUTI COMPATIBILITY NOTES

Review absence of API feature construction, continued cutoff-aware canonical builder use, ICU meaning/units, data-quality semantics, prohibited-field absence, and end-to-end future-row invariance.

## 38. DEFINITION-OF-DONE CHECKLIST

Framework requirements are complete: factory, three endpoints, Phase 5 models, typed errors, single pipeline call, bundle metadata, ICU canonical transform, schema/equality tests, synthetic cutoff tests, no fitting/selection/explanation duplication, and regression suite. Real artifact readiness and Phase 12 recovery reconstruction are intentionally incomplete.

## 39. REAL API READINESS STATUS

**NOT READY.** `/health` reports `ready=false` without a real bundle. `/predict` and `/model-metadata` return 503. Real readiness cannot be claimed until every selected artifact and production dependency validates.

## 40. READINESS FOR PULKIT PHASE 12

The API framework and ICU transform boundary are ready for Phase 12. Phase 12 can implement recovery reconstruction behind the pipeline/postprocessor seam, preserve raw deltas, and add dashboard data transforms without changing HTTP orchestration.
