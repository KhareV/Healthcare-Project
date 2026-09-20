# Pulkit Phase 13 Dashboard and Replay Review

## 1. PULKIT PHASE 13 SUMMARY

Implemented the product-facing dashboard framework, retrospective sequential replay, and factual data-quality display. Work is deliberately limited to Phase 13; no Phase 14 integration suite or Phase 16 release evidence was created.

## 2. REPOSITORY STATE BEFORE CHANGES

The Phase 11 API and Phase 12 recovery transform existed. No dashboard directory, final UI stack, dependency manifest, approved real replay catalog, or real selected serving bundle was available. The pre-change suite passed 837 tests with 2 skips.

## 3. DASHBOARD FRAMEWORK AUDIT

FastAPI, Starlette, httpx, and Jinja2 were installed; Streamlit and Plotly were not. The selected stack is server-rendered FastAPI plus dependency-free HTML/CSS. This is the smallest repository-compatible choice and adds no package freeze.

## 4. FILES CREATED

Created `dashboard/__init__.py`, `dashboard/app.py`, `dashboard/api_client.py`, `dashboard/catalog.py`, `dashboard/replay.py`, `dashboard/render.py`, `dashboard/view_models.py`, the explicitly synthetic Phase-13 fixture, five focused dashboard test modules, and `tests/dashboard_helpers.py`.

## 5. FILES MODIFIED

Updated six older phase-boundary assertions so they recognize the now-authorized Phase-13 dashboard. No scientific implementation, API contract, artifact, or environment file was changed.

## 6. DASHBOARD ARCHITECTURE

The flow is `HTML dashboard -> narrow HTTP client -> FastAPI endpoints -> existing PredictionPipeline`. The catalog supplies only approved demo-safe stays, legal cutoff choices, cutoff-safe visualization rows, and synthetic current-SOFA states. One validated replay snapshot feeds every panel.

## 7. REQUIRED BANNER RESULT

Every render, including unavailable/error renders, prominently includes `RETROSPECTIVE SEQUENTIAL REPLAY - NOT REAL-TIME CLINICAL PREDICTION`. Synthetic operation also displays `SYNTHETIC DEMO / NON-SCIENTIFIC`.

## 8. PATIENT/STAY SELECTOR

The selector exposes only catalog-approved `SYNTHETIC_*` stay identifiers. It contains no subject names, database enumeration, or restricted MIMIC content.

## 9. LEGAL REPLAY CUTOFF CONTRACT

The catalog validates supplied cutoffs against the canonical timestamp generator and exposes only that validated subset. The UI uses a dropdown rather than an arbitrary timestamp input.

## 10. REPLAY CONTROLLER

`ReplayController` owns selection, exact API orchestration, response validation, identity checks, cutoff-safe timeline binding, and state clearing. It owns no inference, feature, target, calibration, threshold-selection, or attribution logic.

## 11. API CALL PER CUTOFF EVIDENCE

Tests verify exact `predict(stay_id, prediction_time)` calls for t1, t2, t3, and backward replay. Each selection performs one prediction call; panels do not issue separate requests.

## 12. HISTORICAL TIMELINE CUTOFF RESULT

Timeline rows are filtered by their explicit availability timestamp with `event_time <= selected cutoff`. Rows after the cutoff are absent even when retained in the synthetic source fixture.

## 13. FORWARD/BACKWARD REPLAY TEST

The three fixture cutoffs expose progressively eligible rows; returning to t1 restores the t1-only timeline. Tests intentionally do not require numerical predictions to change.

## 14. STAY-ISOLATION TEST

Changing stays clears cutoff, snapshot, explanations, prediction metadata, and timeline state. Cross-stay timeline labels are tested for disjointness.

## 15. RECOVERY PANEL

The panel uses Phase-12 `recovery_display_data` and displays API-provided current SOFA, raw independent deltas, and reconstructed +24/+48 SOFA values. It performs no reconstruction or clipping.

## 16. CURRENT VS PREDICTED SOFA SEMANTICS

Current SOFA is labeled `Observed current SOFA at cutoff`; both horizons are explicitly labeled `Predicted`. The two raw delta labels remain separately anchored at +24h and +48h, without chained-forecast claims or future actuals.

## 17. ICU-TIME PANEL / TERMINOLOGY

The title is `Remaining ICU stay time` and the definition is `Remaining time until current ICU stay ends`. The exact API hours value is retained; there is no patient interval, outtime, total LOS, or hospital-discharge terminology.

## 18. ORGAN-SUPPORT INITIATION PANEL

The panel is titled `New Organ-Support Initiation Risk` and defines the next-24-hour eligible OFF-to-ON initiation endpoint. Listed support types are qualifying vasopressor support and invasive mechanical ventilation only.

## 19. CALIBRATED PROBABILITY / THRESHOLD HANDLING

The calibrated probability is read directly from the prediction response. The frozen validation threshold is read from `/model-metadata`. Because the Phase-5 response has no alert/class field, the UI does not invent or locally calculate one.

## 20. EXPLANATION PANEL

All three API-returned task explanation payloads are rendered with signed numeric attributions and non-causal wording. The dashboard imports neither SHAP nor Captum and gives no treatment recommendations.

## 21. SELECTED-FAMILY EXPLANATION ROUTING DISPLAY

Per-task metadata is checked against the prediction response and frozen family-method map. Mixed fixture routing renders recovery GRU/Integrated Gradients, ICU-time XGBoost/TreeSHAP, and support GRU/Integrated Gradients.

## 22. DATA-QUALITY PANEL

The panel shows exact total/observed/padding bin counts and total/observed/missing feature-value counts. Pre-ICU padding and missing feature values have distinct labels.

## 23. FAKE-CONFIDENCE AUDIT

Rendered/static copy contains no confidence, certainty, reliability score, invented quality scalar, or patient prediction interval. Data availability is presented only as transparent counts/status.

## 24. MODEL-METADATA PANEL

Each task independently displays family, model version, full artifact hash, full preprocessor hash, feature version, label version, split hash, and explanation method. Shared manifest/split identity plus support calibrator and threshold hashes are displayed from API metadata.

## 25. MIXED-FAMILY DISPLAY TEST

Contract tests assert all three mixed-family labels and explanation methods independently. No global model-family assumption or model switcher exists.

## 26. ERROR-STATE HANDLING

HTTP 404/422/503 payloads, network failure, malformed success responses, unknown catalog stays, and incompatible identity are normalized to concise safe states. A failed selection clears panels before requesting, and no traceback or path is rendered.

## 27. FUTURE-DATA NONVISIBILITY EVIDENCE

Tests assert every visible event timestamp is at or before the selected cutoff, backward replay removes later rows, and perturbing a future row leaves the earlier timeline, current SOFA, data quality, and deterministic API response unchanged.

## 28. PRECOMPUTED-PREDICTION NONUSE

No prediction table or local prediction cache exists. Every explicit cutoff selection calls `POST /predict`; timeline data is visualization-only and never becomes model input.

## 29. SYNTHETIC VS REAL MODE STATUS

Synthetic mode requires explicit construction and displays a non-scientific banner. The module-level/default application is fail-closed with HTTP 503 and never falls back to synthetic output.

## 30. TESTS ADDED

Added structural tests for app construction, rendering, replay order, legal cutoffs, stay isolation, cutoff leakage, API call count, exact value binding, data-quality drift, terminology, mixed routing, errors, network behavior, and schema validation.

## 31. EXACT COMMANDS EXECUTED

Commands used: `PYTHONPATH=src:. python3 -m pytest -q tests/test_dashboard_contract.py tests/test_dashboard_replay.py tests/test_dashboard_data_quality.py tests/test_dashboard_terminology.py tests/test_dashboard_api_client.py`; upstream Phase 5/7/11/12 pytest selection; `PYTHONPATH=src:. python3 -m compileall -q api dashboard src tests`; and `PYTHONPATH=src:. python3 -m pytest -q`.

## 32. TEST RESULTS

Focused dashboard tests passed 27/27; relevant Phase 5/7/11/12 tests passed 105/105; compilation passed; and the full repository suite passed 864 tests with 2 skips. No UI-framework-specific runner exists for the selected server-rendered stack.

## 33. REAL MODEL-BACKED DASHBOARD STATUS

`BLOCKED — REAL SERVING ARTIFACTS REQUIRED`. No real `selected_models_v1.json`, final compatible checkpoints/preprocessors, real support calibration bundle, approved real replay catalog, or verified real cutoff-specific current SOFA handoff is present.

## 34. BLOCKED / UNLOCKED PRESENTATION PARAMETERS

Chosen unlocked values: server-rendered FastAPI/HTML, five-second HTTP timeout, no retries, exact hours, raw 0–1 probability, full attribution list, and full hashes. Real timeline variables, final formatting policy, and final demo packaging remain subject to reviewed handoff; no scientific default was introduced.

## 35. SANSKRUTI REVIEW NOTES

Review requested for exact clinical labels/units, banner, future-row hiding, current-versus-predicted SOFA, independent horizons, current-ICU meaning, initiation endpoint/support types, factual quality wording, non-causal explanations, absence of future outcomes, and demo-safe identity.

## 36. VEDANT REVIEW NOTES

Review requested for API-only serving, manifest authority, per-task identities, family/method compatibility, canonical cutoffs, preserved API/pipeline equivalence, unchanged calibration/threshold, no fake intervals or final-test access, and Phase-12 recovery binding.

## 37. PHASE-22 DEFERRED-EVIDENCE STATUS UPDATE

Phase 13 now supplies implementation and structural-test evidence for dashboard/replay/data quality. Older Phase-22 evidence is not rewritten, and synthetic fixtures/screens are not registered as final product evidence.

## 38. DEFINITION-OF-DONE CHECKLIST

Framework requirements are complete: API-only dashboard, exact banner, safe selector, legal cutoffs, one request/snapshot, cutoff-safe replay, all required panels, mixed-family metadata, safe errors, explicit synthetic scope, no scientific duplication, no final-test access, and focused tests/docs. Phase 14 and Phase 16 work remains intentionally undone.

## 39. REAL PHASE-13 STATUS

`DASHBOARD FRAMEWORK = COMPLETE`; `RETROSPECTIVE REPLAY = COMPLETE ON SYNTHETIC FIXTURE`; `REAL PHASE 13 = BLOCKED`. Sanskruti and Vedant reviews plus every real readiness dependency remain required before a scientific/final-ready claim.

## 40. READINESS FOR PULKIT PHASE 14

The dashboard application, client boundary, replay controller, legal-cutoff flow, pure view binding, panels, terminology audit, safe error behavior, and blocker inventory are ready for Phase 14 integration work. Phase 14 was not implemented here.
