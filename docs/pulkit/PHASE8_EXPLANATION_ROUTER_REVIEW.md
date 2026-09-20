# Pulkit Phase 8 — Model / Explanation Router Review

## 1. Pulkit Phase 8 summary

The per-task explanation router is implemented and synthetically verified. It performs lineage validation and adapter dispatch only; no attribution algorithm is implemented.

## 2. Repository state before changes

The Phase-6 bundle already hash-validated selected artifacts and family/method metadata. Phase 7 supplied cutoff-safe prepared input. The pre-change suite passed 709 tests.

## 3. Existing explainability audit

No `src/explainability` implementation existed. Phase 5 already froze the public routing names and minimal item shape. Phase 6 already rejected manifest family/method disagreement. No Captum, SHAP, IG, or TreeExplainer implementation was present.

## 4. Files created

- `src/explainability/__init__.py`
- `src/explainability/router.py`
- `tests/test_explanation_router.py`
- `tests/test_explanation_router_integration.py`
- `tests/fixtures/explainability/synthetic_router_phase8_v1.json`
- this review record

## 5. Files modified

The serving pipeline now constructs a router from the already validated bundle and an explicit adapter registry. Serving test helpers and Phase-5/6/7 tests were advanced to the Phase-8 boundary.

## 6. Selected-manifest status

The synthetic Phase-6 manifest is the sole fixture routing authority. `artifacts/models/selected_models_v1.json` remains absent; no fake real manifest was created.

## 7. Routing policy

The immutable policy is `xgboost -> tree_shap` and `gru -> integrated_gradients`. Naive and LSTM are rejected. Manifest disagreement is rejected, never corrected.

## 8. Common adapter contract

Adapters declare exact method, exactly one supported serving family, and synthetic/real scope. They receive one immutable context containing selected model object, prepared input, prediction output, task, cutoff, manifest identity, feature identity, and optional unresolved target metadata.

## 9. Model-hash binding

The context hash, selected identity hash, already-loaded predictor hash, and predictor object identity must all match before adapter invocation. The router does not load another model.

## 10. Prediction-cutoff binding

Prediction and prepared-input cutoff fields must match exactly and be timezone-aware ISO-8601. Mismatches fail before adapter invocation.

## 11. Task binding

Request task, prepared-input task, selected task, and loaded predictor task must agree. Unknown or mismatched tasks fail closed.

## 12. Feature-schema binding

The selected feature version, context version, and feature-metadata version must agree. The current manifest has no feature-schema hash field, so a claimed hash is rejected rather than accepted without authority.

## 13. Mixed-family routing result

Both mixed-family patterns are covered. Each task independently routes through the adapter required by its selected family.

## 14. Wrong-family hostile tests

XGBoost+IG, GRU+TreeSHAP, Naive, LSTM, wrong adapter method, wrong adapter family, wrong model family, missing method, and missing adapter all fail closed.

## 15. Wrong-hash / cutoff tests

Wrong model hash, wrong model object, stale manifest version/hash, wrong cutoff, wrong task, and mutated artifact all fail before an adapter call.

## 16. Phase-7 input-reuse test

The pipeline performs one canonical build per task. The router receives the identical prepared object already passed to prediction and never invokes the feature builder or predictor.

## 17. Phase-5 response-schema result

Internal results carry rich lineage, while the public response retains the existing `family`, `explanation_method`, and `items` shape. The routed synthetic response validates under `prediction_schema_v1`.

## 18. Noncausal terminology audit

A terminology lint rejects causal deterioration/recovery claims and treatment directives. The synthetic fixture explicitly describes attribution as non-causal and non-scientific.

## 19. Synthetic versus real explanation status

Mock adapters/results are explicitly synthetic. Synthetic adapters cannot initialize a real-scope router. Real attributions are unavailable.

## 20. G3 explanation dependency status

Router: complete. Integrated Gradients: pending Phase 9. TreeSHAP: pending Phase 10. **G3 explanation adapter dependency remains blocked.** No G3 state was activated or modified.

## 21. Tests added

Tests cover canonical routes, two mixed-family configurations, every required lineage mismatch, registry capability checks, input reuse, raw-history exclusion, schema preservation, noncausal language, deterministic routing, response compatibility, artifact mutation, and absence of real attribution computation.

## 22. Exact commands executed

```text
PYTHONPATH=src:tests python3 -m pytest -q
PYTHONPATH=src:tests python3 -m pytest -q tests/test_explanation_router.py tests/test_explanation_router_integration.py tests/test_prediction_pipeline.py tests/test_preprocessor_serving.py
find src/explainability ...
rg -n "explanation_method|..." src tests configs docs
rg -n "G3.*explanation|..." docs configs artifacts src tests
rg -n -i "captum|treeexplainer|import shap|integratedgradients" src/explainability src/serving
PYTHONPATH=src:tests python3 -m compileall -q src tests
```

## 23. Test results

Pre-change: 709 passed. Focused and final full-suite results are recorded in the completion response after final verification. No configured formatter, linter, or type checker was found and none is claimed.

## 24. Blocked / unlocked parameters

Blocked: real selected manifest, real Phase-7 input path, real IG adapter, real TreeSHAP adapter, and reviewed adapter evidence. Unlocked: recovery horizon/target presentation, ICU-time explanation output target, support explanation output target, and final explanation-item schema.

## 25. Vedant review notes

Verify sole manifest authority, task independence, artifact/object/hash binding, exact cutoff, no preprocessing path, Naive/LSTM rejection, stale-manifest failure, support calibration identity separation, Phase-5 compatibility, and unchanged G3 blocker.

## 26. Sanskruti compatibility notes

Verify feature-schema identity preservation, future flattened/timestep metadata compatibility, no feature-name derivation, unchanged unit/input semantics, and descriptive non-causal language.

## 27. Definition-of-done checklist

The Phase-8 framework checklist is complete: validated selected routes, per-task mixed-family dispatch, exact lineage/cutoff/task bindings, capability-checked adapters, prepared-input reuse, schema preservation, synthetic isolation, hostile tests, and no Phase-9+ implementation. Real readiness items remain blocked.

## 28. Readiness for Pulkit Phase 9

The router contract is ready to accept a genuine reviewed GRU Integrated Gradients adapter. Phase 9 must freeze target selection and baseline behavior; Phase 8 did not choose them.
