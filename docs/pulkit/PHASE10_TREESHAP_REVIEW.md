# Pulkit Phase 10 — TreeSHAP Review

## 1. PULKIT PHASE 10 SUMMARY

The fail-closed TreeSHAP adapter framework is implemented for selected XGBoost contexts. Structured synthetic feature mapping and adapter behavior are verified. Genuine TreeSHAP execution is blocked because SHAP is absent and XGBoost cannot load; real execution is additionally blocked by missing selected artifacts, loader, production flattening schema, and reviewed output/configuration policies. Phase 11 was not implemented.

## 2. REPOSITORY STATE BEFORE CHANGES

The pre-change suite passed with 747 tests and one Captum-dependent skip. Phase 8 routing and the Phase 9 IG framework existed. No TreeSHAP module, approved dependency manifest, real selected model manifest, or production XGBoost serving handoff existed.

## 3. XGBOOST ARTIFACT / LOADER AUDIT

`artifacts/models/selected_models_v1.json` is absent. `configs/validation_suite_v1.json` records a null XGBoost entrypoint. Only dry-run XGBoost search records and synthetic/mock serving material exist. No Sanskruti production loader, serialization contract, or task-specific selected artifact was found. No competing loader was created.

## 4. SHAP DEPENDENCY STATUS

`shap` raises `ModuleNotFoundError`. `xgboost` raises `XGBoostError` because its native runtime cannot load (the reported missing native dependency is `libomp.dylib`). No approved dependency/environment file exists, so neither package was installed ad hoc. `captum` also remains absent from Phase 9.

## 5. FILES CREATED

- `src/explainability/tree_shap.py`
- `configs/explainability/tree_shap_synthetic_development_v1.json`
- `tests/tree_shap_helpers.py`
- `tests/test_tree_shap.py`
- `tests/test_tree_shap_feature_mapping.py`
- `tests/test_tree_shap_targets.py`
- `tests/test_tree_shap_integration.py`
- `docs/pulkit/PHASE10_TREESHAP_REVIEW.md`

## 6. FILES MODIFIED

Phase-boundary assertions were advanced from “Phase 10 absent” to “Phase 11 absent” in `tests/test_explanation_router.py`, `tests/test_prediction_schema.py`, `tests/test_prediction_pipeline.py`, and `tests/test_integrated_gradients_integration.py`. The Phase 8 routing map was not changed.

## 7. TREESHAP ADAPTER ARCHITECTURE

`TreeShapAdapter` consumes one exact `ExplanationContext` and one `PreparedTreeInput`. It validates family, task, hash, cutoff, schema, model dimension, ordered flattened identities, flattening version/hash, and explicit output target before lazily importing `shap.TreeExplainer`. There is no surrogate or homemade attribution fallback.

## 8. PHASE-8 ROUTER INTEGRATION

The existing dependency-injected adapter registry accepts `TreeShapAdapter` through the existing protocol. The immutable mapping remains `xgboost -> tree_shap` and `gru -> integrated_gradients`. Router tests cover stale manifest and mixed-family behavior; no routing logic changed.

## 9. SELECTED-XGB / MODEL-HASH BINDING

The adapter uses the model object already carried by the validated context and compares the context artifact SHA-256 to that object. It never discovers or independently loads a model. A mismatch fails before TreeExplainer construction. Real selected-model binding remains blocked because the production manifest and artifact are absent.

## 10. PREDICTION-CUTOFF BINDING

Prediction cutoff, context input cutoff, and prepared flattened-input cutoff must match exactly. Mismatches fail before SHAP invocation.

## 11. FLATTENED INPUT CONTRACT

The adapter accepts only an already prepared `[1,D]` numeric row. It performs no history access, truncation, feature building, imputation, scaling, fitting, or flattening. The production Phase 7 flattening handoff is absent, so real mode fails closed.

## 12. FEATURE-NAME / FEATURE-SCHEMA MAPPING

Every synthetic dimension has an explicit `FlattenedFeatureIdentity`; names are not parsed. Names must be unique, count must equal `D`, order must equal the model contract, and schema/flattening identities must agree with the model. The production name list and hash are not frozen.

## 13. TIME-BIN REVERSE MAPPING

Synthetic temporal identities carry explicit bin index and relative interval. Validation enforces the eight canonical intervals from `(t-48,t-42]` through `(t-6,t]`. No wall-clock bin labels or heuristic name parsing are used.

## 14. STATIC / MASK / TSLO MAPPING

Static inputs have no temporal bin. Observation masks, TSLO, padding, and values retain separate channel types. Optional within-bin aggregation metadata is preserved rather than collapsed into a generic clinical variable.

## 15. RECOVERY 24H TARGET RESULT

Synthetic contracts explicitly select `delta_sofa_24h` at output index 0 and validate that name/index against the model. The result is independent and not derived from the 48-hour output. Real recovery artifact structure and output domain remain unresolved.

## 16. RECOVERY 48H TARGET RESULT

Synthetic contracts explicitly select `delta_sofa_48h` at output index 1. Multi-output SHAP arrays are normalized by an explicit index. No 24-hour chaining is performed. Real multi-model versus multi-output serialization remains unresolved.

## 17. ICU-TIME OUTPUT-DOMAIN STATUS

Synthetic tests label the explained value `SYNTHETIC_RAW_MODEL_OUTPUT_V1`. Real ICU-time TreeSHAP output domain is **UNLOCKED / BLOCKED**; the adapter does not silently choose log-space versus postprocessed hours.

## 18. ORGAN-SUPPORT OUTPUT-DOMAIN STATUS

Synthetic contracts can identify a raw support-model output. Real support raw-margin/probability semantics are **UNLOCKED / BLOCKED**. The isotonic calibrator and threshold are not substituted for the XGBoost model and are not explained.

## 19. TREESHAP CONFIG / BACKGROUND POLICY

The committed config is unmistakably synthetic: raw model output, `tree_path_dependent`, no background, and `1e-6` additivity tolerance. Its canonical hash is recorded in results. It is not a scientific freeze. Real perturbation mode, output domain, and leakage-safe background policy require review.

## 20. ATTRIBUTION DIMENSION RESULT

Executable tests require input dimension, ordered identity count, model input dimension, and returned SHAP dimension to match. Count/order/dimension attacks fail closed.

## 21. ADDITIVITY TEST

Adapter-contract tests verify `base_value + sum(signed SHAP) == explained raw output` at the configured tolerance and reject incoherent outputs. The genuine SHAP/XGBoost additivity test is present but skipped because the runtime is unavailable; therefore actual TreeSHAP additivity evidence is blocked, not claimed complete.

## 22. RAW SIGNED ATTRIBUTION RESULT

The full signed vector is returned in stable flattened order in both `items` and detailed metadata. Negative values remain negative. No normalization or top-k truncation occurs.

## 23. ABSOLUTE AGGREGATION RESULT

The pure helper sums absolute values across bins for the same explicit base-feature/channel/aggregation identity. Raw per-bin signed values remain available. No real global population summary or weighting rule was run.

## 24. REPRODUCIBILITY RESULT

Config hash, SHAP version, XGBoost version, target, schema, flattening identity, base value, and model output are recorded. Deterministic adapter behavior is tested. Genuine library reproducibility remains blocked with the genuine integration test.

## 25. MODEL-IMMUTABILITY / NO-TRAINING RESULT

Model bytes are snapshotted before and after explanation; mutation fails. Production code contains no fit, train, search, calibration fitting, or threshold optimization. The optional genuine integration test trains only a tiny non-scientific fixture in test code when dependencies become available.

## 26. FUTURE-DATA / LABEL NONACCESS

The adapter receives only the prepared row and metadata. It has no raw timeline or true-label parameter and does not access protected evaluation/test data.

## 27. NONCAUSAL TERMINOLOGY AUDIT

Existing descriptive-language lint remains active. Results expose model-output attribution without clinical causality, treatment directives, confidence claims, or a universal clinical interpretation of sign.

## 28. PHASE-5/8/9 INTEGRATION RESULT

The adapter reuses `AdapterExplanation`/`ExplanationResult`; it does not create an incompatible response type. Focused Phase 5, 6, 8, and 9 regressions passed. Phase 9 source remains present, but actual Captum execution is still dependency-blocked.

## 29. TESTS ADDED

Tests cover structured reverse mapping, all channel types, count/order/schema/version/hash attacks, invalid bins, finite input/output, explicit target/horizon selection, wrong families/task/hash/cutoff, exact-row reuse, additivity enforcement, immutability, full signed output, absolute aggregation, no top-k behavior, lazy dependency failure, source isolation, terminology, and an optional genuine XGBoost/TreeExplainer integration.

## 30. EXACT COMMANDS EXECUTED

```text
PYTHONPATH=src:tests python3 -m pytest -q
pytest tests/test_tree_shap.py tests/test_tree_shap_feature_mapping.py tests/test_tree_shap_targets.py tests/test_tree_shap_integration.py -q
PYTHONPATH=src:tests python3 -m pytest tests/test_tree_shap.py tests/test_tree_shap_feature_mapping.py tests/test_tree_shap_targets.py tests/test_tree_shap_integration.py -q
PYTHONPATH=src:tests python3 -m pytest tests/test_explanation_router.py tests/test_explanation_router_integration.py tests/test_integrated_gradients.py tests/test_integrated_gradients_integration.py tests/test_prediction_schema.py tests/test_prediction_pipeline.py tests/test_serving_artifacts.py -q
python3 -m compileall -q src tests
PYTHONPATH=src:tests python3 -m pytest -q
```

The bare `pytest` command failed because it is not on PATH; `python3 -m pytest` is the working repository command. A read-only `audit_g3(Path('.'), scope='real')` call was also executed; it returned `BLOCKED` with `test_data_accessed=False` and did not write or activate a freeze.

## 31. TEST RESULTS

Pre-change: 747 passed, 1 skipped. Final Phase 10 focused run: 26 passed, 1 skipped. Phase 5/8/9/serving regressions: 123 passed, 1 skipped. Compilation passed. Final suite: **773 passed, 2 skipped** in 4.95 seconds. The skipped Phase 10 test is the genuine SHAP/XGBoost integration; the Phase 9 skip is genuine Captum execution. No formatter, linter, or type-checker configuration exists in the repository, so none was invented or run.

## 32. REAL XGBOOST TREESHAP STATUS

**BLOCKED — REAL SELECTED XGBOOST ARTIFACT, SANSKRUTI LOADER, FROZEN FLATTENED FEATURE CONTRACT, REVIEWED OUTPUT/EXPLAINER POLICIES, SHAP DEPENDENCY, AND WORKING XGBOOST RUNTIME REQUIRED.**

## 33. BLOCKED / UNLOCKED PARAMETERS

Blocked inputs: real selected manifest/artifacts, loader/serialization interface, production flattened names/order/hash, SHAP, working XGBoost, and real Phase 7 row. Unlocked decisions: task output domains, perturbation mode, background policy/identity, recovery serialization, and global-summary population weighting.

## 34. SANSKRUTI REVIEW NOTES

Review must verify the exact loaded object/booster interface; no duplicate loader or training; same frozen flattened representation/order; complete dynamic/static/mask/TSLO/padding mapping; bin and recovery-horizon identities; task output domains; exact attribution dimension; and absence of feature engineering in the adapter.

## 35. VEDANT REVIEW NOTES

Review must verify selected-manifest authority, artifact hash, task/family agreement, identical cutoff, Phase 6 compatibility, separation of support model/calibrator/threshold, continued test isolation, G3 result metadata, and unchanged Phase 9 routing.

## 36. G3 EXPLANATION-ADAPTER STATUS

Both adapter source implementations are present and routing/hash/cutoff/schema synthetic evidence exists. The implementation prerequisite is **not fully satisfied** because genuine IG and TreeSHAP executions are dependency-blocked. Official G3 remains blocked by this and other recorded real-artifact/governance prerequisites; it was not activated.

## 37. DEFINITION-OF-DONE CHECKLIST

Framework items completed: repository audit; module/config/tests/docs; Phase 8 interface; XGBoost-only guard; hash/cutoff/schema/flattening/target checks; structured mapping; signed values; absolute aggregation; no top-k; no training/preprocessing/future/label access; response compatibility; regressions; no Phase 11 code. Not completed: governed dependencies, genuine synthetic library execution, real artifacts/loader/schema, real output/config/background freezes, and required owner reviews.

## 38. READINESS FOR PULKIT PHASE 11

The fail-closed TreeSHAP boundary and result metadata are ready for later API composition. Phase 11 should not represent TreeSHAP or IG as operational until their dependencies and real contracts clear. No FastAPI, endpoint, health, metadata, or dashboard code was added in Phase 10.
