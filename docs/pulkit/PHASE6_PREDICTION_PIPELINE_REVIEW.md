# Pulkit Phase 6 PredictionPipeline review

## Decision

`PREDICTIONPIPELINE FRAMEWORK: IMPLEMENTED AND TESTED`

`REAL FROZEN SERVING PIPELINE: BLOCKED — REAL SELECTED ARTIFACTS REQUIRED`

Phase 6 implements immutable-process orchestration, hash-first artifact resolution, compatibility checks, dependency injection, independent task routing, synthetic calibration/threshold binding, and Phase-5 response validation. It does not access raw history, construct features, fit preprocessing, run training or selection, compute real attributions, expose HTTP endpoints, or implement a dashboard.

## Repository audit

- `artifacts/models/selected_models_v1.json` is absent.
- Real search winners, selected checkpoints, support calibrator, and support threshold are absent.
- Existing recovery, ICU-time, and support GRU bundle loaders verify hashes and metadata, but the shared checkpoint validator currently requires `synthetic_smoke_test=true`; this is not a real-serving loader path.
- No real XGBoost serving loader or frozen serialization format exists.
- Phase-13 contains hash-aware calibrator/threshold loaders and exact model→calibrator→threshold compatibility semantics.
- Phase-16 provides the one artifact registry and lineage index reused by real-mode serving validation.

No real-looking artifact was created to bypass these blockers.

## Artifact resolver

`src/serving/artifacts.py` resolves only repository-relative paths. It verifies the selected-manifest file hash before JSON parsing, verifies the manifest's canonical content hash, validates scope and exact task membership, and then verifies every model, metadata, preprocessor, target-transform, calibrator, and threshold hash before calling any deserializing loader.

The manifest is the sole selection authority. There is no best-model search, latest-file discovery, family preference, fallback model, or hot swapping.

Real-mode artifacts additionally must be registered with their exact hashes, expected artifact types, scientific run type, and registered status in the existing Phase-16 artifact index. Serving never auto-registers artifacts.

## Compatibility rules

For every task, construction binds:

- task and serving family (`xgboost` or `gru` only)
- model version and artifact SHA-256
- task-specific feature and label versions
- shared split hash/version provenance
- train-fit preprocessor SHA-256
- model/preprocessor binding
- model output-domain contract
- explanation method required by the selected family
- target-transform artifact when declared required

A real recovery GRU is rejected without frozen target inverse-transform statistics.

Organ-support construction additionally binds the selected model to a calibrator, then binds the threshold to that exact calibrator and model. Serving exposes only `transform`; fitting and threshold search remain unavailable.

## Synthetic bundle

The committed manifest is:

`tests/fixtures/serving/selected_models_synthetic_phase6_v1.json`

Exact file SHA-256:

`a7832fce35122eeb59e384cef87896b4938c2e76e544633a6503f04d8ee9d68c`

It is marked `SYNTHETIC_NON_SCIENTIFIC_NON_SERVING_NON_G3_AUTHORIZING`, has `serving_ready=false`, and is not named `selected_models_v1.json`. Its model files explicitly state that they are adapter-test artifacts rather than checkpoints. It is intentionally unregistered and cannot authorize real serving.

The primary mixed-family fixture routes recovery to GRU, ICU time to XGBoost, and organ support to GRU. A second test constructs XGBoost/GRU/XGBoost routing. Outputs are deterministic interface values, not scientific predictions.

## Pipeline interfaces

`src/serving/interfaces.py` defines narrow protocols for task predictors, frozen preprocessors, prepared inputs, postprocessing, calibration, and attribution provision. These interfaces preserve task/family/version/hash identity without exposing training operations.

The prepared-input provider is temporary Phase-6 scaffolding. It is not a public tensor API or a second feature pipeline. Phase 7 must replace it with stored retrospective history, cutoff truncation, the existing feature builder, and exact train-fit preprocessing.

The attribution provider is also injected and synthetic. It receives task, selected family, and selected model hash. No Integrated Gradients, TreeSHAP, or family router is implemented here.

## Immutable-process policy

All compatibility checks occur during pipeline construction. A built bundle retains one resolved manifest/artifact state and does not live-reload or rediscover files. Mutation before construction is detected. Runtime hot-mutation behavior is intentionally unsupported; restart and rebuild against reviewed immutable artifacts is required.

## Test and data isolation

Hostile tests cover missing or mutated manifests, malformed paths/hashes, task membership, unsupported families, missing models, model/preprocessor drift, feature/label/split incompatibility, wrong-task artifacts, target-transform absence, calibrator/model drift, threshold/calibrator drift, explanation mismatch, synthetic-to-real confusion, unregistered real artifacts, and incompatible loader identities.

Training and threshold-search entrypoints are patched to fail if called; pipeline tests make zero calls. Calibrator fitting remains zero. No final-test loader, label, prediction, or metric is accessed.

## Vedant review checklist

Vedant should verify that the selected manifest is the sole selection authority; all task/family routing matches Phase 12; hashes precede deserialization; feature, label, split, and preprocessor identities are preserved; support calibration lineage matches Phase 13; no selection/training/test access occurs; per-task identities reach the Phase-5 response; and synthetic artifacts cannot authorize real serving.

## Sanskruti compatibility notes

Phase 7 will need an exact feature-builder interface that emits the frozen canonical information for both families. Sanskruti should review feature-schema version binding, XGBoost loader/serialization requirements, task output domains, and the absence of duplicate binning, mask, TSLO, imputation, or normalization logic in Phase 6.

## Honest blockers

- `BLOCKED — REAL SELECTED_MODELS_V1 REQUIRED`
- `BLOCKED — REAL SELECTED MODEL ARTIFACTS REQUIRED`
- `BLOCKED — REAL SUPPORT CALIBRATOR REQUIRED`
- `BLOCKED — REAL SUPPORT THRESHOLD REQUIRED`
- `BLOCKED — REAL XGBOOST SERVING LOADER REQUIRED`
- `BLOCKED — REAL GRU SERVING CHECKPOINT PATH REQUIRED`
- `BLOCKED — PULKIT EXPLANATION ADAPTER G3 DEPENDENCY`
- `DEFERRED — RAW-HISTORY TRUNCATION TO PHASE 7`
- `DEFERRED — PREPROCESSING EQUIVALENCE TO PHASE 7`
- `DEFERRED — EXPLANATION ROUTER TO PHASE 8`
- `DEFERRED — INTEGRATED GRADIENTS TO PHASE 9`
- `DEFERRED — TREESHAP TO PHASE 10`
- `DEFERRED — FASTAPI TO PHASE 11`

No Phase-7 or later implementation was added.
