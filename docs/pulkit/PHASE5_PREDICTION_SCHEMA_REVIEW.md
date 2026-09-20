# Pulkit Phase 5 prediction-schema review

## Decision

`PREDICTION SCHEMA CONTRACT: IMPLEMENTED — REVIEW PENDING`

`REAL MODEL METADATA POPULATION: BLOCKED — REAL SELECTED MANIFEST REQUIRED`

This phase defines and validates the interface only. It performs no feature construction, model loading, inference, recovery reconstruction, ICU-time postprocessing, support calibration, thresholding, or attribution computation.

The machine-readable contract is `configs/prediction_schema_v1.json`, using the repository's explicit-contract approach without introducing a JSON Schema dependency. Its exact SHA-256 is `db99a539c41a1e86a4bd3b9f6317aaf73a8481f8396854e6749cbb80869fc265`.

## Request contract

The public request contains exactly:

- `stay_id`: the repository-compatible non-empty integer or string identifier.
- `prediction_time`: a timezone-aware ISO-8601 historical cutoff.

Legal-grid validation remains delegated to `data.timestamps`; Phase 5 does not duplicate the cutoff generator. The request has no subject identifier, model selection, labels, future outcomes, tensors, or cached-result identity.

## Response contract

The response requires:

- `schema_version = prediction_schema_v1`
- the same `prediction_time` used by the request
- `mode = RETROSPECTIVE_SEQUENTIAL_REPLAY`
- independent task entries under `model_versions`
- independent 24-hour and 48-hour recovery deltas and reconstructed SOFA fields
- remaining current-ICU-stay time in hours
- calibrated 24-hour New Organ-Support Initiation probability
- task-specific model-attribution payloads
- explicit mask/missingness counts under `data_quality`
- feature, label, split, and selected-manifest versions

Task model identities preserve family, model version, artifact SHA-256, and preprocessor SHA-256. The support task additionally preserves calibrator SHA-256. Only `xgboost` and `gru` are valid serving families. Attribution routing reuses the existing selection contract: XGBoost uses `tree_shap`; GRU uses `integrated_gradients`.

## Scientific semantics

Recovery deltas are independent:

- `delta_24h` means SOFA(t+24h) minus SOFA(t).
- `delta_48h` means SOFA(t+48h) minus SOFA(t).
- The 48-hour reconstruction never chains through the 24-hour output.

`icu_stay_time_hours` means remaining time until the current ICU stay ends. It is finite, nonnegative, and expressed canonically in hours.

`organ_support_probability_calibrated` is bounded to [0,1] and refers only to calibrated 24-hour probability of New Organ-Support Initiation. Phase 5 does not implement calibration.

The response contains no patient-level uncertainty interval or regression certainty field. `data_quality` reports explicit observation, padding, and missingness counts and is not an estimate of prediction correctness.

## Synthetic fixtures

All committed examples are marked:

`SYNTHETIC_MOCK_CONTRACT_OUTPUT_NOT_MODEL_PERFORMANCE_NOT_CLINICAL_DATA`

Synthetic hashes use `SYNTHETIC_HASH_NOT_REAL` and are accepted only when the validator is explicitly invoked in synthetic-fixture mode. The same payload is rejected in real mode. No restricted row or trained-model result is present.

The mixed-family fixture proves the following unambiguously:

- recovery: GRU
- ICU stay time: XGBoost
- organ support: GRU

Tests also cover every other XGBoost/GRU combination.

## Unlocked schema parameters

The following are proposed engineering shapes rather than falsely frozen scientific decisions:

- Validation format: `project_explicit_contract_v1`
- Unknown/additional-property handling: reject in the v1 proposal
- Attribution item shape: `{feature_name, attribution}` pending Phase-9/10 adapter review
- Data-quality sub-shape: explicit counts pending Vedant/Sanskruti review
- Artifact-version metadata nesting
- Support threshold/binary-alert representation, which is excluded from v1 until reviewed

A semantic revision after approval requires a schema-version bump and compatibility review.

## Vedant review package

Vedant should verify timestamp compatibility, independent recovery horizons, mixed-family task identities, selected-manifest field compatibility, support calibrator metadata, future threshold compatibility, hours-based ICU output, absence of evaluation/test leakage, and absence of unjustified uncertainty fields.

## Sanskruti review package

Sanskruti should verify SOFA terminology and units, current-ICU-stay wording, event-time serialization, mask/missingness terminology, lack of clinical overinterpretation, and compatibility with the future feature builder.

No reviewer approval is recorded by this implementation.

## Scope boundary

No `PredictionPipeline`, history truncation module, FastAPI application, dashboard, attribution algorithm, real selected-model manifest, or mock screenshot was created. Those remain later Pulkit work packages.
