# Sanskruti Rebuild Phase 11 — XGBoost Canonical Integration Review

## 1. Scope and outcome

Phase 11 is complete for the authorized synthetic adult cardiac benchmark.
It integrates XGBoost as a candidate family without running a scientific
candidate, consuming the 30-configuration search budget, inspecting model
performance, selecting a family, calibrating support probabilities, choosing
a threshold, producing production SHAP explanations, changing serving
routing, or accessing the sealed final-test partition.

Decision authority is `USER_DELEGATED_AI_PROJECT_DECISION`. The comparison is
now contractually “same scientific information, different representation and
model inductive bias.”

## 2. Phase-10 prerequisite

The Phase-10 audit passed before implementation. Authoritative identities are:

- split SHA-256: `3d6af5298f219f4608e5318ca09856642a1918a1154563ed0e8aec1785563cf6`;
- feature schema: `synthetic_feature_schema_v2`, F=21, SHA-256
  `f5de442cf4d0bf96356ef4ce4a15156541ee17a075b487cbe0c72926cee7e156`;
- feature preprocessor SHA-256:
  `c28bfdfbd83f08de96c30da94557a4da158ce28fedda254c1823be7b27fd6eb0`;
- recovery target scaler SHA-256:
  `ba2b45c4ed71a855852515dbb28df544156bbe4a6826719503282d1a0640de8f`;
- support class-weight SHA-256:
  `bf5228b119884c918b813fee4efdce28268c23290e74aecca03f37e25791e9fa`.

No old F=15 feature artifact, fixture split, MIMIC-era split, or smoke
preprocessor is used.

## 3. Common transformed input and family views

The one shared post-Phase-10 representation contains:

- standardized/imputed temporal values `[8,21]`;
- genuine-observation mask `[8,21]`;
- frozen-hour TSLO `[8,21]`;
- structural padding mask `[8]`;
- training-encoded static vector `[8]`;
- row identity, eligibility, and targets as non-predictor metadata.

The GRU view keeps these fields structured. The XGBoost view concatenates
five deterministic blocks in this exact order:

1. 168 row-major temporal values;
2. 168 row-major observation-mask indicators;
3. 168 row-major TSLO values;
4. 8 padding indicators;
5. 8 encoded statics.

Therefore `D = 8×21 + 8×21 + 8×21 + 8 + 8 = 520`. The model dtype is
float32. Statics occur exactly once and are not repeated through time.
Identifiers, timestamps, grid index, split, eligibility, and targets are not
in the matrix.

The GRU encoder was advanced from its historical blocked-static placeholder:
it can now explicitly require and append the frozen static vector, while TSLO
remains an explicit required setting in the final parity contract. Historical
smoke configurations remain historical and no GRU was retrained.

## 4. Flattened feature map

`artifacts/features/synthetic_xgb_flat_feature_map_v1.json` contains 520
ordered entries. Every entry records the flat index, interpretable name,
information type, bin index and interval when temporal, canonical channel,
unit/type, schema version, and schema hash. Its SHA-256 is:

`8c46f66dd32029ac5c02b6e99deafd8838de9bf7e499e81b8deb098de6970d5f`

Names such as `bin0__pao2__latest__value`,
`bin0__pao2__latest__mask`, and `bin0__pao2__latest__tslo` replace anonymous
`f0` identities and are ready for later TreeSHAP routing. Reordering, missing
blocks, extra columns, dimension drift, or renamed entries fail closed.

## 5. Complete final-data parity audit

The audit used all model-ready training and validation rows. It did not
construct a test matrix.

| Partition | Rows | Matrix | Row-key SHA-256 | Matrix-content SHA-256 |
|---|---:|---:|---|---|
| train | 8,626 | `[8626,520]` | `96bc55302da2c1d85599a0ddec8d68b0a7eb0bc5b5f1d4f1efdbbbf28af17993` | `06c07f28f194efe4f86cd9b3cf39d49c108907c746cd2b9172f784644a4647bd` |
| validation | 1,754 | `[1754,520]` | `ca601537fcf627651b1bbaca42021ed360c1b4572318587a7b887b82cd2a9ffe` | `8a0bd762230a404b686b7a1af3d491f9a94565b05f3d42ee9d3b243ed464a21b` |

Each row passed exact float32 flatten/unflatten equality for values, masks,
TSLO, padding, and statics. The six support-aware channels are represented in
all three temporal blocks, with no XGBoost-only aggregation. Row order is
bound by canonical `(subject_id, stay_id, prediction_time, grid_index)` keys,
not fuzzy positional matching.

## 6. Eligibility and target parity

Task matrices filter with the shared canonical masks and never place a mask
inside X.

| Task view | Train eligible | Validation eligible |
|---|---:|---:|
| recovery +24 h | 5,005 | 968 |
| recovery +48 h | 1,868 | 332 |
| ICU-like episode time | 8,626 | 1,754 |
| new organ support | 5,559 | 1,116 |

Recovery uses the same Phase-10 per-horizon scaler as GRU. ICU time remains
the Phase-8/10 `log1p` remaining-current-episode-hours target, with no new
target scaler. Support uses the frozen Phase-10 `scale_pos_weight =
1.2736196319018405`; it is not recomputed from validation or test labels.
Stay-balanced weights remain evaluation-only and are not silently introduced
as training sample weights.

## 7. Candidate architecture

Recovery is one candidate bundle with two regressors, +24 h and +48 h. Both
share one candidate ID, normalized configuration, configuration hash, seed,
feature lineage, and preprocessing lineage. Their eligible counts and best
iterations remain horizon-specific. Public output order is fixed at
`[DeltaSOFA24, DeltaSOFA48]`, inverse-transformed once to original delta units,
without clipping or horizon chaining.

ICU time is one `reg:squarederror` regressor. Its adapter reuses
`remaining_icu_hours_from_log_prediction()` for nonnegative-log handling and
`expm1`; no duplicate postprocessor exists.

Support is one `binary:logistic` classifier. The adapter returns
`predict_proba(X)[:,1]` exactly once as a raw uncalibrated probability. It has
no embedded calibrator and no threshold.

## 8. Runtime, early stopping, and serialization

- Python: 3.9.6, arm64 macOS, CPU.
- XGBoost: 2.1.4, matching the recorded development snapshot.
- Missing native runtime fixed by installing Homebrew `libomp` 23.1.1; the
  Python XGBoost package was not replaced or upgraded.
- Tree method: `hist`; device: `cpu`; `n_jobs=1`; explicit seed required.
- Maximum scientific estimators: 1,000.
- Frozen scientific early stopping: 50 rounds, configured on the estimator in
  accordance with XGBoost 2.1.4 rather than passed to `fit()`.
- Native XGBoost JSON is authoritative; arbitrary pickle is not used.

The final environment manager/lock remains a genuine project-level blocker
before scientific Phase-12 execution. The observed snapshot is not promoted
to a final lock by this phase.

## 9. Search-space and registry readiness

`configs/search_space_xgb_v2.json` freezes the retained ranges and Phase-11
entrypoint. Its SHA-256 is
`efbebec31ec012f6b5cacbf06948b9c90c15feb696db7c43f51d628fe1613f26`.
The validator rejects every out-of-range field instead of clamping.

The handoff manifest is
`artifacts/manifests/synthetic_phase11_xgb_input_manifest_v1.json`, SHA-256
`94e1f0a4772db362ad8e742a788b5de8bee3f9c96f47c3114e06a7cc7b2a80a8`.
It independently binds the data entrypoint
`src/data/xgb_canonical.py` and candidate-bundle entrypoint
`src/models/xgb_canonical.py` by SHA-256, in addition to every Phase-10,
schema, preprocessing, target, split, event-dictionary, parity, and search
contract named above.

No Phase-12 sampler ran. Candidate IDs 1–30 were not created or consumed.
There are zero scientific XGBoost registry rows. Bundle metadata captures all
required hashes, native model hashes, XGBoost version, best iterations, and
eligibility counts. A mapper emits these into Vedant's existing registry
fields, using `derived_feature_hash` for the flat-map hash and canonical JSON
notes for XGBoost-specific model details; no Sanskruti-only registry exists.

## 10. Engineering smoke evidence

Tiny in-memory contract fixtures exercised four native models: two recovery
horizons inside one bundle, one ICU regressor, and one support classifier.
Tests proved fit, early stopping, native JSON serialization, hash-first reload,
finite predictions, recovery `[N,2]` order and shared inverse transform, common
ICU postprocessing, and one-time uncalibrated support probability. Smoke
artifacts are temporary, explicitly `ENGINEERING_XGB_SMOKE`, do not count
toward the search budget, and produce no reportable performance metric.

Hostile tests reject changed schema/map/preprocessor/split/target/scaler/class
weight lineage, task mismatch, reordered features, and altered model bytes.

Verification results on the recorded environment:

- focused XGBoost/GRU/LSTM/TreeSHAP contract suite: 93 passed, 1 optional
  integration skipped;
- complete non-evidence regression used to refresh generated evidence: 1,221
  passed, 2 optional integrations skipped;
- final repository regression, including evidence audits: 1,229 passed, 2
  optional integrations skipped in 19.76 seconds (21.03 seconds wall time);
- Phase-10 and Phase-11 audit entrypoints: PASS;
- model and system evidence audits: PASS, with zero evidence orphans.

## 11. Test isolation and Phase-12 boundary

The data gateway allowlists only `train` and `validation`. `test`,
`sealed_test`, and `all` hard-fail before any matrix construction. Phase 10 did
not materialize a test file, and Phase 11 computed no test prediction, metric,
or plot.

Phase 12 receives the frozen split, preprocessor, flat map, target contract,
task matrix builders, model bundles, registry mapper, environment identity,
search-space hash, and validation-only early-stopping interface. Before the
first scientific candidate, Phase 12 must freeze its sampler/candidate
manifest and the final environment lock. It must then create exactly 30 shared
recovery configurations—not 60 horizon-specific configurations—and exactly 30
configurations for each other task.

Phase 12 or later was not implemented here.
