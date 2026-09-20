# Deep Codebase and Sanskruti-Handoff Integration Audit

**Repository:** `HealthCareProject`  
**Audit date:** 2026-09-20  
**Audit scope:** Entire committed working tree, with special attention to Vedant/Pulkit implementation, Sanskruti-owned handoffs, cross-owner interfaces, scientific governance, test evidence, and production integration readiness.  
**Authority used:** `docs/CODEX_PROJECT_CONTEXT_V1.md`, the Final Implementation Master Plan, the Final Work Division, then repository contracts/code/configuration. Instructions embedded in those source documents were treated as project specifications, not as user instructions.

---

## 1. Executive conclusion

The repository contains a broad, carefully fail-closed **contract and synthetic-development implementation** for Vedant's and Pulkit's work. It is unusually strong on temporal invariants, explicit eligibility masks, hash/version checks, validation-only governance, test isolation, artifact lineage, and adversarial tests. The checked development suite currently passes **906 tests with 2 dependency-gated skips**.

However, the statement that “both Vedant and Pulkit work has been completed” is only defensible if “completed” means **their synthetic/framework phases and interface scaffolding have been implemented**. It is not true under the repository's own definition of a completed real scientific or deployable system:

- real-data acceptance is `BLOCKED` and Phase 19 is not authorized;
- the selected-model manifest does not exist;
- G3 is blocked and final-test data has never been opened;
- the real support endpoint is not frozen;
- no real candidate search, calibration, threshold, or selected artifacts exist;
- no approved environment lock or Git commit provenance exists;
- the default API and dashboard intentionally fail closed;
- the repository has no MIMIC extraction, canonical feature construction, production `sofa_at(T)`, or Sanskruti XGBoost training implementation;
- several cross-boundary incompatibilities would prevent a newly delivered Sanskruti bundle from flowing end to end without code and contract work.

### Overall readiness verdict

| Layer | Verdict | Meaning |
|---|---|---|
| Frozen scientific semantics | **Strongly implemented at receiver/test level** | Core timing, target, weighting, selection, and isolation rules are encoded and adversarially tested. |
| Synthetic model/evaluation path | **Implemented and passing** | GRU/LSTM training, metrics, bootstrap, governance, serving, API, and dashboard contracts run on synthetic fixtures. |
| Sanskruti handoff receiver | **Substantial but not yet consumable** | A strict real-data adapter exists, but required real artifacts are absent and downstream tensor/model gaps remain. |
| Pulkit support handoff to Vedant | **Interface gap** | State-machine output and Phase-18 receiver expect different handoff shapes/names; no production bridge exists. |
| Real Vedant training/search/selection | **Blocked** | Real dataset loading/preprocessing, scientific checkpoints, frozen search details, real runs, and selection artifacts are absent. |
| Real Pulkit serving/explanations | **Blocked** | No real bundle/composition, production history/current-SOFA providers, or approved real IG/TreeSHAP configurations. |
| Final evaluation/release | **Blocked by design** | G3/G4, final-test plan, loaders, bootstrap settings, and reproducibility evidence are missing. |

The correct project description is therefore:

> The repository is a comprehensive, defensively tested integration skeleton and synthetic proof of the Vedant/Pulkit architecture. It is well positioned to reject a malformed Sanskruti handoff, but it is not yet able to accept a real handoff and complete training-to-serving without resolving the critical interface and production-composition gaps documented below.

---

## 2. Verification performed

This was a source-level audit, not a review of phase documents alone. The traversal covered:

- all Python modules under `src/`, `api/`, and `dashboard/`;
- all versioned configuration files;
- repository governance, acceptance, validation, reproducibility, and evidence artifacts;
- all test modules and the current collected test suite;
- cross-owner schemas and runtime call paths from canonical rows through API/dashboard output;
- explicit `BLOCKED`, `UNRESOLVED`, synthetic-only, and pending-review branches;
- repository/package state and runtime dependency evidence.

### Current measured repository evidence

| Measure | Observed result |
|---|---:|
| Files traversed, excluding caches | 533 |
| Python lines in application and tests | approximately 39,086 |
| Python test files | 177 |
| Tests executed | 908 |
| Tests passed | 906 |
| Tests skipped | 2 |
| Test failures | 0 |
| Evidence outputs indexed | 17 |
| Evidence orphan outputs | 0 |
| Evidence-system audit | `PASS` |
| Real-data acceptance | `BLOCKED` |
| Phase-19 preflight | `BLOCKED` |
| G3 audit | `BLOCKED`; test data not accessed |
| Phase-20 pretest audit | `BLOCKED`; test loader calls = 0 |
| Reproduction report | `BLOCKED` |

The skipped tests are the real-library adapter tests for Captum Integrated Gradients and SHAP/XGBoost. The current environment lacks the relevant optional dependency combination. This does not invalidate their contract tests, but it means the actual third-party attribution backends were not exercised in this run.

The evidence audit command reported:

```text
status: PASS
indexed_outputs: 17
orphans: 0
blocked_slots: 8
```

That `PASS` means the evidence index is internally consistent. It does **not** mean the eight blocked scientific slots are complete.

---

## 3. Repository architecture and ownership boundary

The intended end-to-end flow is:

```text
MIMIC-IV v2.2
  -> adult, chronological first ICU stay
  -> legal 24h+6k cutoffs (maximum 12)
  -> eight disjoint 6h bins in (t-48h, t]
  -> values + genuine-observation mask + TSLO + padding + statics
  -> independent recovery / ICU-time / support targets and eligibility masks
  -> subject-disjoint coarse temporal split
  -> XGBoost and GRU candidates; fixed LSTM sensitivity
  -> stay-balanced validation metrics and per-task family selection
  -> selected support isotonic calibration and F1 threshold
  -> G3 freeze
  -> one protected final-test execution and G4 evidence
  -> selected-family explanations
  -> PredictionPipeline -> FastAPI -> retrospective replay dashboard
```

### Ownership reflected in the codebase

- **Sanskruti:** extraction, first-stay cohort, clinical timeline/provenance, SOFA, canonical features, recovery and ICU-time labels, and primary XGBoost implementation.
- **Vedant:** timestamp grid, subject split, tensor/batch contracts, recurrent models/training, evaluation, bootstrap, search governance, per-task selection, support calibration/threshold, registry, and final-test governance.
- **Pulkit:** organ-support event state/label, selected-model serving contracts, family-aware explanations, FastAPI, replay dashboard, and product integration.

The repository correctly avoids pretending to implement most Sanskruti-owned clinical derivations. That is scientifically appropriate. The consequence is that this repository is a consumer awaiting a precise external bundle, not a standalone data-to-model package.

---

## 4. Detailed implementation inventory

### 4.1 Data, timestamps, split, and canonical representation

| Module | What is implemented | Quality and integration assessment |
|---|---|---|
| `src/data/timestamps.py` | Legal cutoff generator; `t = intime + 24h + 6h*k`; `k=0..11`; retain only `t <= outtime-6h`; independent 24h/48h/full-follow-up flags. | Strong. Boundary behavior and cap are directly encoded and tested. This is the authoritative receiver-side parity check for Sanskruti's structural index. |
| `src/data/split.py` | Subject-level coarse temporal split, subject isolation checks, deterministic CSV serialization, metadata/hash verification. | Strong. Correctly prevents stay-level leakage and binds split artifacts. Real split files are absent. |
| `src/data/schema.py` | `CanonicalExample`, feature-schema reference, target and eligibility objects, strict shape/dtype/identity validation, deterministic serialization, and GRU/XGBoost parity views. | Strong contract design. It preserves identical canonical information and only changes representation. Final feature names/order/dimension and TSLO sentinel remain upstream-bound. |
| `src/data/collate.py` | Typed `CanonicalBatch`, deterministic collation/loaders, explicit targets and masks, worker seeding. | Strong synthetic and recurrent-model boundary. |
| `src/data/dataset.py` | Canonical-to-tensor conversion and dataset wrapper for synthetic development. | Deliberately not a production imputer/normalizer. It cannot serve as the real training preprocessor without a reviewed implementation. |
| `src/data/support_labels.py` | Manual/synthetic `SupportLabelRecord` attachment with key/split/eligibility checks. | Useful interface test, but explicitly not the real support artifact loader. |
| `src/data/real_adapter.py` | Strict receiver for a hash-bound real canonical bundle. Revalidates timestamp regeneration, split isolation, field mapping, dimensions/order, padding/masks/TSLO, independent recovery formula, ICU formula, support lineage, and protected test handling. | One of the strongest integration components. It validates already-derived data and correctly refuses to invent clinical fields. It currently has no real bundle to load. |
| `src/data/acceptance.py` | Immutable, content-addressed real-data acceptance report and authorization decision. | Strong fail-closed gate. Current result is blocked. |

#### Important data-layer limitation

There is no `src/features` package, MIMIC query/extraction code, first-stay cohort builder, production canonical feature builder, or production `sofa_at(T)` implementation. Those are legitimate Sanskruti-owned omissions, but they must be supplied as versioned artifacts and/or callable production components.

### 4.2 Organ-support endpoint and label state machine

| Module | What is implemented | Quality and integration assessment |
|---|---|---|
| `src/labels/event_dict.py` | Parses and validates event dictionary versions, evidence contracts, state rules, and execution modes. | Strong governance boundary; refuses incomplete scientific mappings. |
| `src/labels/support_state.py` | Vasopressor interval normalization and ON/OFF state at cutoff, overlap/boundary logic, source evidence, duplicate/conflict checks. | Thorough and adversarially tested. Scientific mode requires frozen real definitions. |
| `src/labels/ventilation_state.py` | Invasive-ventilation state normalization with concept/adapter provenance and unknown-category hard failures. | Strong contract; real concept pinning is absent. |
| `src/labels/organ_support.py` | Composite OFF-to-ON initiation in `(t,t+24h]`, both-on not-at-risk handling, positive-before-exit precedence, early-exit censoring, onset evidence, audit dispositions, and Phase-7 record conversion. | Semantically careful. Real execution intentionally throws because source mappings and policies are not frozen. |
| `src/labels/phase4_governance.py` | Endpoint signoff/freeze audit. | Correctly reports human and provenance prerequisites instead of fabricating approval. |
| `src/labels/endpoint_freeze.py` | Produces real endpoint freeze record. | Current freeze is inactive and blocked. |
| `src/labels/support_prevalence.py` | Denominator-safe prevalence/count reporting with partition and G3 protections. | Strong; does not peek at real prevalence before freeze. |

`configs/event_dict_v1.yaml` is a **draft, non-authorizing dictionary**. Qualifying vasopressor mappings are empty/unverified; the ventilation concept and mappings are unverified; onset-at-exit, unknown-state, and final reason-vocabulary policies are unresolved; reviewers have not approved it. `artifacts/labels/support_endpoint_freeze_v1.json` lists eight real-freeze blockers.

### 4.3 Recurrent models and task outputs

| Module | What is implemented | Quality and integration assessment |
|---|---|---|
| `src/models/gru.py` | Unidirectional GRU encoder over exactly eight bins; concatenates values, observation masks, optional TSLO; excludes prefix padding with packed sequences; applies searched dropout to representation. | Good temporal handling and leakage prevention. **Blocks all static features**, which is a real integration defect described in Finding C3. |
| `src/models/gru_recovery.py` | Two-output recovery head with fixed +24/+48 order. | Correct independent outputs; no horizon chaining. |
| `src/models/gru_icu_time.py` | Scalar head for log1p remaining current-ICU time. | Correct task meaning. |
| `src/models/gru_support.py` | Single support logit; sigmoid remains external. | Correct loss/probability separation. |
| `src/models/lstm.py` | LSTM encoder configured as a cell-family swap. | Appropriate for one fixed sensitivity run; not an independent search. |
| `src/models/recovery_output.py` | Recovery output domain/horizon validation. | Good guard against output-order ambiguity. |
| `src/models/icu_time_target.py` | Forward transform and target validation. | Correct `log1p(hours)` target. |
| `src/models/icu_time_postprocess.py` | Stable inverse transform to nonnegative hours. | Shared evaluation/serving semantics are a strength. |
| `src/preprocess/target_scaler.py` | Horizon-specific training-only recovery target scaling and persisted metadata. | Correct conceptually and tested on synthetic data. |

There is no XGBoost model or training implementation in this repository. That is assigned to Sanskruti, but selection, TreeSHAP, and serving cannot complete until a compatible entrypoint, flattened-feature identity contract, artifacts, and loaders arrive.

### 4.4 Training framework

| Module group | Implemented behavior | Assessment |
|---|---|---|
| `training/engine.py`, `tasks/*` | Mask-aware regression/classification losses, zero-eligible batch handling, evaluation hooks, optimizer loop. | Solid reusable training core. |
| `early_stopping.py` | Patience logic, best-state restoration, finite monitor enforcement. | Good; the scientific GRU monitor remains unfrozen in config. |
| `class_weights.py` | Training-only support class weight with persisted provenance. | Correct split discipline. |
| `logging.py` | Deterministic structured run logging. | Good development evidence. |
| `checkpoint.py` | Model/optimizer state, metadata sidecar, SHA-256 verification, strict version compatibility, safe CPU load. | Strong integrity, but hard-coded to synthetic smoke checkpoints; unusable for real training as written. |
| `reproducibility.py` | deterministic seed setup and loader generator behavior. | Good within observed environment; no final environment lock. |
| `smoke.py`, task smoke modules | End-to-end synthetic task training artifacts. | Useful proof, not scientific evidence. |
| `lstm_comparison.py`, `lstm_smoke.py` | Fixed-settings LSTM sensitivity scaffold. | Correct design; no real selected-GRU-bound run exists. |

### 4.5 Metrics, weighting, bootstrap, selection, calibration, and analyses

| Module group | Implemented behavior | Assessment |
|---|---|---|
| `evaluation/weights.py` | Per-row weight `1/n_i` within each eligible stay. | Correct frozen estimator foundation. |
| `weighted_stats.py` | Weighted mean, median, percentile, and validation utilities. | Well tested. Median/tie convention still requires formal scientific freeze. |
| `metrics.py` | Recovery MAE/RMSE/directional agreement; ICU-hour errors; support AUROC/AUPRC/F1-related primitives; raw counts. | Broad and stay-balanced. AUPRC convention and exact-zero directional semantics are still review items. |
| `bootstrap.py` | Stay-cluster resampling with replacement, all rows per sampled stay, unique copy identity, estimator recomputation, percentile CI, invalid-replicate accounting. | Strong implementation of the specified clustered bootstrap. Scientific replicate count, seed, and minimum valid-replicate policy are unresolved. |
| `select.py`, `selection_validation.py`, `selection_audit.py` | Per-task validation-only XGBoost-vs-GRU selection with task-specific primary metrics and audit records. | Correct architecture. No real candidate results exist; tie policy/tolerance is unresolved. |
| `calibrate.py`, `calibration_validation.py`, audits | Selected-support-only isotonic fit on validation and transformed probabilities. | Correct isolation. Exact weighting/fitting policy still needs freeze and no real calibrator exists. |
| `threshold.py` | Validation F1 threshold search with explicit candidates. | Correct placement after calibration. Tie policy/procedure needs freeze; no real threshold exists. |
| `sensitivity.py` | Fixed LSTM comparison logic. | Correctly secondary to primary family selection. |
| `ablations.py`, `error_analysis.py`, `slices.py` and dry runs/audits | Configuration-driven ablation and slice frameworks with test protection. | Framework-complete but scientifically blocked until real definitions and artifacts are frozen. |
| `final_test.py` | G3 validation, one-run access ledger, selected-only evaluation hooks, result lineage, no-retry protections, G4 output. | Strong fail-closed orchestration. Execution inputs and plan are missing. |

### 4.6 Search and experiment governance

| Module | Implemented behavior | Assessment |
|---|---|---|
| `experiments/search_governance.py` | Canonical config hashing, exact budget validation, partition restrictions, frozen-config checks. | Strong control plane. |
| `search_registry.py` | Candidate/run records and artifact linkage. | Suitable schema; only synthetic history exists. |
| `dry_run.py` | Exact-30 synthetic candidate generation and registry exercise. | Verifies mechanics, not a scientific search. |
| `validation_suite.py` | Phase-19 preflight and coordinated real train/validation workflow interfaces. | Correctly blocked by missing acceptance and policy inputs. |
| `lineage.py`, `audit.py`, `registry_cli.py` | Bidirectional artifact/run lineage, hash verification, synthetic/scientific separation, audit CLI. | Strong and transparent. |
| `g3_cli.py`, `final_test_cli.py`, entrypoint | Governed gate/audit execution. | Safe by design; current prerequisites prevent use. |

Both search-space configs preserve the master-plan ranges and budget of 30 candidates per family/task, but do not freeze the sampler, scientific seed, several distributions, GRU early-stopping monitor, or XGBoost early-stopping details. There are no six real candidate-search manifests.

### 4.7 Artifact registry and gate infrastructure

`src/vedant_infra` provides canonical hashing, registry records, governance-state validation, and G3 freeze auditing. The implementation is conservative: synthetic artifacts cannot masquerade as scientific results, and test access is not opened when prerequisites fail.

The current G3 report has two passing checks—registry lineage and test non-use—and eight blocked prerequisites: split, selected manifest, real validation searches, LSTM sensitivity, support calibrator, support threshold, explanation adapters, and review signoffs.

### 4.8 Explainability

| Module | Implemented behavior | Assessment |
|---|---|---|
| `explainability/router.py` | Routes selected XGBoost to TreeSHAP and GRU to Integrated Gradients; binds task, family, model hash, manifest, cutoff, and prepared input. | Strong routing identity checks. |
| `explainability/ig.py` | Captum adapter, explicit target, baseline identity, attributable tensors, fixed padding, aggregation, convergence metadata, state-mutation checks. | Substantial implementation, but deliberately synthetic-only; real baseline/integration/output-domain policy is unresolved. |
| `explainability/tree_shap.py` | SHAP TreeExplainer adapter, explicit flattened feature identities, additivity check, mutation check, aggregation without heuristic name parsing. | Strong development adapter, but deliberately synthetic-only; no approved Sanskruti XGBoost loader/output-domain/background contract. |

The adapters are much more than placeholders, but they are not yet real-scientific adapters. More importantly, the actual `PredictionPipeline` passes `explanation_target=None`, while both real adapter paths require an explicit approved target. Current end-to-end pipeline tests use mock explanation adapters, so this mismatch is not exercised.

### 4.9 Serving and artifact loading

| Module | Implemented behavior | Assessment |
|---|---|---|
| `serving/artifacts.py` | Hash-first selected-manifest loading, exact model/preprocessor/metadata/calibrator/threshold checks, family/task/schema/version binding, real-vs-synthetic scope enforcement. | Excellent defensive boundary. No real selected bundle or concrete production loaders exist. |
| `serving/interfaces.py` | Protocols for canonical input and postprocessing. | Clean dependency inversion. |
| `serving/history.py` | Timeline/event contracts and a synthetic point-event provider/truncator with future-event rejection and truncation trace. | Correct demonstration. Explicitly not a production/stateful timeline provider. |
| `serving/preprocessing.py` | Injected canonical feature builder, target-free serving validation, family parity views, data-quality accounting. | Good boundary, but production builder/provider are absent; mutable last-request state creates a concurrency risk. |
| `serving/pipeline.py` | Loads selected bundle, transforms each task input, predicts, explains, postprocesses recovery/ICU, calibrates support, validates final response. | Architecturally complete with injected mocks. Actual explanation targets and production dependencies are unwired. |
| `serving/postprocessing.py` | Composite task postprocessor. | Correct modular seam. |
| `serving/recovery.py`, `recovery_display.py` | Original-unit deltas, current-SOFA reconstruction, display semantics. | Correct independence and current-SOFA use. Only demo/current-SOFA contracts exist; no real provider. |
| `serving/prediction_schema.py` | Strict request/response validation, finite numbers, hashes, exact fields, retrospective-mode labeling. | Strong. Config still identifies several shapes/policies as pending review. |

The support threshold is loaded and evaluated internally, but the public v1 response intentionally omits an alert/class and returns the calibrated probability only. That is consistent with the current proposed schema.

### 4.10 FastAPI layer

| Module | Implemented behavior | Assessment |
|---|---|---|
| `api/schemas.py` | Strict Pydantic request/response/error models and validators. | Strong transport contract. |
| `api/main.py` | `/health`, `/metadata`, `/predict`; sanitized 404/422/503/500 handling; metadata-manifest binding; no silent synthetic fallback. | Good fail-closed facade. The default `app` has no pipeline, so readiness is false and prediction is unavailable. |

The API is a valid shell around an injected `PredictionPipeline`; it is not currently a live model service.

### 4.11 Dashboard/replay

| Module | Implemented behavior | Assessment |
|---|---|---|
| `dashboard/api_client.py` | Typed client/error mapping for the API. | Clean separation from model internals. |
| `dashboard/catalog.py` | Replay-stay/cutoff catalog contracts. | Synthetic catalog only; no production catalog loader. |
| `dashboard/replay.py` | Validates selected legal cutoff and requests API prediction. | Correctly avoids local recomputation. |
| `dashboard/view_models.py` | Builds recovery, ICU-time, support, explanation, quality, and metadata panels. | Good presentation separation. |
| `dashboard/render.py` | Server-side escaped HTML. | Safe, simple renderer. |
| `dashboard/app.py` | Starlette app and prominent retrospective/not-clinical banner. | Correct product labeling; default app is unavailable without composition. |

### 4.12 Reproducibility and evidence

| Package | Implemented behavior | Assessment |
|---|---|---|
| `reproducibility/*` | Artifact rehashing, synthetic metric/bootstrap regeneration, frozen-artifact hooks, portability/privacy scans, cross-member evidence schema. | Strong honesty about limits. Real inference, metrics, bootstrap, and non-owner reproduction remain blocked. |
| `evidence/*` | Evidence tables/diagrams/report values, manifest generation, terminology/traceability audit, system-level evidence package. | Comprehensive and internally consistent. Generated blocked slots remain visibly blocked. |

There is no approved dependency lock, package-manager decision, or Git repository metadata. `observed_environment_phase15.json` is explicitly observational, not a reproducible lock.

---

## 5. Exact Sanskruti handoff assessment

### 5.1 Expected handoff inventory

The receiver and acceptance configuration expect the following real artifacts:

| Artifact | Expected owner/purpose | Receiver use | Current state |
|---|---|---|---|
| `artifacts/data/feature_dataset_v1.json` | Sanskruti canonical examples | Train/validation canonical rows | Missing |
| `artifacts/data/processed_schema_v1.json` | Field-name mapping and processed schema | Maps concepts without guessing column names | Missing |
| `artifacts/data/feature_schema_v1.json` | Dynamic/static feature order, dimensions, TSLO sentinel | Validates tensors and binds models | Missing |
| `artifacts/data/feature_provenance_v1.json` | Feature definitions/source/timestamp provenance | Audit and leakage verification | Missing |
| `artifacts/data/canonical_structural_index_v1.json` | Identifiers/times/grid/split only | Timestamp and split parity without opening outcomes | Missing |
| `artifacts/data/sofa_provenance_v1.json` | Pinned `sofa_at(T)` implementation/version evidence | Verifies shared current/future SOFA semantics | Missing |
| `artifacts/data/label_spec_v1.json` | Recovery and ICU label version/formulas | Formula and eligibility binding | Missing |
| `artifacts/splits/split_v1.csv` | Vedant subject split derived from upstream subject/year fields | Train/validation/test isolation | Missing |
| `artifacts/splits/split_v1.metadata.json` | Split hash/version/provenance | Integrity check | Missing |
| `artifacts/data/organ_support_handoff_v1.json` | Pulkit support labels/evidence | Attaches support target and mask | Missing |
| `artifacts/data/protected/test_labels_v1.parquet` | Protected final labels | Opened only after valid G3 | Missing/protected |
| `artifacts/data/real_data_inputs_v1.json` | Hash-bound bundle inventory | Entry point for the real adapter | Missing |

The acceptance report currently marks the local tensor/timestamp contracts as present but unbound; all substantive real inputs are absent. Therefore the receiver cannot yet answer whether a real Sanskruti output is compatible.

### 5.2 What should integrate cleanly once supplied

The following receiver checks are already implemented and should provide high confidence if the artifact formats exactly match:

1. regenerated legal cutoff sequence and 12-cutoff cap;
2. one canonical identity per `(stay_id, prediction_time)`;
3. subject-disjoint split parity and anchor-era agreement;
4. exactly eight time bins, correct prefix padding, and no padded observations;
5. dynamic/static feature order and dimensions from the schema sidecar;
6. genuine-observation masks independent of numeric imputation;
7. TSLO sentinel and in-lookback recurrence logic;
8. XGBoost/GRU information parity;
9. independent `SOFA(t+24)-SOFA(t)` and `SOFA(t+48)-SOFA(t)` recomputation;
10. `log1p((outtime-t) hours)` ICU target recomputation;
11. explicit task eligibility rather than inference from null targets;
12. no test outcome rows exposed during train/validation acceptance;
13. hash/version/provenance consistency across every inventory item.

### 5.3 What will not integrate without additional work

Even a perfectly formatted upstream bundle would encounter these downstream blockers:

- no real-data-to-`CanonicalTensorDataset` production preprocessing path;
- GRU rejects non-null static features;
- real checkpoints are rejected by the checkpoint metadata validator;
- search policy is not fully frozen and real search execution is absent;
- no Sanskruti XGBoost entrypoint/artifact loader exists;
- the Pulkit support file producer is not aligned with the real adapter's envelope;
- no selected-model manifest/calibrator/threshold can be produced until validation completes;
- real IG/TreeSHAP configurations and pipeline targets are unresolved;
- no production history provider, canonical feature builder, current-SOFA provider, or application composition root exists.

---

## 6. Critical and high-priority findings

### C1 — Real upstream handoff is wholly absent

**Severity:** Critical project blocker, not a code defect.  
**Evidence:** `artifacts/data_acceptance/real_data_acceptance_v1.json` is `BLOCKED`; Phase 19 is unauthorized.  
**Impact:** No real training, validation, selection, calibration, explanations, or serving can occur.  
**Resolution:** Sanskruti and the team must emit the exact hash-bound inventory above, then rerun acceptance. Do not bypass this gate.

### C2 — Pulkit support producer and Phase-18 receiver are not one executable contract

**Severity:** Critical cross-owner integration defect.  
**Evidence:** `build_phase7_handoff_records` emits `SupportLabelRecord` rows with audit keys such as `vaso_on_at_t` and `vent_on_at_t`. `data.real_adapter` expects an `organ_support_handoff_v1` envelope and receiver concepts/keys including `vasopressor_on_at_t` and `ventilation_on_at_t`. Tests independently fabricate receiver fixtures; no serializer bridges the state-machine output into the Phase-18 artifact.  
**Impact:** Pulkit's otherwise well-tested endpoint output cannot be consumed directly by the real acceptance path.  
**Resolution:** Freeze one versioned schema and add a deterministic serializer plus producer-to-receiver contract test using the real state-machine result.

### C3 — GRU cannot consume canonical static features

**Severity:** Critical model integration defect.  
**Evidence:** `GRUEncoderConfig.validate()` rejects every `static_dim != 0`, and `_validate_batch()` rejects every non-null `batch.static_features`. The canonical contract explicitly contains statics and requires information parity with XGBoost.  
**Impact:** If Sanskruti supplies any approved statics, the GRU cannot train or infer on the same canonical information, violating parity.  
**Resolution:** Review and version a static-input architecture—normally concatenating transformed statics with the temporal representation before task heads—and update both GRU and LSTM, manifests, search/config hashes, explainability aggregation, and parity tests. This is a scientific contract change, not a silent patch.

### C4 — Checkpoint implementation is synthetic-only

**Severity:** Critical training/release defect.  
**Evidence:** checkpoint metadata validation requires `synthetic_smoke_test is True` for both save and load.  
**Impact:** Real scientific checkpoints cannot be written or restored by the generic framework.  
**Resolution:** Introduce a versioned scope field and separate allowed metadata contracts for synthetic and real artifacts; preserve hash/version checks and prevent cross-scope loading.

### C5 — Actual explanation adapters are not wired through the real pipeline

**Severity:** Critical serving defect.  
**Evidence:** `PredictionPipeline.predict()` supplies `explanation_target=None` and no feature-schema hash; IG and TreeSHAP require explicit output targets. Both adapters reject non-synthetic execution because real domains/settings are unresolved. End-to-end tests use mock adapters.  
**Impact:** A real selected model bundle would fail explanation generation even if model prediction succeeded.  
**Resolution:** Put per-task/per-output approved explanation targets and feature-schema hash in the selected manifest; construct contexts from that metadata; freeze real IG baseline/integration policy and TreeSHAP output/background/flattening policy; add true pipeline tests against the actual adapters.

### C6 — Production serving composition does not exist

**Severity:** Critical deployment blocker.  
**Evidence:** only protocols/injection points exist for real model/preprocessor/calibrator loading, history retrieval, feature construction, current SOFA, and replay catalog. The default API/dashboard are intentionally unavailable.  
**Impact:** There is no command that launches a real, ready inference service.  
**Resolution:** After selected artifacts exist, implement a single configuration-driven composition root with concrete loaders/providers, exact manifest hash, restricted-data boundary, and startup self-check.

### C7 — Sanskruti XGBoost implementation and serving contract are absent

**Severity:** Critical dependency blocker.  
**Impact:** Per-task family comparison cannot run; selected-family serving and TreeSHAP cannot be completed if XGBoost wins.  
**Resolution:** Supply reviewed training entrypoint, exact flattened identities/order/hash, task output domains, preprocessing artifact, candidate results, checkpoint/model format, and serving loader.

### C8 — Real search and evaluation decisions are not frozen

**Severity:** High scientific-governance blocker.  
**Items:** sampler and seed; unresolved parameter distributions; GRU monitor; XGBoost early stopping; selection tie policy; AUPRC convention approval; directional-zero behavior; weighted-median convention; bootstrap B/seed/minimum valid repetitions; isotonic weighting; F1 threshold tie rule; ICU percentile list; slice definitions.  
**Impact:** Running real experiments now would create post hoc degrees of freedom.  
**Resolution:** Freeze and review these before Phase 19.

### C9 — Stateful data-quality reporting is unsafe for concurrent API requests

**Severity:** High runtime risk.  
**Evidence:** `CanonicalHistoryInputProvider` stores `_last_example` and `_last_trace` on the shared provider, while `PredictionPipeline` later calls `data_quality()` after processing task inputs. No concurrency/race tests exist.  
**Impact:** Concurrent requests can overwrite the state, causing a mismatch error or returning quality metadata for the wrong request.  
**Resolution:** Return request-scoped canonical input plus quality/trace in one immutable object, or key state by a request token with guaranteed cleanup. Avoid shared mutable “last request” fields.

### C10 — Real organ-support generation is intentionally impossible

**Severity:** High dependency blocker.  
**Evidence:** `generate_organ_support_label(..., SCIENTIFIC)` raises immediately; event dictionary and endpoint freeze are blocked.  
**Impact:** No real support targets can be generated, so the support GRU/XGBoost, calibration, threshold, and selected manifest remain blocked.  
**Resolution:** Verify/pin MIMIC mappings, ventilation concept, boundary/unknown policies, evidence tests, team reviews, and commit provenance; then activate scientific mode through reviewed version change.

### C11 — Final reproducibility and release identity are absent

**Severity:** High release blocker.  
**Evidence:** not a Git repository; no commit ID; no package manager or environment lock; no selected/G3/G4 artifacts; non-owner reproduction not executed.  
**Impact:** Results cannot be immutably attributed or recreated in a clean environment.  
**Resolution:** establish version control state, freeze dependency lock and platform expectations, and execute the existing reproduction plan after real artifacts exist.

### C12 — Proposed API schema contains pending-review decisions

**Severity:** Medium contract risk.  
**Evidence:** `prediction_schema_v1` marks additional-property policy, explanation-item schema, data-quality shape, metadata layout, and support-alert representation as proposed/unlocked.  
**Impact:** Client code may bind to a shape that reviewers later change.  
**Resolution:** complete Vedant/Sanskruti review and version any breaking changes before claiming a stable v1 service.

---

## 7. Test quality and what the tests do—and do not—prove

### Strongly covered areas

- exact timestamp boundaries, cap, timezone/elapsed-hour behavior, and horizon flags;
- subject isolation and split artifact tamper detection;
- canonical tensor shapes, feature order, padding prefix, masks, TSLO, task masks, and family information parity;
- future/pre-lookback/pre-ICU leakage failures;
- target semantics and independent recovery heads;
- masked training, zero-eligible batches, early stopping, checkpoint corruption/version checks;
- stay-balanced weights and metrics;
- clustered bootstrap copy semantics and invalid replicas;
- validation-only selection/calibration/thresholding;
- G3/final-test denial, one-run ledger, and result traceability;
- support state-machine boundaries, duplicates, censoring, and positive precedence;
- artifact hashes, model/preprocessor/calibrator/threshold compatibility;
- request/response schema and sanitized API errors;
- dashboard replay and presentation semantics;
- evidence and reproducibility audits.

### What passing tests do not prove

- correctness of any real MIMIC table/field mapping;
- correctness of Sanskruti's cohort, SOFA, feature aggregation, or real labels;
- producer-to-consumer compatibility of a real support handoff;
- production preprocessing or train/serve equivalence on real features;
- actual 30-candidate searches or per-task selected models;
- real calibration or threshold performance;
- Captum/SHAP runtime behavior in the current environment;
- actual adapter behavior inside the `PredictionPipeline`;
- API safety under concurrent requests;
- clean installation from a locked environment;
- final-test performance or any scientific result.

Thus the suite gives high confidence in **encoded contracts**, but not in a real data/model release.

---

## 8. Recommended integration sequence

The safest path preserves the current fail-closed gates.

1. **Resolve pre-data scientific decisions.** Freeze support mappings/policies, feature schema including statics and TSLO sentinel, search sampling/seeds/monitors, metric conventions, bootstrap policy, calibration/threshold policies, and review signoffs.
2. **Fix cross-boundary code contracts before receiving real data.** Align the support handoff envelope, add static-feature model support, create real checkpoint scope, remove request-global serving state, and wire explicit explanation targets.
3. **Receive Sanskruti's bundle without test outcomes.** Require the exact content-addressed inventory and protected structural-index policy.
4. **Generate Vedant's split artifacts** from verified subject/year inputs, then bind them back into the manifest.
5. **Run real-data acceptance.** Any failure must return to the owning producer; do not patch downstream assumptions around it.
6. **Implement/verify production tensor preprocessing.** Fit every learned transform on training subjects only and persist hashes/versions.
7. **Run exactly the frozen train/validation search.** Execute 30 candidates for XGBoost and 30 for GRU per task; keep test inaccessible.
8. **Select independently by task.** Recovery: stay-balanced 24h MAE; ICU time: stay-balanced weighted median absolute error in hours; support: stay-balanced AUPRC.
9. **Run fixed LSTM sensitivity**, using selected GRU settings where meaningful and no independent search.
10. **Fit selected-support isotonic calibration and F1 threshold on validation only.**
11. **Build the selected manifest and production serving bundle.** Include model, preprocessor, target transform, schema, calibrator, threshold, and explanation identities/hashes.
12. **Complete production composition and actual adapter integration tests.** Include concurrent requests and training-serving equivalence.
13. **Pass G3 and human reviews.** Only then authorize the one protected final-test run.
14. **Execute final test once, write G4 evidence, and reproduce independently.**

---

## 9. Acceptance checklist for Sanskruti's handoff

Before calling the handoff accepted, verify all of the following:

- [ ] versioned manifest is repository-relative and every file hash matches;
- [ ] cohort is adults, chronological first ICU stay, with no global hospital-death exclusion;
- [ ] structural index exactly regenerates the legal grid;
- [ ] subject/year information produces the frozen disjoint split;
- [ ] canonical rows contain no protected test outcomes during Phase 18/19;
- [ ] eight bins are oldest-to-newest and exactly cover `(t-48h,t]`;
- [ ] bin-edge observations follow the frozen open-left/closed-right policy;
- [ ] pre-ICU bins are padding and contain no clinical observations;
- [ ] observation mask means genuine in-bin observation only;
- [ ] forward fill never seeds from `<=t-48h`;
- [ ] TSLO uses only genuine in-lookback observations and the pinned sentinel;
- [ ] feature order/dtype/dimension and statics are explicit;
- [ ] XGBoost flattening is a deterministic view of exactly the same information;
- [ ] one `sofa_at(T)` implementation and provenance are used at all horizons;
- [ ] recovery targets are independently recomputable;
- [ ] ICU target is remaining current-stay time and independently recomputable;
- [ ] support rows join one-to-one on canonical key with explicit eligibility/evidence;
- [ ] train-only preprocessing policy and artifacts are declared;
- [ ] no restricted raw data, credentials, or identifying extracts are committed.

---

## 10. Final assessment

The work is **architecturally mature but scientifically pre-execution**. Vedant's strongest contributions are the temporal/tensor contracts, recurrent training scaffolding, stay-balanced evaluation, clustered bootstrap, artifact lineage, and strict test-set governance. Pulkit's strongest contributions are the endpoint state-machine design, hash-bound serving bundle, family-aware explanation framework, API contract, and replay UI. Both tracks show excellent awareness of leakage and governance failure modes.

The main risk is no longer “missing lots of framework code.” It is **false closure at owner boundaries**: individually tested modules do not yet form a single real artifact path. The support envelope mismatch, static-feature rejection, synthetic-only checkpoints, explanation-context gap, and missing production composition are the clearest examples.

Accordingly:

- Do not label the repository “real pipeline complete,” “production ready,” or “final evaluated.”
- It is reasonable to label it “Vedant/Pulkit framework implementation complete for synthetic contract verification,” provided the listed integration defects remain open.
- Sanskruti's eventual handoff should be treated as a governed integration event, not as a simple file drop.
- The existing fail-closed behavior should be preserved; blockers should be resolved by reviewed contracts and artifacts, never by bypassing validation.

Once C2–C6 are resolved and the full Sanskruti bundle passes Phase-18 acceptance, the repository should be in a strong position to begin real train/validation work. Until then, the most accurate overall status is **BLOCKED BEFORE REAL EXPERIMENTATION, WITH A SUBSTANTIAL AND WELL-TESTED SYNTHETIC/CONTRACT FOUNDATION**.

