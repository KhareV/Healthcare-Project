# Codex Project Context V1

> **HISTORICAL BASELINE — ACTIVELY AMENDED.** For the current synthetic cardiac project scope, read [`CODEX_PROJECT_CONTEXT_V2.md`](CODEX_PROJECT_CONTEXT_V2.md) and [`governance/project_scope_v2.md`](governance/project_scope_v2.md) first. V1 remains authoritative only for scientific and ownership topics not superseded by the v2 amendment. Project Scope v2 is `DRAFT COMPLETE — TEAM FREEZE REQUIRED`; no approval is implied.

## 1. Project identity and authority

**Project:** Personalized Patient Recovery Trajectory — Time-Series Learning for Recovery Forecasting, Remaining ICU Stay Time, and New Organ-Support Initiation Risk  
**Type:** B.Tech three-member retrospective ICU forecasting prototype  
**Dataset:** MIMIC-IV v2.2  
**Baseline status/date:** FINAL DEVELOPMENT BASELINE v1.0, 17 September 2026

Source precedence:

1. `Personalized_Patient_Recovery_Trajectory_FINAL_Implementation_Master_Plan.docx` — scientific and technical source of truth.
2. `Personalized_Patient_Recovery_Trajectory_FINAL_Work_Division_3_Members (1).docx` — ownership, handoff, review, evidence, schedule, and execution source of truth.
3. Frozen repository contracts/configuration and existing implementation.
4. General engineering knowledge.

If the documents appear to disagree scientifically, the Master Plan wins and the Work Division must be corrected. No scientific definition may change silently. Scientific changes require team review, a version increment, identification/regeneration of affected artifacts, and supervisor awareness where applicable. Poor empirical performance never authorizes a target, cohort, endpoint, or test-policy change.

## 2. Research question, scope, and non-scope

Research question: using retrospectively collected MIMIC-IV ICU data, can clinical history available at successive 6-hour cutoffs during an adult subject's first ICU stay forecast:

1. independent changes in SOFA at 24 and 48 hours;
2. remaining time until the current ICU stay terminates; and
3. 24-hour risk of a new qualifying vasopressor or invasive-mechanical-ventilation initiation?

The application demonstrates **RETROSPECTIVE SEQUENTIAL REPLAY — NOT REAL-TIME CLINICAL PREDICTION**. Feature inclusion uses the frozen clinical/event timestamp. Documentation, result-availability, and ingestion latency are not modeled and must be reported as limitations.

In scope: adult first ICU stay, repeated legal cutoffs, eight-bin 48-hour histories, masks/TSLO/static features, deterministic SOFA-based recovery labels, current-ICU remaining time, OFF-to-ON support initiation, naive/XGBoost/GRU comparison, fixed LSTM sensitivity, per-task validation selection, support calibration, cluster-aware evaluation, selected-family explanations, API, and replay dashboard.

Explicit non-claims/non-scope: prospective or continuous real-time operation; clinical validation or bedside safety; treatment recommendation; causal explanation; mortality as a primary task; hospital-discharge LOS; recovery/survival time as the ICU-time target; unconditional performance outside eligible horizons; external-hospital generalization; digital twins, LLMs, federated learning, or unnecessary survival modeling. MTL is optional after the core freeze; TFT is deferred outside the six-week critical path. Regression outputs are point forecasts, not generic confidence percentages or prediction intervals.

## 3. Architecture

The frozen logical flow is:

`MIMIC-IV raw tables -> adult first-ICU cohort -> legal cutoffs -> cutoff-aware 48h/8-bin features + padding/observation masks + TSLO + statics -> deterministic targets and task masks -> subject-level coarse temporal split -> naive/XGBoost/GRU (+ fixed LSTM sensitivity) -> validation selection per task -> selected_models_v1.json -> selected-family explanations + support calibration/threshold -> PredictionPipeline -> FastAPI -> retrospective replay dashboard`

The PredictionPipeline must load the per-task selected-model manifest, validate compatible hashes/versions, truncate stored raw history at the requested historical cutoff, call the same feature builder and fitted training preprocessor, run each selected task model, calibrate only support risk, route explanations by family, and return versioned task-specific metadata. It must not retrieve a precomputed answer or duplicate preprocessing inside the API/dashboard.

## 4. Team ownership

### Sanskruti Satish Shete — data truth and XGBoost

Owns MIMIC access/extraction; adult first-stay cohort; canonical raw timeline; variable and feature provenance; pinned deterministic `sofa_at(T)`; SOFA dependencies and component flags; eight-bin feature grid; strict lookback/missingness behavior; recovery and ICU-time labels; train-only preprocessing inputs; primary XGBoost implementation; data QA/leakage foundations; data/model evidence. She collaborates on, but does not own, the final support-event semantics.

### Vedant Khare (Person 2) — clock, temporal modeling, and statistical validity

Vedant owns:

- legal prediction timestamp generation and horizon-mask attachment;
- subject-level coarse temporal split, frozen mapping, and hash;
- `[B,8,F]` tensor contract, feature order/static handling, masks, TSLO, and synthetic canonical fixtures/loader;
- config-driven training, deterministic seeds, dataloaders, optimizer/early stopping, checkpointing, structured logs, hashes, and run registry;
- independent GRU models for recovery, ICU time, and support; one fixed LSTM sensitivity run per task;
- validation search with exactly 30 XGBoost and 30 GRU configurations per task and pre-frozen spaces;
- stay-balanced regression/classification metrics and mathematical toy tests;
- grouped stay bootstrap and reproducible percentile CIs;
- per-task XGBoost-versus-GRU model selection and `selected_models_v1.json`;
- isotonic calibration and F1 threshold for only the selected support classifier, using validation only;
- selected-family explanation metadata, not explanation computation;
- prespecified ablations, sensitivity and error analysis with denominators;
- G3 freeze marker, test-access governance, final evaluation path, model/evaluation evidence, and experiment lineage.

Sanskruti remains the primary XGBoost implementation owner. Pulkit remains the support-label and serving owner. Vedant owns fairness, selection, statistical evaluation, and the stable artifact handoff—not their internal implementations.

### Pulkit — support endpoint and product/serving truth

Owns qualifying vasopressor and invasive-ventilation state derivations; composite OFF-to-ON support label and censoring; endpoint freeze before prevalence; API prediction schema; raw-history truncation; PredictionPipeline; exact preprocessing reuse; selected model/explanation routing; Integrated Gradients and TreeSHAP adapters; FastAPI; replay dashboard; data-quality display; integration and compatibility tests; packaging, exact final environment, non-sensitive demo fixture, runbook, and release evidence.

## 5. Frozen scientific invariants

### Cohort and prediction unit

- Adults age `>=18` under the frozen age derivation.
- Retain the chronologically first ICU stay per `subject_id`, with deterministic tie handling.
- Require valid `intime`, `outtime > intime`, compatible `anchor_year_group`, and at least one legal cutoff.
- Do not globally exclude hospital deaths. ICU termination may be transfer, discharge, or death.
- `subject_id` is the patient/split unit; `stay_id` is the retained episode and bootstrap cluster; `hadm_id` is linkage only.

### Prediction clock and canonical history

- `t_k = intime + 24h + 6h*k`, for `k = 0,...,11`.
- Emit only if `t_k <= outtime - 6h`; maximum 12 rows/stay; no post-cap sampling.
- Canonical dynamic history is exactly `(t-48h, t]`.
- Eight disjoint bins: `(t-48,t-42], ... , (t-6,t]`.
- The first cutoff at ICU hour 24 has pre-admission padding in the nominal 48-hour tensor. Pre-ICU ward data never fills it.
- Nothing with event time at or before `t-48h` may enter the tensor or seed forward fill. Future-row perturbation and explicit raw-history truncation must leave `features(t)` unchanged.

### Missingness, padding, and TSLO

- Padding identifies a time position before ICU admission; it is distinct from an existing within-ICU bin with no measurement.
- Observation mask `M[b,f]` is 1 only if at least one genuine observation of feature `f` exists in bin `b`.
- Forward fill may carry a genuine in-window value to later bins only; it cannot cross the left boundary or originate pre-ICU.
- TSLO is hours since the most recent genuine observation within the active lookback and uses the frozen no-observation sentinel when none exists.
- Slope requires at least two genuine observations. Imputed repeats do not count.
- Imputation/encoding/normalization is fitted on training subjects only; masks continue to expose missingness.

### SOFA and recovery

- Use one pure, versioned, pinned MIMIC-Code-style `sofa_at(T)` for baseline and future cutoffs; only `T` changes.
- The preceding 24-hour SOFA window and the pinned missing-component convention apply at every cutoff. Store six component-observed flags. The rejected arbitrary `>50% missing` validity rule must not reappear.
- Required dependencies include respiratory PaO2/FiO2 plus invasive-ventilation context, platelets, bilirubin, MAP/qualifying vasoactive agents, GCS, creatinine, and 24-hour urine output, according to the pinned implementation.
- `Delta_SOFA24(t) = SOFA(t+24h) - SOFA(t)`.
- `Delta_SOFA48(t) = SOFA(t+48h) - SOFA(t)`. It is independent; never compute or reconstruct it by chaining the 24-hour delta.
- Recovery outputs are two values. Separate horizon scaling statistics are fitted on eligible training labels only. Scientific metrics use raw deltas. Only dashboard reconstructed absolute SOFA is clipped to `[0,24]`.
- 24h/48h eligibility requires `t+h <= outtime` and an executable deterministic SOFA call under the frozen protocol.
- Report a prespecified complete-six-components sensitivity subset without redefining the primary target.

### Remaining ICU-stay time

- `R_ICU(t) = (outtime - t)` in hours; training target `y = log1p(R_ICU(t))`.
- Report hours as `expm1(max(0, predicted_log_time))`, identically for every family in validation, test, serving, and dashboard paths.
- Meaning is time until the current ICU stay terminates—not hospital discharge, recovery, survival, or mortality.
- Every legal canonical row is ICU-time eligible retrospectively.

### New organ-support initiation

- At `t`, each component is at risk only if qualifying vasopressor support or invasive ventilation is OFF.
- Positive: any eligible OFF-to-ON initiation in `(t,t+24h]`; an event before early ICU exit remains positive.
- Negative: at least 24 hours of ICU follow-up and no eligible initiation.
- Censored: ICU ends before 24 hours with no preceding positive event. Censored rows are excluded from loss and metrics.
- At least one component must be OFF/at risk. Continued already-ON support is not an initiation.
- The endpoint is initiation, not generic escalation; it excludes dose escalation, ventilator-setting worsening, pressor count, graded intensity, and SOFA+2.
- Endpoint semantics and event dictionary are frozen before prevalence. Prevalence is a result, not a tuning target.

### Split and test isolation

- Train: `2008-2010` plus `2011-2013` anchor-year groups.
- Validation: `2014-2016`.
- Test: `2017-2019`.
- Assign and persist by `subject_id`; sets must be pairwise disjoint. Call it a coarse temporal holdout, not an exact chronological deployment simulation.
- Test labels, metrics, or plots cannot affect feature design, preprocessing, target scaling, class weights, hyperparameters, architecture, family selection, calibration, threshold, or explanations. Test opens only after G3. A result-affecting post-open defect invalidates the frozen result and requires explicit version reset and affected-path rerun.

### Candidates, training, and selection

- Candidate families per task: naive, XGBoost, GRU. LSTM is a sensitivity comparison. Naive is a reference, not a serving-selection candidate.
- Naive recovery predicts both deltas as zero; naive ICU time predicts the eligible training-set stay-weighted median log target; naive support predicts eligible training prevalence.
- XGBoost and GRU consume identical eight-bin summaries, masks, TSLO, statics, preprocessing provenance, task definition, and split. XGBoost only flattens the sequence.
- Run exactly 30 validation configurations per task for XGBoost and 30 for GRU; freeze each search space before search. Early stopping/restoring a best iteration or epoch is inside one configuration.
- GRU cap: 60 epochs/configuration, patience 8, restore best epoch, AdamW unless a pre-training versioned feasibility revision is approved.
- LSTM: exactly one declared run per task, using selected GRU hyperparameters wherever meaningful except the recurrent cell; no independent search and no serving-selection role.
- Recovery selection: lowest stay-balanced validation 24h MAE; report 48h separately.
- ICU-time selection: lowest stay-balanced validation weighted median absolute error in hours.
- Support selection: highest stay-balanced validation AUPRC before thresholding.
- Select XGBoost versus GRU independently per task. Model superiority or baseline improvement is never a gate.

### Evaluation, calibration, and explanation

- For task `T`, an eligible snapshot `j` in stay `i` with `n_i` eligible snapshots has weight `1/n_i`; every stay contributes total weight 1.
- Report examples and unique ICU stays separately; classification also reports positive examples and positive stays.
- Recovery: primary 24h stay-balanced MAE; secondary RMSE/directional agreement and corresponding 48h metrics. No 6-hour recovery direction metric.
- ICU time: primary stay-balanced weighted median absolute error in hours; secondary weighted MAE/RMSE/error percentiles.
- Support: primary stay-balanced AUPRC; secondary weighted AUROC, Brier, and frozen-threshold sensitivity/specificity/precision/F1.
- Bootstrap `stay_id` with replacement. Each stay copy brings every eligible snapshot and receives total weight 1; recompute the identical estimator. Use 2.5/97.5 percentiles. Skip class-invalid replicates per metric and report valid replicate counts. This is aggregate-performance uncertainty, not individual-patient uncertainty.
- Only the already-selected support classifier is isotonic-calibrated on validation predictions/labels. Choose the operating threshold by predefined validation F1 on calibrated validation probabilities. Freeze calibrator and threshold before test.
- Explanation routing is mandatory: selected XGBoost -> TreeSHAP; selected GRU -> Integrated Gradients. Use non-causal language such as “contributed to the prediction.”

## 6. Shared contracts and handoffs

All contracts require a versioned schema, sample/synthetic fixture, automated contract test, accountable owner, and mandatory reviewer. Contract-breaking changes identify stale artifacts/tests and bump versions; downstream owners do not locally patch around incompatibility.

| Contract | Owner | Required content / receiver |
|---|---|---|
| `processed_schema_v1.json` | Sanskruti | Raw/processed columns, types, units, event-time and provenance IDs; consumed by features and serving. |
| feature schema v1 | Sanskruti | Ordered feature definitions, aggregation, masks, TSLO and padding; consumed by Vedant/Pulkit. |
| `sofa_spec_v1.yaml` | Sanskruti | Pinned MIMIC-Code commit, variables, thresholds/window and missing-component convention; Vedant reviews. |
| `event_dict_v1.yaml` | Pulkit | Qualifying supports, source mappings, state/transition and censoring rules; Sanskruti and Vedant review. |
| `timestamp_spec_v1.yaml` | Vedant | Cutoff equation, bin boundaries/cap and horizon-mask rules; Sanskruti reviews. |
| `split_v1.csv` + SHA-256 | Vedant | `subject_id -> split`; Sanskruti reviews isolation/eras. |
| `tensor_contract_v1.json` | Vedant | `[B,8,F]`, order/dtypes, statics, padding/missingness/TSLO and task masks; Sanskruti/Pulkit review. |
| `prediction_schema_v1.json` | Pulkit | API units/keys, task-specific models, explanations/data quality/version metadata; Vedant/Sanskruti review. |
| `selected_models_v1.json` | Vedant | Per-task family, path, SHA-256, feature version and explanation; support calibrator/threshold; Pulkit validates. |
| exact environment lock | Pulkit | Actual final training/serving Python/package and GPU/driver details where applicable; Vedant reviews. |

Critical handoffs:

- Sanskruti -> Vedant: canonical examples with subject/stay/time, eight bins, masks, TSLO, statics, labels and task eligibility; SOFA/current/future values and recovery/ICU-time formula inputs. Acceptance requires tensor/flattened parity and label spot checks.
- Pulkit -> Vedant: support label, eligibility, OFF-at-`t` evidence, onset time, and censor reason. Acceptance requires state-machine/censoring spot checks and only post-freeze prevalence reporting.
- Vedant -> Pulkit: selected manifest and frozen preprocessors/target statistics/model artifacts with hashes and training provenance. Acceptance requires artifact compatibility and direct-pipeline equality.
- Sanskruti -> Pulkit: raw demo schema, cutoff-aware shared feature builder, exact preprocessing interface, and current-SOFA/component flags. Acceptance requires feature and recovery-reconstruction equivalence.

## 7. Repository ownership map

| Area | Accountable owner |
|---|---|
| `src/data/` extraction, cohort, timeline, SOFA | Sanskruti |
| `src/data/` timestamps and split | Vedant |
| `src/features/` | Sanskruti |
| `src/labels/` recovery and ICU time | Sanskruti |
| `src/labels/` support state/label | Pulkit |
| `src/models/` XGBoost | Sanskruti; Vedant owns fair comparison/selection |
| `src/models/` GRU/LSTM and training framework | Vedant |
| `src/evaluation/` | Vedant |
| `src/explainability/`, `src/serving/`, `api/`, `dashboard/` | Pulkit |
| `configs/` | Domain owner plus mandatory reviewer |
| `tests/` | Distributed by invariant/failure domain |
| `experiments/registry.*` | Vedant owns registry; every owner writes metadata |
| `artifacts/` | Immutable, versioned outputs; no restricted raw data |
| `docs/evidence/` | All three by domain |
| `data/demo/` | Pulkit; non-sensitive fixture |

## 8. Testing and leakage rules

Every test must prove a semantic invariant rather than merely demonstrate execution. For each component, ask which plausible implementation would silently invalidate the science, then encode that failure as a test.

Required focus:

- cutoff earliest/latest/equality rules, six-hour grid, maximum 12 rows, exact bin-edge assignment;
- future-row perturbation and full-history versus explicit truncation equivalence;
- exclusion of values at/before the hard left boundary, including a 60-hour-old forward-fill seed;
- distinct pre-ICU padding versus clinical missingness and genuine-observation-only masks/TSLO/slopes;
- `sofa_at(T)` future exclusion and identical implementation/config at every cutoff;
- exact independent recovery formulas and independent dashboard reconstruction;
- current-ICU-time transform/postprocess with no hospital `dischtime` dependency;
- support already-ON, OFF-to-ON, dual-component, early-exit censoring, and positive-before-censor cases;
- task horizon masks near `outtime`;
- subject isolation, era assignment, frozen split hash, and training-fit subject provenance;
- identical raw XGBoost/GRU information before representation;
- hand-calculated stay weights, weighted median, weighted classification metrics, and stay-copy bootstrap;
- validation-only selection/calibration/threshold provenance;
- mixed-family manifest routing and family-correct explanations;
- serialization/deserialization, run lineage and deliberate schema/hash incompatibility rejection;
- direct training pipeline versus API numerical equivalence and replay recomputation from truncated input.

Do not use correlation similarity, train/all scaler-statistic inequality, near-perfect AUROC, arbitrary prevalence ranges, grouped-vs-row CI width, or expected performance as “leakage tests.” A changed prediction between cutoffs is also not a replay-correctness requirement.

## 9. Experiment and artifact governance

Every registered run records at least: `run_id`, task, family, seed, full config and config/search-space hash, MIMIC/cohort/extraction version/hash, feature/label/event-dictionary/split versions and hashes, preprocessor and training-fit provenance, code commit, artifact/checkpoint hash, validation metrics, status, and notes.

Final reproducibility additionally pins the exact environment actually used, MIMIC-Code commits for SOFA/ventilation, extraction SQL/code SHA-256, feature dictionary hash, selected model/calibration/threshold lineage, and serving build/demo fixture versions. Evidence tables/plots record their generating script and input run/artifact IDs. Raw restricted data is never committed.

The serving source of truth is `selected_models_v1.json`, with task-specific family/version/hash rather than a misleading generic model version. Any semantic or preprocessing change capable of changing predictions increments the appropriate version and retrains/regenerates dependent artifacts.

## 10. Integration gates and six-week sequence

| Gate | Timing | Required outcome |
|---|---|---|
| G0 — scientific/contract freeze | End Week 1 | Research/non-claims, event-time/SOFA/support/timestamp/split semantics and processed/tensor/prediction contracts frozen; synthetic data -> mock model -> API/dashboard path works; no unresolved scientific/interface semantics. |
| G1 — data/label freeze | End Week 2 | Real cohort, split, canonical rows/features, SOFA, three labels/masks and QA; manual cases plus structural leakage/censoring tests pass. |
| G2 — core modeling integration | End Week 3 | Naive/XGBoost/GRU use identical frozen information/task/split; grouped metrics tested; first real PredictionPipeline/replay works; no performance threshold. |
| G3 — selection/calibration freeze | End Week 4 | Searches complete, fixed LSTM sensitivities run, per-task manifest plus support calibrator/threshold and explanation adapters frozen before test access. |
| G4 — test/evaluation freeze | End Week 5 | Test opened once; final stay-balanced metrics, grouped CIs, sensitivity/error analysis, integration suite and screenshots reproducible with no post-test tuning. |
| G5 — submission freeze | Mid Week 6 | Non-owner clean reproduction of data, one model/evaluation run, and demo; report/slides/runbook/evidence/viva complete. Remaining time is bug/documentation polish only. |

Week sequence: Week 1 contracts and synthetic path; Week 2 real data/labels/split and structural QA; Week 3 XGBoost/GRU searches plus first real replay; Week 4 selection/LSTM/calibration/threshold/manifest; Week 5 single final test and frozen analyses; Week 6 independent reproduction, report, slides, runbook, viva, and buffer. Drop MTL/TFT and UI polish before sacrificing rigor.

## 11. Reproducibility and cross-review

A component is not reproducible merely because its owner can run it. A non-owner must:

- reproduce cohort/feature/label outputs from the frozen extraction/config;
- reproduce one frozen model-training/evaluation run within declared stochastic tolerance; and
- install the exact environment and run raw demo timeline -> cutoff -> pipeline -> API -> dashboard without undocumented local steps.

Key mandatory reviews: Vedant reviews cohort/event-time/provenance, SOFA, recovery/ICU-time math, support censoring, and API model metadata; Sanskruti reviews Vedant's timestamps, split, stay-balanced metrics/bootstrap, and Pulkit's event mappings/dashboard semantics; Pulkit reviews GRU serialization, Vedant's selected manifest/calibration handoff, and feature-serving compatibility. Authors cannot self-approve scientific or cross-layer changes.

## 12. Vedant Definition of Done

- Timestamp generator and split are frozen, versioned/hashed, boundary-tested, and subject-disjoint.
- Tensor contract and synthetic loader prove ordered shapes, padding/missingness, masks/TSLO, task censoring, and flattened-versus-sequence information parity.
- Naive/XGBoost/GRU paths are config-driven and solve identical task definitions with identical information; exactly 30 XGBoost and 30 GRU validation configurations per task are traceable; LSTM has one fixed run/task.
- Stay-balanced regression/classification metrics and stay-cluster bootstrap pass analytic toy fixtures.
- Recovery, ICU-time, and support selections use only their frozen validation criteria; `selected_models_v1.json` exists and validates before test access.
- Selected support calibrator and F1 threshold have validation-only IDs/provenance.
- G3 marker prevents premature final test evaluation; the eventual single test result is reproducible from frozen artifacts.
- Ablations, complete-component sensitivity, error slices and all denominators are generated without cherry-picking or must-improve criteria.
- Every result-table row traces through the registry to code/config/data/split/model hashes.
- A non-owner reproduces one training/evaluation run; Vedant can defend data/labels, serving/replay, limitations, and his ML/statistical layer.

## 13. Known unresolved items requiring verification

These are deliberately not guessed and must be resolved through the named pre-freeze verification or contract process:

- **UNRESOLVED — REQUIRES VERIFICATION:** Actual MIMIC-IV v2.2 access and the exact modules, tables, fields, clinical/event timestamps, units, conversions, and derived concepts available for extraction.
- **UNRESOLVED — REQUIRES VERIFICATION:** The exact adult-age derivation and deterministic tie rule for simultaneous first-ICU-stay `intime` values.
- **UNRESOLVED — REQUIRES VERIFICATION:** Exact MIMIC-Code commit/hash and verified schema feasibility for every SOFA dependency, including FiO2/PF ratio, invasive ventilation context, vasoactive rates, GCS, and 24-hour urine output.
- **UNRESOLVED — REQUIRES VERIFICATION:** Exact MIMIC-Code ventilation-concept commit/hash and the concrete invasive-ventilation source mapping.
- **UNRESOLVED — REQUIRES VERIFICATION:** Qualifying vasopressor set, source fields, units/rates, interval semantics, and the finalized OFF/ON rules in `event_dict_v1.yaml`.
- **UNRESOLVED — REQUIRES VERIFICATION:** The numeric/encoded TSLO no-observation sentinel, imputation safe defaults, per-feature valid ranges/aggregations, and delta behavior when the preceding bin is unobserved. These belong in frozen feature contracts, not ad hoc code.
- **UNRESOLVED — REQUIRES VERIFICATION:** Concrete primary seed/project seed policy, target-stochastic tolerance, Integrated Gradients baseline, and final grouped-bootstrap replicate count (`B`) if not set by the phase config.
- **UNRESOLVED — REQUIRES VERIFICATION:** Actual final Python/package/CUDA/driver versions and hashes. Freeze the environment that successfully runs final training and serving before final training.
- **UNRESOLVED — REQUIRES VERIFICATION:** MIMIC data-use constraints governing the eventual demo fixture; only synthetic or explicitly policy-compliant data may enter the repository/package.
- **UNRESOLVED — REQUIRES VERIFICATION:** Minor repository-contract naming differences between the documents (for example `feature_schema_v1.yaml` in the Master Plan repository sketch versus `feature_schema_v1.json` in the Work Division contracts; `src/data/time_grid.py` versus `src/data/timestamps.py`; and `src/serving/prediction_pipeline.py` versus `src/serving/pipeline.py`). These are implementation/interface decisions, not permission to alter science; freeze one versioned path/format at G0. The Master Plan's `artifacts/models/selected_models_v1.json` location is the current authoritative default.

No substantive scientific conflict was found between the documents. The Work Division expressly defers scientific definitions to the Master Plan and is consistent with it on the locked cohort, clock, lookback, targets, split, selection, calibration, evaluation, and gate policy.

## 14. Permanent execution rules

- Contract-first and mock/synthetic-first development allows parallel work without coupling to another owner's internals.
- No notebook-only implementation, hidden local state, manual metric edits, stale artifacts, or silent assumptions.
- Use deterministic/config-driven entry points, structured logs, explicit run IDs, immutable versioned artifacts, hashes, and fail-closed compatibility checks.
- Preserve phase isolation: feature/label work cannot inspect outcomes to redefine science; validation makes development decisions; test is final evaluation only.
- Maintain exact terminology in code, API, dashboard, report, and viva: “new organ-support initiation,” “remaining time until current ICU stay ends,” “coarse temporal holdout,” and “retrospective sequential replay.”
- Always report retrospective event-time, informative horizon eligibility/censoring, MIMIC-Code missing-component behavior, single-center practice dependence, and lack of individual prediction intervals as limitations.
