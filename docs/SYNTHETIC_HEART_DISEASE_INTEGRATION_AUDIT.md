# 1. EXECUTIVE VERDICT

> **Synthetic Heart-Disease Integration Acceptance Audit**  
> Audit date: 2026-09-20 (Asia/Kolkata)  
> Integrated root: `/Users/Home/Downloads/HealthCareProject`  
> Candidate root: `/Users/Home/Downloads/HealthCareProject/Heart-Failure-Recovery-and-Deterioration-Trajectory-Prediction`

| Question | Verdict |
|---|---|
| Correct current scope | Synthetic data and cardiac focus are confirmed by the later user decision; no repository evidence approves other scientific changes |
| Sanskruti | **REBUILD** the scientific track against retained contracts; salvage selected generator/QA ideas only |
| Vedant | **USE WITH CONTRACT MIGRATION**; framework/code reuse **HIGH**, trained/final artifact reuse **LOW** |
| Pulkit | **USE WITH CONTRACT MIGRATION**; serving/product code reuse **HIGH**, task artifacts and data mappings require regeneration |
| Best-supported scenario | **Scenario A:** synthetic cardiac data with the original longitudinal tasks/clock retained |
| Confidence | **HIGH** on code/integration findings; **MEDIUM** on final cohort name because heart disease versus heart failure is not versioned |

The earlier MIMIC-specific objections are superseded. Lack of MIMIC SQL, item IDs, access, MIMIC cohort counts, and MIMIC-Code provenance is not a defect in a synthetic final project. The prior acceptance decision nevertheless remains **REBUILD** because its decisive non-MIMIC findings remain: the candidate implements a different clock and target problem, fits imputation on all data, exposes the latent target class in the canonical handoff, does not meet the shared tensor/serving contracts, and uses a fake backend prediction path.

The integrated Vedant/Pulkit implementation should not be discarded. Its models, training engine, metrics, bootstrap, search/selection/calibration governance, artifact validation, cutoff-safe serving pipeline, explanations, API, replay, packaging, and tests are primarily data-source agnostic. The synthetic migration requires new upstream contracts and retraining, not wholesale downstream rewriting.

# 2. REVISED PROJECT SCOPE FOUND

The precedence audit found:

1. The current user instruction is the only explicit later authority found that says MIMIC authorization was not granted, synthetic patient data is final, and the population is heart-disease/cardiac focused.
2. No checked-in `project_scope_v2`, decision record, commit, review, config, or report in either root states that prediction times, eight-bin history, SOFA recovery, remaining ICU time, organ-support initiation, model families, metrics, calibration, or replay semantics changed.
3. The integrated root still consistently implements the original three tasks and repeated legal cutoffs. Its root README says these contracts are structurally tested with synthetic fixtures, not that the final data source/task was migrated.
4. Candidate README/docs implement heart-failure 24/48/72-hour trajectory classification, but candidate documentation ranks below the integrated contracts and is not evidence of team approval.

Therefore the currently defensible scope is: **final synthetic cardiac cohort; retained original longitudinal modeling semantics unless and until the team approves a versioned amendment**. The old MIMIC-specific split rule cannot be carried over literally and needs a synthetic subject-level replacement, while its isolation/governance purpose remains.

**MAJOR GOVERNANCE GAP — FINAL SYNTHETIC HEART-DISEASE SCOPE IS NOT VERSIONED IN THE REPOSITORY.** This must be closed before final training/evaluation.

# 3. SCOPE-DELTA MATRIX ORIGINAL VS REVISED

| Topic | Original FINAL DOCX | Latest verified decision | Evidence | Classification |
|---|---|---|---|---|
| Dataset source | MIMIC-IV v2.2 | Synthetic patient database/artifact | Explicit later instruction | **SUPERSEDED** |
| Population | All adult first-ICU subjects | Heart-disease/cardiac focused | Explicit later instruction | **SUPERSEDED**; exact cardiac definition ambiguous |
| Episode boundary | Current ICU stay | No later change found | Integrated timestamp/schema/API contracts | **RETAINED** |
| Patient identifier | `subject_id` | Synthetic subject identifier required | Data-source change; receiver identity remains | **SYNTHETIC EQUIVALENT** |
| Stay/admission identifier | `stay_id` | Synthetic ICU-like episode ID required | Receiver contracts | **SYNTHETIC EQUIVALENT** |
| First-stay rule | Chronologically first ICU stay | No explicit removal; trivial only if generator creates one episode | No amendment | **RETAINED/AMBIGUOUS implementation** |
| Timestamp generator | `intime+24h+6h*k`, `k=0..11` | No later change found | `timestamp_spec_v1` remains current | **RETAINED** |
| First prediction | 24h after episode start | No later change found | Same | **RETAINED** |
| Cadence/max rows | Every 6h; max 12 | No later change found | Same | **RETAINED** |
| End bound | `t <= outtime-6h` | Synthetic episode-end equivalent | Data source changes fields, not purpose | **SYNTHETIC EQUIVALENT** |
| History | `(t-48h,t]` | No later change found | Tensor/history contracts | **RETAINED** |
| Bins | Eight × six hours, open-left/closed-right | No later change found | `timestamp_spec_v1`, tests | **RETAINED** |
| Feature set/order | Frozen versioned canonical schema | Must be frozen for synthetic variables | Candidate list not approved | **SYNTHETIC EQUIVALENT/AMBIGUOUS** |
| Statics | Versioned canonical statics | Cardiac/synthetic statics to freeze | No v2 spec | **SYNTHETIC EQUIVALENT** |
| Missingness | Genuine observation state; train-only fitting | Same methodological requirement | Data-source agnostic | **RETAINED** |
| Masks/TSLO/padding | Separate observation mask, TSLO, structural padding | No later change found | Current tensor contract | **RETAINED** |
| Severity | One cutoff-safe standard/pinned `sofa_at(T)` | MIMIC pin obsolete; exact synthetic severity definition not frozen | User changes data source only | **SYNTHETIC EQUIVALENT/AMBIGUOUS** |
| Recovery targets | Independent continuous ΔSOFA +24/+48 | No approved replacement found | Integrated task/model/API contracts | **RETAINED pending synthetic SOFA spec** |
| Recovery horizons | 24h and 48h | No approved 72h addition | Same | **RETAINED** |
| Remaining-stay target | `log1p(outtime-t)` current ICU hours | Synthetic episode-end equivalent | Data source changes boundary source | **SYNTHETIC EQUIVALENT** |
| Third target | New organ-support initiation in `(t,t+24h]` | No approved deterioration replacement | Pulkit phases/config/API | **RETAINED** |
| Eligibility/censoring | Task-specific; incomplete negative follow-up censored | No later change found | Current timestamp/label contracts | **RETAINED** |
| Split | MIMIC era groups by subject | Meaningless for generator's single 2020 synthetic year | User changes source; candidate uses seeded patient split | **SYNTHETIC EQUIVALENT REQUIRED** |
| XGBoost | Same canonical data, flattened | No later change found | Search/parity framework | **RETAINED** |
| GRU | Per-task recurrent candidates | No later change found | Integrated code | **RETAINED** |
| LSTM | Fixed sensitivity | No later change found | Integrated code | **RETAINED** |
| Search budget | Exactly 30/family/task on validation | No later change found | Search governance | **RETAINED** |
| Metrics | Task-specific stay-balanced metrics | No later change found | Evaluation modules | **RETAINED** |
| Bootstrap | Grouped episode bootstrap | Still relevant with repeated cutoffs | Evaluation modules | **RETAINED** |
| Calibration/threshold | Selected support classifier; validation isotonic/F1 | No later change found | Phase 13 | **RETAINED** |
| Explainability | XGB→TreeSHAP; GRU→IG | Data-source agnostic | Router/config | **RETAINED** |
| API | Three-task cutoff request/response | No later change found | `prediction_schema_v1` | **RETAINED** |
| Dashboard/replay | Retrospective legal-cutoff replay | No later change found | Pulkit phases 11–14 | **RETAINED** |
| Test isolation | Validation-only development; guarded final test | Data-source agnostic | Governance code | **RETAINED** |
| Evidence | Full lineage/reproducibility | Replace MIMIC provenance with generator provenance | User source change | **SYNTHETIC EQUIVALENT** |

# 4. UNRESOLVED SCOPE QUESTIONS

The team must freeze: exact cardiac inclusion; whether episodes remain ICU-like; whether “first” matters with repeated synthetic episodes; standard SOFA versus a renamed cardiac severity score; exact synthetic feature/statics set; third-task event generation; synthetic split design; and whether the candidate's three-class deterioration task was ever formally adopted.

No evidence supports silently changing the clock or targets. Accordingly this audit provides both scenarios but treats Scenario A as operative:

- **Scenario A — minimal amendment:** synthetic cardiac data; retained repeated-cutoff, recovery, remaining-stay, and support architecture.
- **Scenario B — full candidate task:** synthetic heart-failure data; fixed first-48h representation and 24/48/72 deterioration classification.

# 5. HEART DISEASE VS HEART FAILURE DECISION

**AMBIGUOUS — HEART-DISEASE COHORT DEFINITION MUST BE FROZEN.** The later instruction says heart disease/cardiac patients. Candidate code assumes every generated patient has heart failure and contains no diagnosis distribution or inclusion logic beyond that assertion. Folder/title and `heart_failure_codes.yaml` do not constitute a team decision.

If the intended population is heart failure only, the candidate's population label broadly matches, although there is no subtype/diagnostic generator. If it is all or selected heart diseases, the candidate excludes coronary disease, arrhythmia, cardiomyopathy without HF, valvular disease, and other potential groups by construction. Scenario A can support either after a versioned generator/cohort definition; current candidate data support only an undifferentiated HF-only interpretation.

# 6. DATA-SOURCE CHANGE ASSESSMENT

The MIMIC→synthetic change removes requirements for PhysioNet authorization, MIMIC SQL/item IDs, real table schemas, real cohort counts, and MIMIC-Code commits. It does **not** remove requirements for exact event timestamps, episode boundaries, units, provenance, train-only preprocessing, leakage prevention, target definitions, subject isolation, or artifact lineage.

Replace “real/MIMIC” gates with an **AUTHORIZED_PROJECT_DATA** scope, backed by `synthetic_schema_v1`, `synthetic_generator_v1`, and a generator-produced manifest. Do not weaken hash validation, test guards, or training-serving equivalence.

# 7. SYNTHETIC DATABASE QUALITY AUDIT

The candidate is a generated Parquet dataset, not a relational database. An in-memory execution of `src/data/demo_data.py` at seed 42 produced:

| Property | Observed result |
|---|---|
| Patients/admissions | 100/100; exactly one admission per patient |
| Rows/columns | 5,936 × 21 |
| Latent trajectories | 35 improving, 35 stable, 30 deteriorating |
| Episode observed span | mean 72.835h; min 23.535h; max 118.798h |
| Observations/episode | mean 59.36; range 18–99 |
| Age | 46–88; mean 65.88 |
| Gender | 51 F, 49 M |
| Vital missingness | approximately 11.1–12.5% |
| Lab missingness | approximately 29.5–31.2% |
| Duplicate timestamp rows | 28 rows involved; duplicates deliberately injected |
| Exact numeric episode clones | 0 |
| Calendar | all episode starts in synthetic year 2020 |
| Seed reproducibility | repeat generation was DataFrame-exact |

Strengths: deterministic local RNG; explicit ranges/units for 12 dynamic variables; irregular 0.5–2h cadence; variable episode lengths; variable per-patient missing rates; no cross-patient clone detected; clear synthetic disclaimer.

Weaknesses: only 100 independent subjects; one episode each; no disease subtype/comorbidity generation despite configured comorbidities; no interventions/support events; no censoring process; no explicit episode end (last measurement is later relabeled discharge); independent feature equations can violate multivariate physiology; missingness is mostly random and not care-process-dependent; trajectory classes use three fixed prototype baselines/slopes; and the stored `trajectory` is an outcome-generating latent variable in raw rows.

# 8. SYNTHETIC GENERATOR / PROVENANCE AUDIT

`src/data/demo_data.py:192-269` records parameters in function defaults and uses `np.random.default_rng(42)`. `config/config.yaml:4-5` enables demo mode and repeats the seed. However there is no immutable generator schema/version artifact, code/config hash bound to the Parquet, producer run ID, generation timestamp/command manifest, or artifact member hash. The committed Parquet could not be proven byte-for-byte to be the output of the current generator because no Parquet engine is installed and no producer manifest exists.

Timestamps begin at a random hour in 2020. `n_hours` is not persisted; `cohort_builder.py:75-95` substitutes first and last measurement for admission/discharge. That loses the intended episode boundary and biases remaining-stay labels. The generator emits no support intervals, severity components, explicit outcome events, or censoring.

Required provenance: generator version/hash, config hash, seed or per-record seed derivation, schema version, generation command/runtime, output hash, exact population rules, episode boundary fields, feature equations/units/ranges, missingness/intervention/outcome mechanisms, and known limitations.

# 9. SYNTHETIC TARGET-LEAKAGE AUDIT

**TARGET-GENERATION LEAKAGE RISK: CRITICAL for the documented shared handoff; HIGH for the actual XGBoost path.**

- The generator chooses `trajectory` first (`demo_data.py:218-228`) and uses its class-specific baseline and slope for every dynamic feature (`:25-69`, `:115-150`).
- It writes `trajectory` into every raw row (`:132-140`).
- Target generation uses that same trajectory as the base label (`generate_targets.py:101-151`). Across eligible rows, 99.30% of 24h, 97.53% of 48h, and 96.79% of 72h labels equal the latent class.
- `docs/FEATURE_BUILDER.md:61,145-153` explicitly places `trajectory` in the canonical grid and calls it a static covariate. A GRU/TFT consumer following this handoff receives the target-generating class directly.
- XGBoost `_get_feature_columns()` excludes names containing `trajectory` (`xgboost_baseline.py:82-95`), avoiding literal direct leakage in that path. Yet it predicts a class nearly deterministically encoded by class-specific trends, so its reported AUROC mainly measures recovery of generator prototypes.
- Targets use future SpO2 only as a rare override; they do not primarily measure a future transition. This is phenotype classification repeated at timestamps, not a nontrivial trajectory forecast.

The target artifact also retains `trajectory_pattern`. The literal field must never enter model-facing data. More importantly, the generator needs a time-varying latent disease process where history and future outcomes share plausible but non-deterministic causes rather than a static class copied into both.

# 10. PATIENT/SPLIT ISOLATION AUDIT

Candidate split code is deterministic at seed 42 and patient-level: 70/15/15 subjects with disjoint IDs. For the generated data it yielded class counts train 25/24/21, validation 6/7/2, test 4/4/7 (improving/stable/deteriorating), revealing unstable class balance at only 15 validation/test subjects. No exact numeric episode clones were detected across the 100 generated episodes.

The isolation mechanism is a useful basis for `split_spec_v2`, but the assignment is recomputed in training instead of serialized/hashed/frozen. Candidate training immediately evaluates the test partition, violating final-test isolation. Preprocessing occurs before splitting. The integrated `split.py` isolation/integrity machinery is reusable, but its hard-coded era mapping is not; it requires a versioned synthetic split policy.

# 11. FINAL TEMPORAL DESIGN

No later authority changes the original repeated-cutoff design. Final operative design is therefore: synthetic ICU-like cardiac episode; first legal prediction at 24h; 6h cadence; at most 12 rows; legal through episode end minus 6h; history `(t-48h,t]`; eight right-closed 6h bins.

Candidate design is different: one grid over the first 48h from the first observed measurement, left-closed/right-open bins, trailing padding, and one-hour intermediate alignment. It has no per-cutoff examples. This is not a data-source consequence and requires rebuild under Scenario A.

# 12. FINAL TARGET DEFINITIONS

Best-supported targets remain:

1. Continuous independent recovery changes at +24h and +48h using one versioned cutoff-safe severity function.
2. `log1p` hours until the synthetic current episode ends.
3. New qualifying support initiation in `(t,t+24h]`, with incomplete negative follow-up censored.

The exact synthetic SOFA/severity calculation and support-event generator are unresolved synthetic equivalents. Candidate 24/48/72 three-class trajectory labels are **not approved**. Under Scenario B they could become the target only through an explicit scope amendment and redesigned generator.

# 13. SANSKRUTI CANDIDATE — CORRECTED ASSESSMENT

Removing obsolete MIMIC penalties improves the candidate assessment in three areas: it has a deterministic synthetic generator, a cardiac/HF focus, and a patient-level random split mechanism. It still fails the best-supported project because:

- episode boundaries are inferred from observations;
- temporal grids and prediction units differ;
- target semantics differ;
- the proxy score is not standard SOFA and is mislabeled;
- missing-value medians are fit on all subjects;
- canonical masks/TSLO/padding do not match;
- XGB receives engineered information rather than the shared canonical view;
- no support events/target exist;
- test metrics are consumed during development;
- artifact/serving contracts do not integrate.

Under Scenario B, candidate task alignment becomes closer but scientific quality remains insufficient: target-generation leakage/triviality is high, preprocessing leaks, the test split is not protected, the dataset is too small/prototypical for strong claims, and the backend returns random probabilities while marking an available model path as real.

# 14. WHICH PREVIOUS AUDIT FAILURES ARE NOW OBSOLETE

The following prior objections must not be used against Sanskruti: no MIMIC-IV v2.2 extraction; no SQL; no authorized schema check; no real item-ID validation; no MIMIC cohort count; no PhysioNet credential; lack of unrestricted adult population; missing MIMIC-Code commit as such; and absence of real-data support mappings.

They are replaced by synthetic equivalents: generator/schema provenance, cardiac cohort specification, episode fields, standard or renamed severity definition, and generated support-event mappings.

# 15. WHICH PREVIOUS AUDIT FAILURES REMAIN VALID

Still valid and decision-driving: unapproved target/clock change; incorrect history boundaries; absent legal cutoffs; all-data imputation; pre-lookback state carry; conflated missingness/padding; approximate TSLO; proxy SOFA called SOFA; wrong recovery/remaining-stay tasks; no support target; XGB/GRU information inequality; test evaluation during development; weak structural tests; no accepted manifest; no training-serving equivalence; no compatible artifact; fake backend predictions; environment/package friction; and missing lineage.

# 16. SANSKRUTI RECOMMENDATION USE / PATCH / REWORK / REBUILD

**REBUILD.** This is not because the data are synthetic or cardiac-focused. It is because the scientific core does not implement the retained current task and cannot be repaired through field adapters. Reuse generator concepts and generic QA snippets, but rebuild the generator/episode/canonical/label/XGBoost path directly against the existing receiver contracts.

Scenario B alone would warrant **SUBSTANTIAL REWORK**, not use-as-is: fix the generator/labels, train-only preprocessing, frozen split, sequence parity, artifacts, and serving. Scenario B is not currently authorized, so it does not control the final decision.

# 17. VEDANT OVERALL REUSE VERDICT

**USE WITH CONTRACT MIGRATION.** Architecture/framework reuse is **HIGH**; config migration is **MODERATE**; trained/final artifact reuse is **LOW/NONE**. The split rule, real-data acceptance wording, canonical `anchor_year_group` field, and MIMIC provenance fields require versioned synthetic variants. Timestamp, tensor shape (if Scenario A), models, training, metrics, bootstrap, search, selection, calibration, registry, test guard, reproducibility, and evidence generators are largely reusable.

# 18. VEDANT PHASE-BY-PHASE REUSE MATRIX

| Phase | Purpose | Requirement retained? | Code / tests reusable? | Artifact/config disposition | Action |
|---|---|---|---|---|---|
| 0 | Governance/hash/registry baseline | Yes | Yes/Yes | Historical baseline valid | KEEP EXACTLY |
| 1 | Prediction timestamps | Yes in Scenario A | Yes/Yes | `timestamp_spec_v1` reusable if ICU-like episode retained | USE AS-IS |
| 2 | Subject split | Isolation yes; era rule no | Integrity utilities yes; mapping no | New `split_spec_v2` and split artifact | VERSIONED CONTRACT CHANGE |
| 3 | Canonical tensor/loader | 8×F retained in A | Most/Yes | Remove era dependence through v2 adapter/schema | VERSIONED CONTRACT CHANGE |
| 4 | Training infrastructure | Yes | Yes/Yes | Smoke outputs not final | RETRAIN ONLY |
| 5 | Recovery GRU/scaler | Yes in A | Yes/Yes | Checkpoints/scalers regenerate | RETRAIN ONLY |
| 6 | ICU-time GRU/postprocess | Yes in A | Yes/Yes | Regenerate from exact synthetic episode end | RETRAIN ONLY |
| 7 | Support GRU/class weight | Yes in A | Yes/Yes | Needs synthetic support labels | NEW DATA ADAPTER + RETRAIN |
| 8 | LSTM sensitivity | Yes in A | Yes/Yes | Rerun from selected GRU configs | RETRAIN ONLY |
| 9 | Stay-balanced metrics | Yes with repeated cutoffs | Yes/Yes | Old result rows stale | KEEP CODE; REGENERATE RESULTS |
| 10 | Grouped bootstrap | Yes | Yes/Yes | Use synthetic stay clusters | KEEP CODE; REGENERATE CIs |
| 11 | Exact-30 search governance | Yes | Yes/Yes | New candidate runs/config bindings | KEEP CODE; RE-RUN |
| 12 | Per-task selection | Yes | Yes/Yes | Old/smoke decisions not final | KEEP CODE; RE-RUN |
| 13 | Support calibration/threshold | Yes | Yes/Yes | Refit validation only | KEEP CODE; REFIT |
| 14 | Recovery ablation/completeness | Yes if SOFA retained | Yes/conditional | New schema/flags required | CONFIG CHANGE + RE-RUN |
| 15 | Prespecified error analysis | Yes | Yes/Yes | Cardiac/synthetic slice definitions needed | CONFIG CHANGE + RE-RUN |
| 16 | Registry/lineage | Yes | Yes/Yes | Extend provenance vocabulary; keep history | SMALL CODE/CONFIG PATCH |
| 17 | G3/test guard | Yes | Yes/Yes | Scope name/config changes only | USE AS-IS with config migration |
| 18 | Data acceptance | Purpose yes; MIMIC checks no | Framework partly | Add synthetic acceptance v2 | VERSIONED CONTRACT MIGRATION |
| 19 | Validation orchestration | Yes | Yes/Yes | Point to new accepted bundle/runs | CONFIG CHANGE |
| 20 | Final-test evaluation | Yes | Yes/Yes | No old final result exists | USE AS-IS; run once later |
| 21 | Reproducibility | Yes | Yes/Yes | Add generator reproduction | SMALL CODE/CONFIG PATCH |
| 22 | Evidence generation | Yes | Yes/Yes | Regenerate scientific tables/limitations | CONFIG/TEXT CHANGE + REGENERATE |

# 19. VEDANT CODE VS ARTIFACT REUSE

| Layer | Code reusable? | Config reusable? | Trained artifact reusable? | Evidence reusable? |
|---|---|---|---|---|
| Clock | Yes | Yes if episode semantics retained | N/A | Structural tests yes |
| Split | Utilities yes | No; era policy obsolete | Old split no | Isolation tests yes |
| Tensor/loader | Mostly | Version bump for era/provenance | Fixture only | Structural tests yes |
| GRU/LSTM | Yes | Task configs mostly | No final checkpoints | Architecture/smoke evidence yes |
| Evaluation | Yes | Minor synthetic grouping/slices | No results | Formula evidence yes |
| Search/selection/calibration | Yes | New task/data hashes | No winners/calibrators | Governance evidence yes |
| Registry/G3/final test | Yes | Scope vocabulary changes | No final scientific state | Adversarial evidence yes |
| Reproducibility/evidence | Yes | Generator inputs added | Regenerate manifests/tables | Framework evidence yes |

# 20. VEDANT CONTRACT CHANGES REQUIRED

Create, after team approval: `project_scope_v2.md`; `cohort_spec_v2.yaml`; `split_spec_v2.yaml` using seeded subject-level assignment or approved generator-batch holdout; a split artifact/hash; and a synthetic acceptance manifest. Version `CanonicalExample` only if removing mandatory `anchor_year_group` cannot be handled by an explicit neutral source-era field. Do not fake MIMIC era strings.

Replace `real_data_input_manifest_v1` with an authorized-project-data/synthetic manifest that binds generator/schema/cohort/feature/label/split versions. Retain task fields and `[8,F]` shapes under Scenario A. Registry/lineage schemas should accept `generator_version/hash` in place of MIMIC extraction/commit fields.

# 21. PULKIT OVERALL REUSE VERDICT

**USE WITH CONTRACT MIGRATION.** Serving/product framework reuse is **HIGH** under Scenario A; data mapping/artifact reuse is **LOW**; task schema reuse is **HIGH** if targets remain. Replace MIMIC event mappings with versioned synthetic event fields, generalize “real” to authorized project data, and supply a synthetic current-severity provider. Do not duplicate state machines or PredictionPipeline.

# 22. PULKIT PHASE-BY-PHASE REUSE MATRIX

| Phase | Purpose | Retained? | Code/tests reusable? | Artifact/config | Action |
|---|---|---|---|---|---|
| 1 | Vasopressor state | Yes in A | Yes/Yes | Synthetic mapping required | USE WITH SYNTHETIC MAPPING |
| 2 | Ventilation state | Yes in A | Yes/Yes | Synthetic interval/category spec | USE WITH SYNTHETIC MAPPING |
| 3 | Composite support label | Yes in A | Yes/Yes | Regenerate labels | USE AS-IS + NEW DATA |
| 4 | Endpoint freeze | Yes | Yes/Yes | Replace MIMIC blockers with generator signoff | CONFIG/TERMINOLOGY CHANGE |
| 5 | Prediction schema | Yes in A | Yes/Yes | v1 remains if outputs unchanged | USE AS-IS |
| 6 | PredictionPipeline/artifacts | Yes | Yes/Yes | New selected bundle/checkpoints | USE AS-IS; NEW ARTIFACTS |
| 7 | History/preprocessing seam | Yes | Yes/Yes | Synthetic timeline contract/builder | NEW DATA ADAPTER |
| 8 | Explanation router | Yes | Yes/Yes | New model hashes only | USE AS-IS |
| 9 | Integrated Gradients | Yes for GRU | Yes; runtime dep missing | New trained model/baseline | USE AS-IS + ENVIRONMENT |
| 10 | TreeSHAP | Yes for XGB | Yes; runtime dep broken/missing | New XGB/feature map | USE AS-IS + ENVIRONMENT |
| 11 | FastAPI | Yes | Yes/Yes | Bundle composition changes | USE AS-IS |
| 12 | Current SOFA/recovery display | Yes if SOFA retained | Yes/Yes | Synthetic severity provider/spec | NEW DATA ADAPTER |
| 13 | Dashboard/replay | Yes | Yes/Yes | Cohort wording/metadata updates | CONFIG/TERMINOLOGY CHANGE |
| 14 | Full-stack integration | Yes | Yes/Yes | Re-run with final synthetic bundle | KEEP CODE; RE-RUN |
| 15 | Packaging/demo fixture | Yes | Yes/Yes | Final synthetic data replaces “demo only”; lock unresolved | CONFIG/DOC CHANGE |
| 16 | Release evidence | Yes | Yes/Yes | Regenerate after approved bundle/environment | KEEP CODE; REGENERATE |

# 23. PULKIT CODE VS ARTIFACT REUSE

State machines, composite labeling, cutoff truncation, pipeline orchestration, compatibility checks, explanation routing, API transport, dashboard architecture, replay, and packaging validators are reusable code. MIMIC draft dictionaries, synthetic engineering fixtures, old selected-model placeholders, model hashes, preprocessors, explanation outputs, screenshots, and release evidence are not final artifacts. `prediction_schema_v1` remains reusable only under Scenario A.

# 24. PULKIT CONTRACT CHANGES REQUIRED

Add a versioned synthetic timeline/event dictionary with point/interval semantics; generalize authorized data scope names; bind generator provenance; implement an authorized synthetic `HistoryProvider`; and supply the shared canonical builder and current severity provider. Update endpoint signoff to synthetic source semantics. Version the public prediction schema only if targets change; a data-source change alone does not justify an API break.

# 25. CURRENT CANONICAL-SCHEMA COMPATIBILITY

Candidate is incompatible. It uses `patient_id`, `admission_id`, and `bin_idx`, while the receiver requires subject/stay/cutoff identities, repeated examples, values, genuine observation mask, TSLO, padding, statics, eligibility, and three-task targets. Integrated schema also hardcodes `anchor_year_group` and validates it through MIMIC era mapping, so a clean synthetic v2 or explicit adapter is required even for a rebuilt producer.

# 26. TIMESTAMP/CUTOFF COMPATIBILITY

Candidate has no legal-cutoff generator and one fixed first-48h grid. Integrated `src/data/timestamps.py` is data-source agnostic after it receives subject/stay/intime/outtime and should be kept. Candidate generator lacks explicit `intime/outtime`; first/last measurements are not defensible substitutes. Runtime integration stops here until episode boundaries are generated.

# 27. HISTORY / FEATURE-BUILDER COMPATIBILITY

Candidate `GridBuilder` is training-only, first-observation anchored, and left-closed. Pulkit expects `history<=t -> same canonical builder -> model view`. No honest adapter can convert the current output without recomputing semantics. Rebuild one builder that consumes the synthetic timeline and legal cutoff and is injected into both training and serving.

# 28. PREPROCESSING COMPATIBILITY

Candidate is incompatible and leaky: `missing_values.py:11-15,208-213` fits medians on the entire dataset before splitting. Forward fill operates over whole admissions rather than within a cutoff lookback. Integrated training-only scaler/class-weight/target-scaler machinery is reusable. Fit every learned transformation on training subjects, persist fit IDs and hashes, then transform validation/test/serving.

# 29. SOFA / RECOVERY COMPATIBILITY

Candidate `1.0-demo` score is a 0–12, three-active-component proxy and must not be called standard SOFA. Under Scenario A, either generate all inputs needed for standard six-component SOFA and freeze a synthetic standard-SOFA specification, or formally rename/redefine the severity measure and version every recovery/model/API/display contract. The former preserves more tested code. Current `CurrentSOFAProvider` and reconstruction arithmetic remain reusable once backed by the rebuilt function.

# 30. REMAINING-STAY COMPATIBILITY

Candidate approximates remaining time by grid-bin count and derives “discharge” from last observation. This is invalid under retained semantics. The generator must emit explicit episode start/end; label is exact `log1p((episode_end-t).hours)`. Vedant ICU model/loss/postprocessor and Pulkit inverse transform remain reusable and should be retrained, not rewritten.

# 31. THIRD-TASK / ORGAN-SUPPORT COMPATIBILITY

Candidate generates no vasopressor/ventilation events and has no compatible third task. Under Scenario A, add synthetic support intervals with explicit agent/category, start/stop, rate/unit, pause/resume, invasive/NIV distinction, and boundary rules; feed them to Pulkit's existing state engines. Under Scenario B, organ support becomes unnecessary only if an approved contract retires it; Pulkit phases 1–4 and support model would then be retired, not declared wrong.

# 32. XGBOOST / GRU INFORMATION-PARITY COMPATIBILITY

Candidate fails parity: its baseline uses engineered rolling features, while the documented grid handoff differs and even recommends target `trajectory` as a static covariate. Integrated `temporal_information_views()` already defines sequence/flatten equivalence. Rebuilt XGBoost must consume that flattening only. Candidate low-level training mechanics may be reused after removing its split/evaluation/preprocessing ownership.

# 33. XGBOOST SERVING / TREESHAP COMPATIBILITY

No tracked candidate model satisfies Pulkit's manifest, feature-order, preprocessing, task-output, hash, and route contract. Metrics/PNGs are not models. Retrain after data acceptance and emit the governed selected artifact. Pulkit TreeSHAP adapter architecture remains reusable; current environment lacks a working XGBoost/OpenMP runtime and SHAP, which is an environment blocker rather than a semantic defect.

# 34. PREDICTIONPIPELINE COMPATIBILITY

Pipeline orchestration is reusable unchanged under Scenario A. Candidate backend must not be merged. Its `prediction_service.py:37-51` creates random probabilities from a patient-ID hash and marks the model path `is_real_model: true`, despite not loading/inferencing the model. This violates truthfulness and artifact integrity.

# 35. API SCHEMA COMPATIBILITY

Under Scenario A, keep `prediction_schema_v1` and FastAPI transport. Only metadata/scope language changes. Under Scenario B, introduce `prediction_schema_v2` for three horizon-class distributions; recovery reconstruction, ICU-time, and support fields cannot be silently repurposed. API framework remains reusable even then.

# 36. DASHBOARD / REPLAY COMPATIBILITY

Under Scenario A, architecture reuse is high: update population/data-source labels and populate it with the final synthetic catalog/bundle. Under Scenario B, replay controller/transport/layout remain, but task panels and view models need a versioned migration; recovery/remaining-stay/support panels are retired or replaced. Candidate React frontend should remain separate rather than replacing the tested dashboard.

# 37. EXPLAINABILITY COMPATIBILITY

Family routing is fully data-source agnostic. Target descriptors and output indices must match final tasks, and attribution baselines/reference features must be regenerated. No clinical causal meaning should be inferred from synthetic attributions. Keep router/IG/TreeSHAP code; resolve Captum/SHAP/XGBoost dependencies in the final lock.

# 38. METRICS / BOOTSTRAP / CALIBRATION COMPATIBILITY

Scenario A retains all code and estimands. Repeated cutoffs justify stay-balanced weighting and stay-cluster bootstrap. Support calibration/threshold remains validation-only. Scenario B would replace task metrics with prespecified multiclass measures and likely calibration, while grouping is patient-level because rows repeat. Governance utilities remain reusable; metric contracts require versioning.

# 39. REGISTRY / ARTIFACT GOVERNANCE COMPATIBILITY

Registry, immutable hashes, dependency graph, selection lineage, and final-test guard should remain. Extend lineage from MIMIC fields to generator/schema/config/seed hashes and replace real/synthetic-smoke binary language with authorized-final-synthetic versus engineering-fixture scope. Never promote old smoke/demo runs merely because synthetic data are now final; authorized final data must still be versioned and frozen.

# 40. EXISTING TEST RESULTS

| Suite | Isolated command context | Result |
|---|---|---|
| Integrated | Parent cwd; `PYTHONPATH=src:tests:.`; cache disabled; bytecode in temporary `/tmp` | **906 passed, 2 skipped**, exit 0, 11.00s |
| Candidate | Candidate cwd; `PYTHONPATH=.`; cache disabled; bytecode in temporary `/tmp` | **8 collection errors, 0 executed**, exit 2, 2.29s |

Integrated skips: Captum unavailable; genuine SHAP/XGBoost execution unavailable. Candidate failures: Python 3.9 cannot evaluate `str | Path` in `helpers.py`; XGBoost cannot load `libomp.dylib`. This is packaging/integration friction, not evidence that every candidate test would fail scientifically. Assertion review from the prior/current audit still shows that many tests prove candidate semantics rather than retained invariants.

# 41. TEMPORARY INTEGRATION PROBE RESULTS

The generator was executed in memory with stubbed logger/filesystem helpers; no output file was written. Repeat seed equality, dataset summaries, duplicate counts, clone fingerprints, label eligibility, latent-label agreement, and deterministic split composition were measured.

The end-to-end probe stopped at the first honest semantic boundary: candidate data do not contain explicit episode `intime/outtime`, so they cannot enter `RetainedICUStay` without treating first/last measurements as boundaries. That assumption would change remaining-stay and cutoff eligibility. Candidate also lacks a cutoff-aware canonical builder. No fake adapter was created. Lower-layer inspection confirms integrated synthetic history truncation and canonical provider are injection-based and reusable once a valid timeline/builder exists.

# 42. SCIENTIFIC CLAIM LIMITATIONS FOR SYNTHETIC DATA

Final results demonstrate software methodology on one designed synthetic generator only. They are not externally validated clinical performance, bedside accuracy, treatment utility, causal inference, or evidence that the system generalizes to real cardiac patients. High candidate AUROC is expected from strongly separated generator prototypes and cannot support clinical claims.

Candidate disclaimers correctly say demo data are synthetic and not for clinical decisions. They must be updated because synthetic is now the final academic dataset, while preserving the nonclinical limitation. README's claim that MIMIC is the project dataset is stale. Reports must state that MIMIC validation was impossible because authorization was not granted.

# 43. SECURITY / RELEASE-HYGIENE FINDINGS

No MIMIC credential/token or confirmed real patient data was found. Synthetic data greatly lowers privacy risk. Candidate nevertheless tracks `backend/auth.db`, contains demo credentials and a fallback development JWT secret, and can create WAL sidecars on import. These are release issues. The backend's random “real model” flag is also a trust/safety problem. Do not publish those defaults or the mutable database.

# 44. CONTRACT-MIGRATION IMPACT GRAPH

```text
MIMIC -> final synthetic source
  -> project_scope_v2 + synthetic_generator/schema manifests
  -> cohort/episode spec v2
  -> synthetic split_spec_v2 + split artifact
  -> canonical schema adapter/v2 (remove era fiction)
  -> train-only preprocessors
  -> all model training/search/selection
  -> support calibration/threshold
  -> selected-model manifest
  -> serving composition/explanations
  -> API/dashboard evidence and release report

MIMIC SOFA provenance -> synthetic standard-SOFA spec
  -> sofa_at(T) + six flags
  -> recovery labels/current severity
  -> recovery training/selection
  -> reconstruction, explanations, panels, evidence

MIMIC support mappings -> synthetic interval-event dictionary
  -> vaso/vent state artifacts
  -> composite labels/eligibility
  -> support training/selection/calibration/threshold
  -> manifest, API, replay, evidence

Synthetic split policy change
  -> split hash
  -> fitted preprocessors/class weights/target scalers
  -> every checkpoint/prediction/result
  -> selection/calibration/threshold
  -> manifest/G3 authorization/evidence
```

If Scenario B is approved, add: `label_spec_v2 -> model heads/losses/metrics -> selection/calibration -> prediction_schema_v2 -> API/view models/dashboard`, while keeping encoders, training engine, registry, pipeline orchestration, and transport.

# 45. MINIMUM SAFE INTEGRATION PLAN

1. Freeze `project_scope_v2.md`, including cardiac population, ICU-like episode, retained tasks/clock, and synthetic limitations.
2. Redesign/version the generator: explicit episode boundaries, cardiac inclusion/subtypes, nontrivial latent dynamics, support events, severity inputs, missingness/censoring, provenance, and larger sample size justified by analysis.
3. Freeze `synthetic_schema_v1.json`, `synthetic_generator_v1.json`, and `cohort_spec_v2.yaml`.
4. Create `split_spec_v2.yaml` and immutable subject-level split using existing integrity/governance utilities.
5. Build one cutoff-aware canonical producer targeting the existing eight-bin information contract; version only fields that truly change.
6. Implement standard synthetic SOFA or approve a renamed severity v2; generate exact recovery/episode-time/support labels.
7. Prove future perturbation, strict lookback, padding/missingness, TSLO, target leakage, subject isolation, parity, and training-serving equality.
8. Adapt Sanskruti XGBoost to `temporal_information_views()`; do not reuse its random split/test evaluation.
9. Retain Vedant/Pulkit frameworks; retrain exact searches, select, calibrate, and regenerate manifests/evidence.
10. Resolve and lock Python/XGBoost/SHAP/Captum environment; run one governed final test only after freeze.

# 46. FILE-BY-FILE SANSKRUTI MERGE MAP

| Candidate path | Revised-scope disposition |
|---|---|
| `src/data/demo_data.py` | **REUSE AFTER SUBSTANTIAL FIX** as generator seed; redesign dynamics/events/boundaries/provenance |
| `src/data/cohort_builder.py` | **REBUILD** synthetic cardiac/episode cohort; remove MIMIC path from final route |
| `src/data/load_data.py` | **REBUILD/RETIRE** demo fallback; final synthetic loader must fail closed on manifest |
| `src/data/extract_features.py` | **REBUILD** as generator-schema normalizer, not MIMIC extractor |
| `src/data/validate_data.py` | **ADAPT** generic range/duplicate ideas |
| `src/preprocessing/time_alignment.py` | **REBUILD** only if canonical event normalization needs it |
| `src/preprocessing/missing_values.py` | **DO NOT USE**; all-data fit and wrong cutoff boundary |
| `src/preprocessing/clean_data.py` | **ADAPT** with stable provenance/order |
| `src/preprocessing/feature_engineering.py` | **DO NOT USE** in parity path |
| `src/preprocessing/grid_builder.py` | **REBUILD** against legal cutoff/history contract |
| `src/preprocessing/sofa.py` | **DO NOT USE** as SOFA; concepts only if score renamed |
| `src/targets/generate_targets.py` | **REBUILD** for Scenario A; substantial redesign for B |
| `src/models/xgboost_baseline.py` | **REUSE LOW-LEVEL WRAPPER IDEAS ONLY**; remove split/preprocessing/test ownership |
| `src/models/xgboost_validation_search.py` | **DO NOT USE**; use Vedant governance |
| `run_pipeline.py`, `run_sofa_pipeline.py` | **REBUILD** as contract-driven final synthetic entrypoint |
| `tests/` | **KEEP SEPARATE; ADAPT FIXTURE IDEAS**; assertions need retained semantics |
| `data/**/*.parquet` | **STALE SCIENTIFICALLY; REGENERATE** after generator freeze |
| `outputs/` | **DO NOT USE AS FINAL EVIDENCE** |
| `backend/`, `frontend/` | **DO NOT MERGE**; Pulkit stack is authoritative |
| `config/clinical_ranges.yaml` | **ADAPT/VERIFY** for synthetic generator spec |
| `config/config.yaml`, `features.yaml` | **REPLACE WITH VERSIONED APPROVED CONTRACTS** |
| `docs/FEATURE_BUILDER.md` | **DO NOT USE**; it explicitly exposes `trajectory` as input |
| `README.md`, target/SOFA docs | **REWRITE AFTER SCOPE FREEZE**; current MIMIC/demo terminology is stale |

# 47. EXISTING VEDANT/PULKIT FILES THAT MUST NOT BE REWRITTEN

Preserve and extend through adapters/configuration: `src/data/timestamps.py`; most validation/flattening logic in `src/data/schema.py`; `src/data/dataset.py` and `collate.py`; `src/models/gru*.py`, `lstm.py`, task-neutral training engine/checkpoint/early stopping; evaluation weights/metrics/bootstrap/search/selection/calibration/threshold; registry/lineage/G3/final-test guard; support/ventilation/composite state engines; `src/serving/pipeline.py`, artifact resolver, preprocessing injection boundary; explanation router/IG/TreeSHAP; FastAPI transport; dashboard replay architecture; reproducibility/evidence tooling.

Do not overwrite v1 files to hide changed semantics. Add reviewed v2 contracts/adapters and retain v1 history.

# 48. EXISTING FILES THAT REQUIRE VERSIONED CHANGE

Likely changes: `configs/split_spec_v1.yaml` successor and `src/data/split.py` policy abstraction; `configs/tensor_contract_v1.json`/`src/data/schema.py` only for removal of mandatory era fields; `src/data/real_adapter.py` successor for authorized synthetic data; `configs/event_dict_v1.yaml` successor; acceptance/G3 configs that say real/MIMIC; registry/lineage provenance fields; SOFA/recovery source metadata; dashboard/API metadata terminology; packaging README/runbook/limitations.

`configs/prediction_schema_v1.json` changes only if tasks change. `configs/timestamp_spec_v1.yaml` should remain unchanged under Scenario A.

# 49. ARTIFACTS THAT MUST BE REGENERATED

Final synthetic dataset and manifest; cohort/episode index; split artifact; feature schema/data dictionary; canonical examples; severity/SOFA and support-event artifacts; all labels/eligibility; train-only preprocessors/scalers/class weights; all XGB/GRU/LSTM checkpoints; search results; validation predictions; selection; calibrator/threshold; selected manifest; explanations; metric/bootstrap/ablation/error tables; registry entries; G3/G4 state; API/dashboard fixture; release/reproducibility/evidence reports.

# 50. ARTIFACTS THAT REMAIN VALID

Structurally valid: versioned code-level contracts/tests that do not encode MIMIC era/source; synthetic engineering fixtures as tests only; phase review history; hash/registry schemas and adversarial evidence; model architecture descriptions; API/dashboard transport tests; packaging privacy controls; previous audit records. None of these becomes a final scientific result automatically.

# 51. EXACT COMMANDS EXECUTED

Principal commands included:

```bash
sed -n '1,2200p' /Users/Home/.codex/attachments/38493c37-e408-42df-9bb1-63d8e9af90bb/pasted-text.txt
find . -name .git -type d -prune -print
git -C Heart-Failure-Recovery-and-Deterioration-Trajectory-Prediction status --short --branch
git -C Heart-Failure-Recovery-and-Deterioration-Trajectory-Prediction rev-parse HEAD
git -C Heart-Failure-Recovery-and-Deterioration-Trajectory-Prediction log --all --date=iso --format='%H%x09%ad%x09%s'
git -C Heart-Failure-Recovery-and-Deterioration-Trajectory-Prediction show 66b3376:docs/TEAM_HANDOFF.md
git -C Heart-Failure-Recovery-and-Deterioration-Trajectory-Prediction show 66b3376:docs/MODEL_STATUS.md
rg -n -i '(synthetic|authorization|heart disease|heart failure|cardiac|revised scope|scope change)' .
rg -n -i '(MIMIC|PhysioNet|anchor_year_group|2008-2010|2017-2019)' src configs api dashboard
nl -ba Heart-Failure-Recovery-and-Deterioration-Trajectory-Prediction/src/data/demo_data.py
nl -ba Heart-Failure-Recovery-and-Deterioration-Trajectory-Prediction/src/targets/generate_targets.py
nl -ba Heart-Failure-Recovery-and-Deterioration-Trajectory-Prediction/src/preprocessing/grid_builder.py
nl -ba Heart-Failure-Recovery-and-Deterioration-Trajectory-Prediction/src/preprocessing/missing_values.py
nl -ba Heart-Failure-Recovery-and-Deterioration-Trajectory-Prediction/src/models/xgboost_baseline.py
PYTHONPYCACHEPREFIX="$tmp_parent" PYTHONPATH=src:tests:. python3 -m pytest tests -q -rs -p no:cacheprovider
PYTHONPYCACHEPREFIX="$tmp_candidate" PYTHONPATH=. python3 -m pytest -q -rs -p no:cacheprovider
python3 --version
python3 -c 'import importlib.util; ...'
```

Two in-memory Python probes executed the candidate generator and target generator without writing outputs. They calculated dataset shape, episode spans, missingness, duplicates, reproducibility, clone fingerprints, label distributions/eligibility, latent-label agreement, and split composition. Standard read-only `find`, `rg`, `sed`, `nl`, `file`, `stat`, `shasum`, and SQLite schema/count inspection from the preceding acceptance audit were also relied upon. No package was installed; no training or final-test access occurred.

# 52. TEST COUNTS / FAILURES / SKIPS

- Integrated: **906 passed, 0 failed, 2 skipped, 0 xfailed; exit 0; 11.00s**.
- Candidate: **0 executed, 8 collection errors; exit 2; 2.29s**.
- Candidate statically contains 57 test functions; collection failure prevents a pass claim.
- Integrated skips: Captum dependency; SHAP/XGBoost runtime.
- Candidate errors: seven import paths encounter Python 3.9's unsupported evaluated `str | Path`; one XGBoost import encounters missing `libomp.dylib` (with overlap in affected modules).
- Python upgrade or adding `from __future__ import annotations`/`Union` is a small packaging patch. A final environment change must rerun all 908 integrated outcomes; it is not a scientific rebuild.

# 53. FINAL ANSWER: "SHOULD WE USE SANSKRUTI'S WORK?"

**Not as the final integrated implementation. Recommendation: REBUILD.** Reuse selected ideas from the deterministic generator, clinical ranges, QA, and XGBoost wrapper, but do not merge its cohort/grid/SOFA/target/preprocessing/split/backend paths. The answer remains rebuild after removing all obsolete MIMIC objections because retained task semantics and leakage controls still fail.

# 54. FINAL ANSWER: "CAN WE REUSE VEDANT'S WORK?"

**YES — architecture/framework reuse HIGH; trained artifact reuse LOW; contract migration MODERATE.** Keep models, training, evaluation, governance, registry, test isolation, and evidence infrastructure. Add a synthetic split policy, authorized-data acceptance path, generator provenance, and limited canonical metadata migration; retrain everything scientific.

# 55. FINAL ANSWER: "CAN WE REUSE PULKIT'S WORK?"

**YES — serving/product framework reuse HIGH; MIMIC mapping reuse NONE; task-specific schema reuse CONDITIONAL on Scenario A.** Keep state engines, PredictionPipeline, artifact checking, truncation/preprocessing seam, explanations, API, dashboard/replay, integration, and packaging. Supply synthetic event mappings/current severity/final artifacts and update terminology.

# 56. FINAL ANSWER: "WHAT IS THE SHORTEST SAFE PATH TO PROJECT COMPLETION?"

Freeze the minimal scope amendment; rebuild only the upstream synthetic data truth layer to emit the existing repeated-cutoff canonical contract; version the synthetic split/provenance fields; retain Vedant/Pulkit infrastructure; retrain/search/select/calibrate; then regenerate all scientific and serving artifacts. Do **not** adopt candidate 24/48/72 deterioration tasks without a separate explicit decision, and do not rewrite tested downstream frameworks merely because the source is synthetic.

# 57. NEXT ACTION

Before coding or training, obtain team signoff on a one-page `project_scope_v2` checklist covering: final title; heart disease versus heart failure; synthetic-only statement; ICU-like episode and first-episode policy; retained or changed tasks/horizons; history/cutoffs; standard SOFA versus renamed severity; remaining-stay endpoint; third task; split; feature/mask/TSLO contract; metrics/models/calibration; API/replay semantics; and nonclinical limitations.

Once Scenario A is confirmed, implement the generator/cohort/schema/split/label contracts and run the existing receiver acceptance tests. This is the smallest path that preserves correct work without preserving stale science.
