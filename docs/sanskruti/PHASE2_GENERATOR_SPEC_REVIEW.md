# Sanskruti Rebuild Phase 2 — Synthetic Cardiac Generator Specification Review

## 1. SANSKRUTI REBUILD PHASE 2 SUMMARY

Phase 2 specifies a deterministic, auditable, event-level synthetic cardiac generator without implementing it. The package contains a serialized process specification, format-independent raw schema, manifest template, field-role matrix, leakage threat model, specification validator, tests, and integrity hashes. No patient rows, cohort, canonical examples, labels, split, or model artifacts were created.

## 2. PHASE-1 PREREQUISITE STATUS

**SATISFIED FOR DRAFT SPECIFICATION WORK.** Human/machine Scope v2, delta, impact matrix, hashes, and Phase-1 review exist. Phase 1 expressly permits Phase-2 specification work while retaining review-required status.

## 3. PROJECT_SCOPE_V2 VERSION / STATUS

Version `PROJECT_SCOPE_V2`; status `DRAFT_COMPLETE_TEAM_FREEZE_REQUIRED`; approval `REVIEW_REQUIRED`; no approved reviewers recorded. Phase-2 artifacts inherit draft/review-required status.

## 4. REPOSITORY STATE BEFORE CHANGES

Integrated root was not a Git repository. Historical candidate was clean on `main` at `dc95164b808becd6e822bccd28c0135303753e40`. It remained read-only.

## 5. BASELINE INTEGRATED TEST RESULT

`921 passed, 2 skipped, 0 failed, 0 errors in 11.43s`. Skips were existing optional Captum and SHAP/XGBoost integrations.

## 6. OLD SANSKRUTI GENERATOR AUDIT

The candidate uses `np.random.default_rng`, irregular 0.5–2h observations, per-variable noise/missingness, bounded values, variable durations, deliberate duplicates, and one episode per subject. It first assigns one of three trajectories, uses class-specific baselines/slopes, exports `trajectory`, and downstream reuses it as target. It has no explicit structural episode bounds, support intervals, six-component SOFA inputs, governed provenance, or retained repeated-cutoff tasks.

## 7. REUSABLE OLD-GENERATOR IDEAS

Local seeded RNG, irregular timing, variable-specific observation density, synthetic disclosure, bounded domains, one core episode per subject, and separate measurement noise are reusable only after contract redesign.

## 8. REJECTED OLD-GENERATOR IDEAS

Public trajectory/outcome class, three fixed templates, class-derived targets, inferred admission/discharge, first-48h-only grid, 24/48/72 classes, final size 100 by convenience, ambiguous duplicates, global preprocessing, and unsupported realism claims.

## 9. GENERATOR SPEC PATH

`configs/synthetic/synthetic_generator_v1.yaml`; human explanation at `docs/sanskruti/SYNTHETIC_GENERATOR_SPEC_V1.md`.

## 10. GENERATOR SPEC VERSION

Scientific version `synthetic_generator_v1`; serialized config version `synthetic_generator_config_v1`; status `DRAFT_SPECIFICATION_REVIEW_REQUIRED`.

## 11. GENERATOR SPEC SHA-256

Config: `5f7d3f73572a34282c214860f1c0c656b18727a16fa0c4687caf071f690a7484`. Human spec: `a069de5db5c4088fa5102657c4919b2618d3586e9b9dcb453fb810ed87df7b16`.

## 12. RAW SCHEMA PATH

`configs/synthetic/synthetic_schema_v1.json`.

## 13. RAW SCHEMA VERSION

`synthetic_raw_schema_v1`; status `DRAFT_SPECIFICATION_REVIEW_REQUIRED`; artifact status `SPECIFICATION_ONLY_NO_DATA_GENERATED`.

## 14. RAW SCHEMA SHA-256

`85751b498c7529f465ace6c07fae064902f7f06d8e243000d59de31c86a0b51c`.

## 15. POPULATION DOMAIN

Adult cardiac/heart-disease-focused synthetic benchmark cohort. It is neither unrestricted ICU nor a claim of coverage across all cardiovascular disease.

## 16. CARDIAC SUBTYPE STATUS

`REVIEW REQUIRED — CARDIAC SUBTYPE TAXONOMY AND MIXTURE`. `cardiac_condition_group` is reserved in the schema; exact values/proportions are null and cannot be represented as epidemiologic prevalence.

## 17. ADULT POLICY

Minimum age 18 is frozen. Maximum and synthetic age distribution remain unlocked. Pediatrics are excluded.

## 18. SUBJECT IDENTITY CONTRACT

Opaque deterministic `SYN-S-{ordinal}` identity, no real identifiers/names/dates, no target/subtype/split token, stable regeneration, and hard-fail collision policy.

## 19. EPISODE IDENTITY CONTRACT

Opaque deterministic `SYN-E-{ordinal}` stay identity linked one-to-one to the core subject. Exactly one core episode is specified.

## 20. INTIME / OUTTIME CONTRACT

Both are generated structural truths, UTC-aware, with `outtime > intime`; never inferred from observations. `outtime` is structural/label-only and prohibited from model features.

## 21. EPISODE-DURATION DESIGN

Truncated log-normal duration driven probabilistically by initial hidden burden, support propensity, and independent noise. Exact coefficients/bounds remain review-required. Variable durations must enable natural cutoff/horizon/censoring variation without prevalence quotas.

## 22. TIME / TIMESTAMP POLICY

UTC ISO-8601 with microsecond capability, deterministic synthetic epoch in 2100, stable ordering, events bounded to the episode, no pre-episode data, no DST ambiguity, no calendar features/split, and no fake MIMIC era.

## 23. LATENT-STATE DESIGN

Five continuous factors—systemic severity, cardiac stress, respiratory, renal, and neurologic burden—follow a coupled mean-reverting stochastic process with patient effects, drift, sparse shocks, shared/domain-specific innovations, and bounded delayed intervention effects. No outcome class exists.

## 24. TARGET-LEAKAGE PREVENTION DESIGN

Hidden state is nonexported; model fields use an allowlisted role vocabulary; structural/provenance fields are excluded; labels/splits/future summaries are prohibited; future rows are cutoff-truncated; support cessation is availability-aware; and post-performance generator editing is forbidden.

## 25. PHYSIOLOGICAL VARIABLE DOMAINS

Fifteen raw concepts are specified. Mandatory domains cover SOFA inputs; candidate critical-care concepts include heart rate, systolic/diastolic pressure, respiratory rate, oxygen saturation, temperature, and lactate. Domains are synthetic safety bounds, not clinical references.

## 26. SIX-COMPONENT SOFA INPUT COVERAGE

Respiratory: PaO2, FiO2, ventilation context. Coagulation: platelets. Liver: bilirubin. Cardiovascular: MAP and vasopressor rate/context. CNS: GCS. Renal: creatinine and interval urine-output volume. Scoring remains Phase 6.

## 27. STATIC VARIABLE DESIGN

Restricted to age, reviewed sex/gender category, and reviewed cardiac condition group. No future-severity, duration, support-need, trajectory, discharge, or recovery statics.

## 28. OBSERVATION PROCESS

Variable-specific severity-modulated renewal process with local-RNG exponential waits, different base rates, subject effects, and time-varying density. It is an auditable benchmark mechanism, not a hospital-workflow claim.

## 29. MISSINGNESS PROCESS

Missingness is absence of an observation row. Current hidden severity may influence observation rate, but no future class controls deletion. No null placeholder events, imputation, filling, normalization, masks, or TSLO are generated.

## 30. MEASUREMENT NOISE POLICY

Observed value is a bounded link-scale transformation of true physiology plus subject bias and random measurement error. Invalid/nonfinite outputs hard-fail; silent clipping cannot hide defective parameters.

## 31. SUPPORT-EVENT DESIGN

Stochastic current/recent-state-dependent OFF→ON and ON→OFF hazards produce genuine closed-open intervals, with patient effects and innovations. Intervals may span cutoffs; deterministic visible thresholds and independent coin flips are prohibited.

## 32. VASOPRESSOR CONTRACT

Interval `agent_key`, start/end, segmented rate/value and unit, explicit qualifying mapping, and adapter target `NormalizedActiveInterval`. Exact final vocabulary/units require review.

## 33. VENTILATION CONTRACT

Categories `INVASIVE`, `NON_INVASIVE`, `HFNC`, and `ORDINARY_OXYGEN`; only invasive qualifies; unknown categories hard-fail; adapter target `NormalizedVentilationInterval`.

## 34. CENSORING / FOLLOW-UP CAPABILITY

Variable episode ends and timed support intervals allow full negative follow-up, early-exit censoring, continuation at cutoff, and positive initiation before exit. The generator emits no eligibility or censor label.

## 35. EPISODE-END PROCESS

`outtime` is sampled as a structural property before event export, probabilistically related to hidden burden/noise, not measurement cessation or a deterministic public input. Active support is truncated at episode end.

## 36. RAW TABLE / ARTIFACT LAYOUT

Exported: subjects, episodes, long-form raw events, support intervals, and future dataset manifest. Nonexported: optional internal latent audit. One storage format will be selected later; no database server is required.

## 37. FIELD-ROLE MATRIX

`docs/sanskruti/SYNTHETIC_FIELD_ROLE_MATRIX_V1.csv` contains all 37 schema fields exactly once, with type/unit/generator/timing/role/risk/consumer/status.

## 38. MODEL-ELIGIBLE FIELD POLICY

Only `MODEL_ELIGIBLE_RAW` observed facts and `STATIC_MODEL_ELIGIBLE` reviewed statics may reach the future canonical builder, and only when genuinely available by the cutoff. Phase 5/7 still decide feature inclusion/order.

## 39. PROHIBITED FIELD POLICY

Targets, labels, trajectory/outcome/recovery/deterioration classes, future state/SOFA/support, remaining duration, total duration as feature, eligibility/censor reason, latent state, split identity, final outcome, and prediction outcome are prohibited.

## 40. RNG / REPRODUCIBILITY POLICY

Local NumPy `Generator(PCG64)` only; final primary seed null/unlocked; deterministic `SeedSequence([seed, subject ordinal, component code])` substreams; no global mutation or call-order dependence; bind code/config/schema/environment and canonical ordering.

## 41. GENERATOR PROVENANCE CONTRACT

Requires scope version/hash, generator version/code identity, config version/hash, schema version/hash, seed policy/identity, Python/environment, execution timestamp/command, requested/generated subjects, record counts, artifact paths/formats/counts/hashes.

## 42. DATASET MANIFEST CONTRACT

Template `configs/synthetic/synthetic_dataset_manifest_template_v1.json`, SHA-256 `3b9b14b63d18083f1a236ce1a496f1a698358f609202bae2781eaf772b435fa7`. It contains null execution values and is not a produced dataset manifest.

## 43. VERSIONING POLICY

Trajectory-changing parameters create a new config hash and full regeneration. Scientific-process changes require generator version review/bump. Field/semantic changes require schema review/bump. Cosmetic documentation alone does not.

## 44. FINAL COHORT-SIZE STATUS

`UNLOCKED — FINAL SYNTHETIC COHORT SIZE`. No number or recommended range was invented without compute/statistical analysis.

## 45. FINAL GENERATOR-SEED STATUS

`UNLOCKED — FINAL GENERATOR PRIMARY SEED`. The API field and deterministic derivation policy are specified.

## 46. STORAGE-FORMAT STATUS

`UNLOCKED ENGINEERING PARAMETER`. One of Parquet, Arrow IPC, CSV, or JSONL must be selected after environment/size/typing/determinism review.

## 47. VEDANT INTEGRATION MAPPING

Subject/stay identities map directly; episode bounds instantiate `RetainedICUStay`; raw timeline feeds the future shared builder; statics feed canonical statics; full episode supports horizon derivation; hashes feed acceptance/registry. Generator creates neither timestamps nor split.

## 48. PULKIT INTEGRATION MAPPING

Long-form events feed a future authorized HistoryProvider; event time enables truncation; support intervals feed existing state engines through adapters; SOFA raw inputs feed future CurrentSOFAProvider; provenance feeds compatibility. Generator has no serving-specific path.

## 49. LEAKAGE THREAT MODEL

`docs/sanskruti/SYNTHETIC_GENERATOR_LEAKAGE_THREAT_MODEL_V1.md` enumerates 30 threats, prevention controls, test owners, and executable phases. SHA-256 `5906e326d57c02579da9c04c582ea02d049f68c52529973d7f9c36c8ef83d89d`.

## 50. FILES CREATED

- Three machine configs/schema/template under `configs/synthetic/`.
- Specification, threat model, field matrix, review, and hash manifest under `docs/sanskruti/`.
- `src/data/synthetic/__init__.py` and specification validator `spec.py`.
- `tests/test_synthetic_generator_spec.py`.

## 51. FILES MODIFIED

`AGENTS.md` received one discovery pointer to the draft Phase-2 specification and an explicit statement that it does not authorize generation. No Vedant/Pulkit/runtime scientific module was modified.

## 52. TESTS ADDED

Twenty-eight focused tests cover artifact identity, versions/status, scope binding, table relationships, explicit bounds, long-form events, downstream separation, prohibited fields, SOFA/support coverage, RNG, unresolved parameters, manifest, role matrix, machine-readable nonclinical limitations, absence of MIMIC/model dependencies, threat model, import isolation, no Phase-3/data artifacts, and hostile mutations.

## 53. FOCUSED TEST RESULTS

`28 passed in 0.64s`.

## 54. FULL REGRESSION RESULTS

`949 passed, 2 skipped, 0 failed, 0 errors in 11.52s`. The two skips remain the optional Captum and SHAP/XGBoost integrations.

## 55. EXACT COMMANDS EXECUTED

Core verification commands were:

```bash
PYTHONPYCACHEPREFIX=<temporary-directory> PYTHONPATH=src:tests:. python3 -m pytest tests -q -rs -p no:cacheprovider
PYTHONPYCACHEPREFIX=<temporary-directory> PYTHONPATH=src:tests:. python3 -m pytest tests/test_synthetic_generator_spec.py -q -p no:cacheprovider
PYTHONPYCACHEPREFIX=<temporary-directory> PYTHONPATH=src:tests:. python3 -m compileall -q src api dashboard tests
PYTHONPATH=src:. python3 -c "from pathlib import Path; from data.synthetic import validate_phase2_specification; print(validate_phase2_specification(Path('.')))"
python3 -m json.tool configs/synthetic/synthetic_schema_v1.json
shasum -a 256 configs/synthetic/synthetic_generator_v1.yaml configs/synthetic/synthetic_schema_v1.json configs/synthetic/synthetic_dataset_manifest_template_v1.json docs/sanskruti/SYNTHETIC_GENERATOR_SPEC_V1.md docs/sanskruti/SYNTHETIC_GENERATOR_LEAKAGE_THREAT_MODEL_V1.md docs/sanskruti/SYNTHETIC_FIELD_ROLE_MATRIX_V1.csv
git -C Heart-Failure-Recovery-and-Deterioration-Trajectory-Prediction status --short
```

Repository inspection also used `pwd`, bounded `find`, `rg`, `sed`, JSON/YAML parsing, CSV consistency checks, and read-only Git revision checks.

## 56. UNLOCKED PARAMETERS

Final cohort size, primary seed, age maximum/distribution, primary storage format/output location, package manager, and final environment.

## 57. REVIEW-REQUIRED PARAMETERS

Cardiac taxonomy/mixture, sex/gender vocabulary/distribution, duration parameters, latent matrices/distributions, true physiology parameters, observation rates/loadings, measurement noise, support vocabulary/units/hazards/effects, and exact final serialization decision.

## 58. DEFINITION-OF-DONE CHECKLIST

- [x] Phase-1 scope inspected and no scientific conflict found.
- [x] Baseline tests recorded and old generator audited read-only.
- [x] Versioned generator process and raw schema specified.
- [x] Subject/episode/event/support/manifest structures defined.
- [x] Explicit `intime/outtime`, one episode, event time, and UTC policy defined.
- [x] Continuous hidden dynamics never model-eligible.
- [x] Generator labels/split/preprocessing/binning/canonical examples prohibited.
- [x] Six SOFA domains and Pulkit support prerequisites covered.
- [x] Observation, missingness, noise, episode-end, RNG, provenance, and change control specified.
- [x] Field-role matrix, leakage threat model, validator, and hostile tests created.
- [x] No arbitrary prevalence gate, cohort size, split, or seed chosen.
- [x] Focused tests pass.
- [x] Full regression passes.
- [x] Compile/import checks pass.
- [x] No final data, model training, or Phase-3 implementation.

## 59. PHASE 2 IMPLEMENTATION STATUS

`IMPLEMENTED AND TESTED` for the specification/validator scope. Scientific freeze remains review-required.

## 60. PHASE 3 READINESS

`CONDITIONAL`. Architecture/schema are implementation-ready; scientific scalar/vocabulary parameters and team review must be resolved before final Phase-3 execution.

## 61. EXPLICIT CONFIRMATION: FINAL DATASET WAS NOT GENERATED

**CONFIRMED.** No final subject, episode, event, support, cohort, or manifest artifact was generated.

## 62. EXPLICIT CONFIRMATION: NO MODEL TRAINING WAS PERFORMED

**CONFIRMED.** No training, tuning, selection, calibration, prediction, or metric execution occurred.

## 63. EXPLICIT CONFIRMATION: PHASE 3 WAS NOT IMPLEMENTED

**CONFIRMED.** No generator runtime, dataset command, or final-data writer exists in the integrated project.
