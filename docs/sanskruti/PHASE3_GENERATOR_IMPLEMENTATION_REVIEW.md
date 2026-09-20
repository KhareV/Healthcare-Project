# Sanskruti Rebuild Phase 3 — Generator Implementation Review

## 1. SANSKRUTI REBUILD PHASE 3 SUMMARY

PASS for generator implementation, validation, reproducibility, provenance, and explicitly non-scientific fixture/smoke evidence. BLOCKED for an authorized final raw dataset because the Phase-2 scientific parameters and team freeze remain unresolved. This phase emits only raw subjects, episodes, observations, an empty support extension artifact, manifests, and descriptive QA.

## 2. PHASE-1 PREREQUISITE STATUS

PASS. `docs/CODEX_PROJECT_CONTEXT_V2.md`, `docs/governance/project_scope_v2.md`, and `configs/governance/project_scope_v2.json` were inspected. The machine-readable scope SHA-256 is `4fcab7045b30b631bdbb9e8219226c93288d6ef1c2550125d3456ab431d04994`.

## 3. PHASE-2 PREREQUISITE STATUS

PASS. The generator YAML, raw schema JSON, manifest template, human specification, leakage threat model, field-role matrix, Phase-2 review, and Phase-2 hash manifest exist and validate. No `SCIENTIFIC SCOPE CONFLICT` declaration was found.

## 4. PROJECT_SCOPE_V2 STATUS

Complete draft pending genuine team freeze; file presence is not approval. The implementation preserves retrospective event time, adult synthetic cardiac scope, one ICU-like episode per subject, UTC dates, and downstream ownership boundaries.

## 5. SYNTHETIC GENERATOR SPEC V1 STATUS

The authority remains `synthetic_generator_v1`, SHA-256 `5f7d3f73572a34282c214860f1c0c656b18727a16fa0c4687caf071f690a7484`, with status `DRAFT_SPECIFICATION_REVIEW_REQUIRED`. Its scientific values were not edited to fit the implementation.

## 6. REPOSITORY STATE BEFORE CHANGES

Root Git commit was `3a62569f6eed24edb1a25a6a39eb3d11d5398138`; worktree was clean. The historical nested Sanskruti repository remained ignored and read-only. The repository already contained completed Vedant/Pulkit code and 949 passing tests.

## 7. BASELINE INTEGRATED TEST RESULT

Command: `PYTHONPYCACHEPREFIX=<mktemp> PYTHONPATH=src:tests:. python3 -m pytest tests -q -rs -p no:cacheprovider`. Result: exit 0; 949 passed, 0 failed, 2 skipped, 0 errors; pytest 11.44 s, measured wall time 13 s; 2026-09-20T16:31:18Z through 16:31:31Z. Skips were missing optional Captum and SHAP/XGBoost dependencies.

## 8. GENERATOR IMPLEMENTATION PLAN USED

The work was separated into configuration, RNG, strict records, latent dynamics, observation/measurement generation, orchestration, validation, hashing/provenance, manifest verification, CLI, tests, and generated evidence. Final mode is rejected before generation. No downstream phase was called to construct artifacts.

## 9. FILES CREATED

Created runtime configs under `configs/synthetic/fixtures/`; generator modules `config.py`, `rng.py`, `records.py`, `latent_state.py`, `observations.py`, `generator.py`, `validation.py`, `provenance.py`, and `manifest.py`; CLI `scripts/generate_synthetic_dataset.py`; three Phase-3 test modules; this review; and two small generated evidence directories under `artifacts/data/synthetic/{fixtures,smoke}/`.

## 10. FILES MODIFIED

Modified `src/data/synthetic/__init__.py` to expose the implemented API. Modified one obsolete assertion in `tests/test_synthetic_generator_spec.py`: it now permits Phase-3 implementation and only FIXTURE/SMOKE manifests while continuing to reject a final-data claim. No Vedant or Pulkit implementation file changed.

## 11. GENERATOR PACKAGE STRUCTURE

`config` loads/fails closed; `rng` derives stable PCG64 streams; `records` defines the only serializable rows; `latent_state` keeps hidden arrays in memory; `observations` separates physiology, observation timing, and measurement error; `generator` orchestrates and atomically promotes output; `validation` enforces schema/time/leakage contracts; `provenance` hashes code/files/environment; `manifest` creates and verifies bindings.

## 12. GENERATOR VERSION

`synthetic_generator_v1`. The implementation consumes the versioned Phase-2 contract and records a generator-source bundle SHA-256. Generated evidence recorded source-bundle hash `8bb0e78d9d4897ade47321bc69d1cd552b16a3365d6efa0f288604fad3f590bb` and honestly recorded the source tree as dirty at generation time.

## 13. GENERATOR CONFIG PATH / HASH

Fixture source: `configs/synthetic/fixtures/phase3_fixture_v1.json`, file SHA-256 `11940ca70aedbad04dc56f407e85b9fa850d481fae8ac3ea3f0a289a5fcc6406`, effective canonical hash `049d60e1cd850b4261dc2e6323650e5cc5511872648058a6823b671b0d584b07`. Smoke overlay: `phase3_smoke_v1.json`, file hash `ecb6eaac443badd7f590399a7311819de98dabce1c41d0429d533599171e0cbe`, inherited effective hash `a2a4bd048335fa3ffe46b624521d7feeff52e3e59ffdf3b3aa883f91c1cd9554`.

## 14. RAW SCHEMA VERSION / HASH

`synthetic_raw_schema_v1`; `configs/synthetic/synthetic_schema_v1.json`; SHA-256 `85751b498c7529f465ace6c07fae064902f7f06d8e243000d59de31c86a0b51c`. It was not modified.

## 15. RNG ARCHITECTURE

Every draw uses local `numpy.random.Generator(PCG64(SeedSequence(...)))`. Streams derive from `[primary_seed, subject_ordinal, Phase-2 component_code, optional_concept_child]`; no spawn-order dependence or global NumPy RNG mutation exists. Identity, episode/calendar, latent, observation, and measurement streams are isolated.

## 16. PRIMARY SEED STATUS

Final seed: `UNLOCKED_FINAL_GENERATOR_PARAMETER`. Engineering-only seeds 31001 (fixture) and 31002 (smoke) are explicit, hash-bound, and never represented as approved or final.

## 17. FINAL COHORT-SIZE STATUS

Final cohort size: `UNLOCKED_FINAL_SYNTHETIC_COHORT_SIZE`. Engineering evidence uses 4 and 24 subjects only; these counts carry no power, prevalence, or representativeness claim.

## 18. SUBJECT GENERATION IMPLEMENTATION

One synthetic adult subject is generated per ordinal. Age, controlled fixture sex category, and synthetic cardiac condition group are drawn from explicit engineering parameters. There are no names, addresses, MRNs, real identifiers, real-patient values, or additional hidden exported statics.

## 19. SUBJECT-ID CONTRACT

IDs follow `SYN-S-{subject_ordinal:08d}`, are unique, stable for identical version/config/seed, and encode only an ordinal—not condition, state, outcome, split, or support. The validator rejects duplicate subject IDs.

## 20. EPISODE GENERATION IMPLEMENTATION

Exactly one ICU-like episode is emitted for every subject. Boundaries are structural facts generated before observation events; they are never inferred from first/last measurements. Episode sequence is exactly 1 and the subject foreign key is enforced.

## 21. STAY-ID CONTRACT

IDs follow `SYN-E-{subject_ordinal:08d}` and remain stable and unique. They encode no target or state. Duplicate stay IDs, invalid subject references, or multiple retained episodes fail validation.

## 22. INTIME / OUTTIME IMPLEMENTATION

`intime` is a deterministic subject-stream offset from the synthetic epoch `2100-01-01T00:00:00Z`; no MIMIC era or anchor group is fabricated. `outtime = intime + generated duration`. Both serialize as timezone-aware UTC ISO-8601 with `Z`; `outtime > intime` is mandatory.

## 23. EPISODE-DURATION PROCESS

The Phase-2 truncated lognormal latent-burden equation is implemented. Fixture-only intercept, five latent coefficients, support-propensity coefficient, noise, and 20–132 h truncation bounds live in the runtime config. Duration is variable; it is structural label-only and never exported as a model feature.

## 24. LATENT-STATE IMPLEMENTATION

A five-dimensional continuous state uses reviewed-form group mean/covariance, continuous patient effects, mean reversion, drift, bounded cross-domain coupling, sparse signed multi-hour shocks, and domain-specific innovations at hourly internal steps. There are no improving/stable/deteriorating classes or three templates.

## 25. LATENT-STATE EXPORT PROTECTION

Latent arrays exist only within `latent_state.py` and orchestration memory. Export dataclasses have no latent member; strict schema equality rejects extras; manifests assert no latent export. Hostile serialization tests inject `latent_state` and receive fail-closed errors.

## 26. PHYSIOLOGICAL GENERATION IMPLEMENTATION

All 15 Phase-2 raw variables are generated through shared latent drivers plus variable-specific effects and noise. A documented logistic bounded link produces valid safety-domain observations without post-hoc silent clipping. GCS is rounded to an integer. Urine output is a six-hour-or-shorter interval-volume observation.

## 27. SOFA RAW-DOMAIN COVERAGE

Structural coverage passes for respiratory (`pao2`, `fio2`), coagulation (`platelet_count`), liver (`bilirubin_total`), cardiovascular (`mean_arterial_pressure`), CNS (`glasgow_coma_scale`), and renal (`creatinine`, `urine_output_volume`). No SOFA value or component score was calculated.

## 28. CARDIAC RAW-DOMAIN COVERAGE

The implementation exactly matches the Phase-2 15-concept inventory, including heart rate, systolic/diastolic pressure, respiratory rate, oxygen saturation, temperature, and lactate. No speculative extra variable was added.

## 29. OBSERVATION PROCESS

Each concept uses a separate exponential renewal stream. Intensity is variable-specific, contemporaneously latent-severity-modulated, patient-modulated, and bounded by explicit engineering rates. Event generation is irregular and separate from the hourly latent process.

## 30. MISSINGNESS IMPLEMENTATION

Missingness is event absence only. No null placeholder observation, imputation, interpolation, forward fill, backfill, or model grid is emitted. Smoke QA shows one subject without a platelet observation, demonstrating natural event absence without a missing-value row.

## 31. MEASUREMENT-NOISE IMPLEMENTATION

Measurement draws use concept-isolated substreams and explicit per-variable scales. The bounded inverse-link construction is deterministic for fixed inputs, finite, and domain-preserving; invalid baselines fail rather than being silently repaired.

## 32. RAW EVENT SCHEMA

Every row has deterministic `event_id`, subject/stay IDs, availability time, nullable interval bounds, concept, exactly one numeric value, null text value, exact unit, and generator component. Canonical order is `(stay_id, event_time, event_id)`.

## 33. EVENT-BOUND VALIDATION

Validation parses aware UTC timestamps and enforces `intime <= event_time <= outtime`. Interval measurements require `intime <= interval_start < interval_end == event_time`; point events require both interval fields null. The hostile after-outtime test passes by rejection.

## 34. SUPPORT-EVENT STATUS

The schema-compatible `support_intervals.jsonl` artifact is present and empty in both runs. Status is exactly `DEFERRED_TO_PHASE_9`; no positive, negative, label, or eligibility row was fabricated.

## 35. PHASE-9 BOUNDARY CONFIRMATION

Only the existing Phase-2 support table contract, identity seam, manifest entry, and empty placeholder are preserved. No vasopressor vocabulary, transition hazard, ventilation state, intervention effect, Pulkit mapping, or endpoint label was implemented.

## 36. FIELD-ROLE ENFORCEMENT

All schema fields must use the five Phase-2 role values. Export rows must have exactly the schema-declared keys. `outtime` is separately asserted to remain `STRUCTURAL_LABEL_ONLY` and `model_eligible=false`.

## 37. FORBIDDEN-FIELD ENFORCEMENT

Unknown/unclassified keys and target, future, split, or latent aliases fail closed. Tests cover trajectory, split, target, delta SOFA, latent state, and an outtime-role mutation. This complements—rather than replaces—the schema role check.

## 38. TARGET-LEAKAGE PROTECTION

No row contains a trajectory template, future state, future SOFA, future support, total/remaining stay feature, outcome class, censor reason, eligibility, prediction time, or label. Full future events remain timestamped raw observations for later cutoff-safe truncation.

## 39. SPLIT-NEUTRALITY RESULT

PASS. No split/fold/train/validation/test field exists in schema or generated rows. Dates have no split meaning. Subject assignment remains Phase 10 work.

## 40. PREPROCESSING-ABSENCE RESULT

PASS. Generator output is not imputed, scaled, encoded, normalized, binned, or tensorized. Source scanning confirms there is no sklearn import. The bounded measurement link is data generation, not learned preprocessing.

## 41. MODEL-IMPORT ABSENCE RESULT

PASS. The synthetic package imports neither torch, XGBoost, sklearn, models, calibration, metrics, search, nor selection. No model was loaded or executed.

## 42. PROVENANCE IMPLEMENTATION

Provenance records scope/version/hash, generator version, Git commit, dirty flag, generator source-bundle hash, effective and source config hashes, schema hash, RNG policy/seed identity, Python/NumPy/platform, UTC generation time, exact command, counts, and artifact identities.

## 43. DATASET MANIFEST IMPLEMENTATION

`synthetic_dataset_manifest_v1.json` is created from execution, never hand-edited. The validator checks manifest/generator/status/support state, config/schema hashes, every file hash and semantic hash, record counts, generated-subject count, foreign keys, field schema, domains, and temporal bounds.

## 44. ARTIFACT HASH POLICY

Every serialized JSONL table has byte SHA-256 in its manifest. Fixture byte hashes: subjects `24b950...2877`, episodes `6ad3da...0b0a`, events `a67f4c...01de`, empty support `e3b0c4...b855`. Smoke: `96d899...6546`, `2ea441...6e0f`, `0521c5...b2e7`, and the same empty-file hash.

## 45. SEMANTIC FINGERPRINT POLICY

SHA-256 covers canonical JSON containing table identity and ordered rows, independent of path, manifest time, and filesystem metadata. Fixture semantic hashes are subjects `46e569...efad`, episodes `99ae0e...d8be`, events `3facb2...4094`, support `248c47...d0fa`; smoke equivalents are `59af2d...93d2`, `b2ed20...7180`, `940c95...540f`, and `248c47...d0fa`.

## 46. DETERMINISM TEST RESULT

PASS. Two independent same-config generations produce identical subjects, episodes, events, support rows, table bytes, byte hashes, and semantic hashes. Manifest creation time/code-state metadata are provenance and are not treated as table semantics.

## 47. DIFFERENT-SEED TEST RESULT

PASS. Changing only the engineering seed changes generated content while field sets, concepts, roles, units, schema, and other contracts remain unchanged.

## 48. LATENT-SERIALIZATION HOSTILE TEST

PASS by rejection. Adding `latent_state` to a serialized row is rejected as forbidden/unclassified, and latent state is absent from every exported dataclass.

## 49. OUTTIME-FEATURE HOSTILE TEST

PASS by rejection. Mutating the schema so `outtime` becomes model eligible raises the explicit outtime-role validation error.

## 50. TRAJECTORY-FIELD HOSTILE TEST

PASS by rejection. Injecting `trajectory` into subject output fails strict schema and forbidden-field validation.

## 51. SPLIT-FIELD HOSTILE TEST

PASS by rejection. Injecting `split` fails before artifact authorization.

## 52. FUTURE-TARGET HOSTILE TEST

PASS by rejection. `delta_sofa_24` and generic `target` injection are covered; neither may enter raw output.

## 53. EVENT-OUTSIDE-STAY HOSTILE TEST

PASS by rejection. An event timestamp set after `outtime` produces `event_time outside episode bounds`.

## 54. MANIFEST-TAMPER TEST

PASS by rejection. Appending a row after manifest creation causes an artifact SHA-256 mismatch.

## 55. CONFIG-HASH MISMATCH TEST

PASS by rejection. Supplying a changed effective config identity against an existing manifest raises `generator config hash mismatch`.

## 56. SCHEMA-HASH MISMATCH TEST

PASS by rejection. Even a schema-byte change without regeneration raises `raw schema hash mismatch`.

## 57. FINAL-MODE FAIL-CLOSED TEST

PASS. A runtime config marked `final` raises `FINAL mode refused` and lists every Phase-2 unresolved parameter. There is no fixture/smoke fallback and no output directory promotion.

## 58. EXACT-CLONE CHECK

PASS. Whole-subject trajectory fingerprints exclude identifiers as the distinguishing mechanism and include statics plus complete relative trajectories; smoke exact-clone count is 0. No arbitrary near-similarity threshold was used.

## 59. VEDANT TIMESTAMP COMPATIBILITY PRECHECK

PASS read-only. Generated IDs and aware UTC boundaries construct `RetainedICUStay` objects accepted by the unchanged timestamp utility. Fixture potential legal counts are `[6,4,10,7]`; smoke has 136 total, includes one zero-cutoff episode and 22 multiple-cutoff episodes.

## 60. PULKIT HISTORY COMPATIBILITY PRECHECK

PASS at the raw precondition level. Every observation has subject ID, stay ID, stable event ID, explicit event time, concept, value, and unit, enabling later cutoff truncation. No HistoryProvider, support state engine, PredictionPipeline, or serving adapter was implemented or modified.

## 61. GENERATION MODES IMPLEMENTED

Explicit `fixture`, `smoke`, and `final` mode recognition exists. Fixture/smoke require the non-scientific scope marker and positive explicit parameters. Final always fails while the authority lists unresolved requirements.

## 62. FIXTURE GENERATION RESULT

PASS. `artifacts/data/synthetic/fixtures/phase3_fixture_v1`: 4 subjects, 4 episodes, 1,222 raw events, 0 support intervals, all 15 concepts, durations 48.048481–88.736766 h, 214–384 observations/stay, 27 potential legal cutoffs, 0 exact clones.

## 63. SMOKE GENERATION RESULT

PASS. `artifacts/data/synthetic/smoke/phase3_smoke_v1`: 24 subjects, 24 episodes, 7,870 raw events, 0 support intervals, all 15 concepts, durations 23.809363–107.699859 h, 148–666 observations/stay, 136 potential legal cutoffs, 0 exact clones.

## 64. AUTHORIZED FINAL DATASET GENERATION STATUS

BLOCKED / NOT GENERATED. No artifact or manifest is named or marked `AUTHORIZED_FINAL_SYNTHETIC_DATA`.

## 65. IF GENERATED — FINAL DATASET ARTIFACT INVENTORY

Not applicable: no authorized final dataset was generated. The fixture/smoke inventory is subjects JSONL, episodes JSONL, raw-events JSONL, empty support-intervals JSONL, generated QA JSON, and manifest JSON.

## 66. IF GENERATED — FINAL DATASET MANIFEST PATH

Not applicable. Only fixture/smoke manifests exist at their respective artifact directories.

## 67. IF GENERATED — FINAL DATASET HASHES

Not applicable. No final-data hashes exist or are claimed.

## 68. IF GENERATED — FINAL DATASET STRUCTURAL QA

Not applicable. Fixture/smoke structural QA passes but cannot authorize final data.

## 69. IF NOT GENERATED — EXACT BLOCKERS

Unresolved: final size; final seed; age maximum/distribution; sex/gender vocabulary/distribution; cardiac subtype taxonomy/mixture; duration values; latent values; true-physiology values; observation/missingness values; measurement-noise values; support vocabulary/hazards; final storage/output decision; final environment/package manager; and three-member review/freeze. Support science additionally belongs to Phase 9.

## 70. DESCRIPTIVE GENERATOR QA

Automatically generated QA reports counts, events/concept, duration min/median/max, observations/stay, per-concept subject absence, value ranges, potential legal-cutoff counts, clone count, structural result, and support deferral. It computes no target distribution or predictive metric and explicitly disclaims clinical validation, prevalence, realism, and clinical use.

## 71. FOCUSED PHASE-3 TEST RESULTS

PASS: 29 Phase-3 implementation tests. Final combined Phase-2/Phase-3 focused run: 57 passed, 0 failed/skipped/errors in 3.17 s.

## 72. FULL INTEGRATED REGRESSION RESULTS

PASS: 978 passed, 0 failed, 2 skipped, 0 errors in 13.24 s; measured wall time 14 s, 2026-09-20T16:44:27Z through 16:44:41Z. Existing Vedant, Pulkit, API, dashboard, registry, governance, evaluation, and historical fixture tests remained green. The two skips are unchanged optional-dependency blocks.

## 73. STATIC / IMPORT CHECK RESULTS

`python3 -m compileall -q src/data/synthetic scripts/generate_synthetic_dataset.py`: exit 0. `git diff --check`: exit 0. No repository Ruff/Black/Mypy configuration was found, so no invented lint gate was asserted.

## 74. EXACT COMMANDS EXECUTED

Principal reproducible commands: baseline/full `PYTHONPYCACHEPREFIX=<mktemp> PYTHONPATH=src:tests:. python3 -m pytest tests -q -rs -p no:cacheprovider`; focused `python3 -m pytest tests/test_synthetic_generator_{spec,runtime,leakage,manifest}.py ...`; generation `PYTHONPATH=src python3 scripts/generate_synthetic_dataset.py --config <fixture-or-smoke-config> --output <artifact-directory> --generation-timestamp 2026-09-20T16:50:00Z`; compileall and `git diff --check`; manifest verification through `validate_manifest`.

## 75. UNLOCKED PARAMETERS REMAINING

Final primary seed, final cohort size, primary storage/output decision, and final environment/package-manager identity remain unlocked. JSONL and repository artifact locations are Phase-3 engineering choices for fixture/smoke only.

## 76. REVIEW-REQUIRED PARAMETERS REMAINING

Static vocabulary/distributions, age design, cardiac mixture, duration parameters, latent matrices/distributions, true-physiology mappings, observation/missingness process values, measurement-noise values, support vocabulary/hazards, and all final freeze approvals remain review-required.

## 77. SCIENTIFIC FEASIBILITY ISSUES

No contradiction or implementation impossibility was found in the Phase-2 algorithms/schema. The issue is incomplete scientific parameterization, not computational feasibility. Fixture behavior is evidence of software feasibility only and cannot validate physiological realism or endorse the chosen engineering numbers.

## 78. PHASE 3 DEFINITION-OF-DONE CHECKLIST

Implementation checklist: all items pass, including prerequisites, baseline, modularity, config consumption, local RNG, IDs/boundaries/events, domains/units, irregular observation/missingness, no fill/bin/cutoff/split/label/model, raw SOFA coverage, support boundary, provenance/manifest, hashes, deterministic/different-seed/tamper/mismatch/final-fail tests, focused/full regression, and no Vedant/Pulkit changes. Final raw dataset checklist remains unchecked because the blockers in section 69 persist.

## 79. GENERATOR IMPLEMENTATION STATUS

`GENERATOR IMPLEMENTATION COMPLETE — PASS FOR FIXTURE/SMOKE USE`. It is importable, CLI-driven, config-bound, validated, deterministic, provenance-producing, and fail-closed for final mode.

## 80. FINAL RAW DATASET STATUS

`BLOCKED — NOT YET GENERATED`. Human freeze and final scientific values are required. Phase-3 engineering artifacts must never be relabeled as final.

## 81. PHASE 4 READINESS

Implementation interface readiness: PASS. Final Phase-4 execution readiness: BLOCKED because no authorized raw synthetic dataset exists. Phase 4 must wait for freeze and approved regeneration; it must not accept fixture/smoke evidence as its cohort input.

## 82. EXPLICIT CONFIRMATION: NO RECOVERY LABELS WERE GENERATED

Confirmed. No DeltaSOFA24, DeltaSOFA48, future SOFA, recovery class, or recovery eligibility exists.

## 83. EXPLICIT CONFIRMATION: NO REMAINING-STAY LABELS WERE GENERATED

Confirmed. No remaining hours or `log1p` remaining-time label exists; only structural `outtime` is exported in episode metadata.

## 84. EXPLICIT CONFIRMATION: NO ORGAN-SUPPORT LABELS WERE GENERATED

Confirmed. The support artifact is empty and marked deferred; no OFF-to-ON label, eligibility, censoring, or threshold was created.

## 85. EXPLICIT CONFIRMATION: NO TRAIN/VALIDATION/TEST SPLIT WAS CREATED

Confirmed. The generator is split-neutral and contains no split fields or date-based assignment.

## 86. EXPLICIT CONFIRMATION: NO MODEL TRAINING WAS PERFORMED

Confirmed. No XGBoost, GRU, LSTM, baseline, calibration, threshold selection, metric evaluation, or learnability tuning ran.

## 87. EXPLICIT CONFIRMATION: PHASE 4+ WAS NOT IMPLEMENTED

Confirmed. No cohort acceptance/index, canonical timeline, SOFA, 8x6-hour features, labels, support science, split/preprocessor, model integration/search, G1 acceptance, or handoff implementation was created. Work stops at Phase 3.
