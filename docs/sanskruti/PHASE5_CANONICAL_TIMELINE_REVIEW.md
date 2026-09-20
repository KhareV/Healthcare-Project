# Sanskruti Rebuild Phase 5 — Canonical Timeline Review

## 1. SANSKRUTI REBUILD PHASE 5 SUMMARY
Implementation and engineering fixture/smoke execution pass. Final timeline execution is blocked because upstream data/cohort are not authorized final artifacts.
## 2. PHASE-1 PREREQUISITE STATUS
PASS. Project Scope v2 hash: `4fcab7045b30b631bdbb9e8219226c93288d6ef1c2550125d3456ab431d04994`.
## 3. PHASE-2 PREREQUISITE STATUS
PASS. Generator and raw-schema contracts validate.
## 4. PHASE-3 PREREQUISITE STATUS
PASS for engineering artifacts; fixture/smoke manifests revalidated.
## 5. PHASE-4 PREREQUISITE STATUS
PASS for engineering cohorts/indexes; both Phase-4 manifests revalidated.
## 6. SOURCE DATA / COHORT STATUS
Sources are `FIXTURE`/`SMOKE` and `ENGINEERING_COHORT_*`, never authorized final.
## 7. UPSTREAM MANIFEST VALIDATION
PASS for Phase-3 and Phase-4 hashes, schemas, counts, identities, events, and timestamp binding.
## 8. REPOSITORY STATE BEFORE CHANGES
Clean at commit `a5eb6a0589dd7799c14cd5b7772165b028074d55`.
## 9. BASELINE INTEGRATED TEST RESULT
Exit 0: 1,017 passed, 2 skipped, 0 failures/errors in 14.73 s; 16 s wall time, 2026-09-20T16:58:32Z–16:58:48Z.
## 10. PROCESSED-SCHEMA PATH
`configs/synthetic/processed_schema_v1.json`.
## 11. PROCESSED-SCHEMA VERSION
`synthetic_processed_timeline_v1`.
## 12. PROCESSED-SCHEMA SHA-256
`7e0ba3d3789b5b469198773a5b1a7af2062ad973466a4af0135fdf97383688d9`.
## 13. SCHEMA VERSIONING DECISION
V1 is honest because no prior production processed-timeline schema existed; historical tensor fixtures are different artifacts.
## 14. RAW->CANONICAL CONCEPT MAP PATH
`configs/synthetic/concept_map_v1.json`.
## 15. CONCEPT MAP VERSION / SHA-256
`synthetic_concept_map_v1`; `24de193ea0041c128d921d1585dff17cd461af5fc7c415853449aa400e7bf74d`.
## 16. CANONICAL EVENT MODEL
Fourteen strict fields preserve event/subject/stay identity, availability time, intervals, concept/value/unit, kind, source, and provenance.
## 17. SUBJECT-ID SEMANTICS
Unchanged synthetic routing identity; exactly matches retained cohort subject.
## 18. STAY-ID SEMANTICS
Unchanged retained-stay routing identity; unknown/non-retained stays cannot contaminate the timeline.
## 19. EVENT-ID SEMANTICS
Phase-3 deterministic event ID is preserved and globally unique; duplicates fail closed.
## 20. EVENT-TIME SEMANTICS
`event_time` is the first instant the complete observation is available; it alone controls retrospective cutoff visibility.
## 21. EVENT-KIND SEMANTICS
`point` for measurements/assessments; `quantity_interval` for urine volume with preserved collection bounds and availability at interval end.
## 22. EPISODE-BOUND ENFORCEMENT
Canonicalization independently enforces `intime <= event_time <= outtime` and interval bounds.
## 23. CANONICAL CONCEPT INVENTORY
Exactly the 15 Phase-2 concepts; canonical names are stable machine keys and currently identity-map from raw keys.
## 24. CARDIAC RAW-CONCEPT COVERAGE
Heart rate, systolic/diastolic pressure, MAP, respiratory/cardio-critical-care variables, temperature, and lactate are covered without additions.
## 25. CANONICAL UNIT CONTRACT
Every concept has one explicit canonical unit bound in the concept map and provenance CSV.
## 26. UNIT-CONVERSION INVENTORY
All 15 mappings are explicit `IDENTITY`; no hidden conversion exists.
## 27. UNSUPPORTED-UNIT POLICY
Hard failure; units are never inferred from magnitude.
## 28. UNKNOWN-CONCEPT POLICY
Hard failure; no raw concept is silently dropped.
## 29. DUPLICATE-EVENT POLICY
Duplicate event IDs fail; exact corruption is not arbitrarily deduplicated.
## 30. SAME-TIMESTAMP EVENT POLICY
Distinct event IDs are preserved without Phase-7 aggregation.
## 31. CANONICAL ORDERING
Deterministic `(subject_id, stay_id, event_time, canonical_concept, event_id)`.
## 32. STATIC-DATA CONTRACT
One separate row per retained stay; statics are not repeated on every dynamic event.
## 33. STATIC PROVENANCE
Schema `synthetic_canonical_statics_v1`, SHA-256 `d333c62c53c2d4d7f2de59953018352d07910bf5db3e651a927c7114ac89d138`; age/sex/cardiac group plus structural bounds and lineage.
## 34. SOFA RAW-DEPENDENCY MAP
All six raw domains map through machine-readable provenance; no SOFA score exists.
## 35. RESPIRATORY DEPENDENCY STATUS
PaO2/FiO2 available; ventilation context explicitly deferred to Phase 9.
## 36. COAGULATION DEPENDENCY STATUS
Platelet count available in `10^9/L`.
## 37. LIVER DEPENDENCY STATUS
Total bilirubin available in `mg/dL`.
## 38. CARDIOVASCULAR DEPENDENCY STATUS
MAP available; vasopressor context deferred to Phase 9.
## 39. CNS DEPENDENCY STATUS
Integer GCS available with score semantics.
## 40. RENAL DEPENDENCY STATUS
Creatinine and interval urine volume are available with distinct semantics.
## 41. SUPPORT-DEPENDENCY STATUS
`DEFERRED_TO_PHASE_9`; no support state/event science was created.
## 42. FEATURE-DICTIONARY PATH
`docs/sanskruti/FEATURE_DICTIONARY.md`.
## 43. FEATURE-DICTIONARY STATUS
Phase-5 raw-input dictionary only; no ordered model feature schema or empirical statistics.
## 44. FEATURE-PROVENANCE PATH
`docs/sanskruti/feature_provenance_v1.csv`.
## 45. FEATURE-PROVENANCE VERSION / HASH
`phase5_v1`; `ed6c06c04533bd8b07cef585c178e3d9d17fb3745d3aaba0cb0a736ef9b7757a`.
## 46. PROVENANCE COMPLETENESS RESULT
PASS. Every present canonical concept maps to exactly one stable provenance record.
## 47. PROHIBITED-VARIABLE AUDIT
PASS. Schema and strict raw shape reject targets, outcomes, split, support labels, censoring, and future summaries.
## 48. GENERATOR-LATENT-FIELD AUDIT
PASS. No latent, trajectory, risk-group, or future-state field exists.
## 49. FUTURE-SUMMARY AUDIT
PASS. No whole-stay extrema, duration summary, final value, or future severity exists.
## 50. OUTTIME MODEL-ELIGIBILITY AUDIT
PASS. `outtime` exists only in static structural metadata with `model_eligible=false`.
## 51. PREPROCESSING-ABSENCE AUDIT
PASS. No imputation, scaler, encoder, learned transform, or normalization is present.
## 52. FORWARD-FILL ABSENCE AUDIT
PASS. No fill, carry-forward, interpolation, or fake missing row exists.
## 53. BINNING ABSENCE AUDIT
PASS. Timeline remains full event-level history; no hour/6-hour/lookback grid exists.
## 54. LABEL-ABSENCE AUDIT
PASS. No recovery, ICU-time, support, eligibility, or censoring label exists.
## 55. SPLIT-ABSENCE AUDIT
PASS. Manifest is `NOT_ASSIGNED_PHASE_10`; no split or fake era field exists.
## 56. CANONICAL TIMELINE ARTIFACT PATH
Primary smoke: `artifacts/data/synthetic/timelines/smoke/phase5_smoke_v1/canonical_timeline.jsonl`.
## 57. CANONICAL TIMELINE STATUS
`ENGINEERING_CANONICAL_TIMELINE_SMOKE`; never promoted to final.
## 58. CANONICAL TIMELINE SHA-256
Smoke `71b5eaabaf65d5a99df3646e749d2ac38cbd7a17d6131f3b855e59c3b92cd233`; fixture `4be8eba228968ba295519c9f799dc37604211e8453d3f475066f80dccdeb6f9a`.
## 59. CANONICAL TIMELINE SEMANTIC HASH
Smoke `014ef72f9610086f1e4f4ea594727dfa4b183b3f76e6f85e20915956d84fe5a2`; fixture `8e09bf0a0d29add741648d5c3492cfff43c3bdb9a7372f470b0e2c6d68803df4`.
## 60. CANONICAL STATIC ARTIFACT PATH / HASH
Smoke `canonical_statics.jsonl`, SHA-256 `c95b835b243d91e415e6b248be0d2995051c450d637b4e60134a41b10ce50f8d`.
## 61. PROCESSED-DATA MANIFEST PATH
Primary smoke: `artifacts/data/synthetic/timelines/smoke/phase5_smoke_v1/synthetic_processed_manifest_v1.json`.
## 62. PROCESSED-DATA MANIFEST SHA-256
Smoke `e73456990afdac79b155276f88ea5a1fb10a8c316ec35a18738f0664d2314c9b`; fixture `c7d87bf4522fc740a5106aaa708d14262af7738ee98843c9732c8dc7a9124f1b`.
## 63. CANONICAL EVENT COUNTS
Smoke 7,708 events across 23 stays; fixture 1,222 across 4. Smoke kinds: 7,466 point, 242 quantity-interval.
## 64. EVENT COUNTS BY CONCEPT
Smoke: bilirubin 92, creatinine 133, DBP 984, FiO2 274, GCS 157, HR 1,045, lactate 150, MAP 688, SpO2 1,179, PaO2 224, platelets 116, RR 882, SBP 974, temperature 568, urine 242.
## 65. INVALID/REJECTED EVENT COUNTS
Invalid 0, unknown 0, out-of-bounds 0; 162 valid raw events were structurally filtered because their stay was not retained.
## 66. POST-OUTTIME TEST RESULT
PASS by rejection.
## 67. PRE-INTIME TEST RESULT
PASS by rejection.
## 68. UNKNOWN-CONCEPT HOSTILE TEST
PASS by rejection.
## 69. UNSUPPORTED-UNIT HOSTILE TEST
PASS by rejection.
## 70. DOUBLE-NORMALIZATION HOSTILE TEST
PASS. Canonical-stage rows cannot be processed as Phase-3 raw rows.
## 71. OUTCOME-FIELD HOSTILE TEST
PASS by rejection for trajectory, outcome, remaining-stay, split, and latent attacks.
## 72. DUPLICATE-EVENT-ID HOSTILE TEST
PASS by rejection.
## 73. CROSS-STAY HOSTILE TEST
PASS by rejection of subject/stay mismatch and provider stay mixing.
## 74. SCHEMA-VERSION ATTACK RESULT
PASS by processed-schema hash rejection.
## 75. PROVENANCE-HASH ATTACK RESULT
PASS by provenance hash rejection.
## 76. CONCEPT-MAP TAMPER RESULT
PASS by concept-map hash rejection.
## 77. PULKIT HISTORY-PROVIDER COMPATIBILITY
PASS through storage-backed `CanonicalTimelineHistoryProvider`; unknown stays fail with no fallback.
## 78. EXISTING PULKIT TRUNCATION REUSE CONFIRMATION
Confirmed. Existing `SyntheticPointEventHistoryTruncator` performs legal-cutoff validation and `event_time <= t`; Phase 5 adds no truncator.
## 79. EXACT-CUTOFF EVENT RESULT
PASS. A point event exactly at legal `t` is included.
## 80. FUTURE-EVENT EXCLUSION RESULT
PASS. An event one microsecond after `t` is excluded.
## 81. RAW-HISTORY FUTURE-PERTURBATION RESULT
PASS. Modifying events strictly after `t` leaves truncated history unchanged.
## 82. STAY-ISOLATION RESULT
PASS with distinct overlapping stay histories and disjoint event IDs.
## 83. VEDANT RECEIVER PRECHECK
PASS. Subject/stay identities and Phase-4 cutoffs remain stable; feature order/tensors remain Phase 7+.
## 84. FILES CREATED
Created three schemas/maps, three implementation modules, CLI, dictionary/CSV, four test modules plus helper, two generated bundles, and this review.
## 85. FILES MODIFIED
Only `src/data/synthetic/__init__.py`; no Vedant/Pulkit scientific behavior changed.
## 86. TESTS ADDED
44 focused tests covering the requested mapping, time, leakage, provenance, manifest, history, and final-mode cases.
## 87. FOCUSED PHASE-5 TEST RESULTS
PASS: final-tree run completed with 44 passed, 0 failed/skipped/errors in 3.11 s.
## 88. FULL INTEGRATED REGRESSION RESULTS
PASS: final-tree run completed with 1,061 passed, 2 skipped, 0 failures/errors in 17.79 s; 20 s wall time, 2026-09-20T17:08:02Z–17:08:22Z.
## 89. STATIC / IMPORT CHECK RESULTS
Compileall exit 0; `git diff --check` exit 0; no configured Ruff/Black/Mypy gate found.
## 90. EXACT COMMANDS EXECUTED
Baseline/full isolated pytest; focused four-module pytest; two `build_synthetic_timeline.py` engineering commands with timestamp `2026-09-20T17:25:00Z`; manifest validation; SHA-256 inventory; compileall; diff check.
## 91. UPSTREAM DEFECTS FOUND
None. Non-retained smoke events were correctly filtered, not defective.
## 92. UNRESOLVED PARAMETERS
Final Phase-3 science/freeze, final Phase-4 cohort, Phase-9 support context, Phase-7 feature schema, and Phase-10 split/preprocessing remain unresolved.
## 93. SCIENTIFIC FEASIBILITY ISSUES
None established. All 15 raw concepts and six SOFA prerequisite domains are present; support-aware respiratory/cardiovascular completion remains deferred.
## 94. PHASE-5 DEFINITION-OF-DONE CHECKLIST
Implementation DoD passes: contracts, mappings, units, kinds, validation, provenance, deterministic artifacts, provider/truncation tests, hostile tests, and regressions. Final execution remains blocked by upstream authorization.
## 95. PHASE-5 IMPLEMENTATION STATUS
`COMPLETE — ENGINEERING FIXTURE/SMOKE PASS`.
## 96. FINAL CANONICAL-TIMELINE STATUS
`BLOCKED — AUTHORIZED FINAL RAW DATA AND RETAINED FINAL COHORT REQUIRED`.
## 97. PHASE-6 READINESS
Engineering interface ready with names, units, timestamps, identities, provenance, and SOFA dependencies. Final execution and support-aware scoring remain blocked.
## 98. EXPLICIT CONFIRMATION: PROJECT REMAINS CARDIAC / HEART-DISEASE FOCUSED
Confirmed; it was not narrowed to heart failure only.
## 99. EXPLICIT CONFIRMATION: NO 8-BIN FEATURE GRID WAS BUILT
Confirmed.
## 100. EXPLICIT CONFIRMATION: NO MASKS / TSLO / PADDING WERE GENERATED
Confirmed.
## 101. EXPLICIT CONFIRMATION: NO SOFA SCORES WERE COMPUTED
Confirmed.
## 102. EXPLICIT CONFIRMATION: NO RECOVERY / ICU-TIME LABELS WERE GENERATED
Confirmed.
## 103. EXPLICIT CONFIRMATION: NO ORGAN-SUPPORT LABELS WERE GENERATED
Confirmed.
## 104. EXPLICIT CONFIRMATION: NO TRAIN/VALIDATION/TEST SPLIT WAS CREATED
Confirmed.
## 105. EXPLICIT CONFIRMATION: NO PREPROCESSOR WAS FIT
Confirmed.
## 106. EXPLICIT CONFIRMATION: NO MODEL TRAINING WAS PERFORMED
Confirmed.
## 107. EXPLICIT CONFIRMATION: PHASE 6+ WAS NOT IMPLEMENTED
Confirmed. Work stops at Phase 5.
