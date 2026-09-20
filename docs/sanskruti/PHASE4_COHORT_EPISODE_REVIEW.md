# Sanskruti Rebuild Phase 4 — Cohort, Episode, and Structural Index Review

## 1. SANSKRUTI REBUILD PHASE 4 SUMMARY

Implementation, engineering execution, validation, provenance, and tests pass. Final cohort execution remains blocked because Phase 3 produced only FIXTURE/SMOKE data, never `AUTHORIZED_FINAL_SYNTHETIC_DATA`. No Phase 5+ work was performed.

## 2. PHASE-1 PREREQUISITE STATUS

PASS. Project Scope v2 was inspected and remains the population/time authority. Scope SHA-256: `4fcab7045b30b631bdbb9e8219226c93288d6ef1c2550125d3456ab431d04994`.

## 3. PHASE-2 PREREQUISITE STATUS

PASS. `synthetic_generator_v1` and `synthetic_raw_schema_v1` exist, validate, and contain no scope conflict. Generator-spec hash: `5f7d3f73572a34282c214860f1c0c656b18727a16fa0c4687caf071f690a7484`; schema hash: `85751b498c7529f465ace6c07fae064902f7f06d8e243000d59de31c86a0b51c`.

## 4. PHASE-3 PREREQUISITE STATUS

PASS for implementation and engineering artifacts. Both Phase-3 manifests, all artifact hashes, effective configs, schema, foreign keys, roles, values, and event bounds revalidated before Phase-4 work.

## 5. SOURCE DATASET STATUS

Only `FIXTURE` and `SMOKE` exist. Neither source is authorized final data. Fixture manifest SHA-256 is `276e8825e70b1fd87a1f210e54b6cfdf1f7aa7918be3f0222aee1459b20ca361`; smoke manifest SHA-256 is `7ee523d11460119633d7eb6ea89dbd1bc9978c4741289fa2cf72cc4bada7b428`.

## 6. SOURCE MANIFEST VALIDATION RESULT

PASS for fixture and smoke through the Phase-3 manifest validator. A final-mode build against either source fails before output creation because their status is not `AUTHORIZED_FINAL_SYNTHETIC_DATA`.

## 7. REPOSITORY STATE BEFORE CHANGES

Clean Git worktree at `8704521a1f188811d4c4443b17ded247f9fca1d3` (`Implement Sanskruti Phase 3 synthetic generator`). Historical nested candidate code remained read-only.

## 8. BASELINE INTEGRATED TEST RESULT

Command: `PYTHONPYCACHEPREFIX=<mktemp> PYTHONPATH=src:tests:. python3 -m pytest tests -q -rs -p no:cacheprovider`. Exit 0: 978 passed, 0 failed, 2 skipped, 0 errors in 12.93 s; measured 14 s from 2026-09-20T16:47:59Z to 16:48:13Z. Skips were unchanged optional Captum and SHAP/XGBoost dependencies.

## 9. COHORT SPEC PATH

`configs/synthetic/cohort_spec_v2.yaml`.

## 10. COHORT SPEC VERSION

`synthetic_cohort_v2`, status `DRAFT_IMPLEMENTATION_REVIEW_REQUIRED`.

## 11. COHORT SPEC SHA-256

`a77a2c7e8d3fc01195a867c2d1223c51f5e8199f2b3ebc76309813b5d26d84a7`.

## 12. POPULATION DOMAIN USED

`CARDIAC_HEART_DISEASE`: adult, synthetic, cardiac/heart-disease-focused subjects. It is not narrowed to heart failure and not expanded to unrestricted ICU.

## 13. ADULT POLICY

Require integer `age_years >= 18`; no invented upper-age exclusion. Underage subjects receive deterministic reason `NOT_ADULT`.

## 14. CARDIAC INCLUSION POLICY

Membership uses the baseline `cardiac_condition_group` field. Allowed values are read from the source Phase-3 effective runtime config, not inferred from physiology or future outcomes. This permits an honest engineering build while the final taxonomy remains review-required.

## 15. RETAINED-EPISODE POLICY

Exactly one generated ICU-like episode per subject, using explicit Phase-3 boundaries. A subject is retained only when the existing Vedant generator returns at least one legal cutoff.

## 16. MULTIPLE-EPISODE / TIE POLICY

Multiple episodes are a hard Phase-3 generator-contract violation in v1; therefore no tie rule is needed or invented. Tests reject multiple episodes rather than sorting by stay ID or row order.

## 17. SUBJECT-ID CONTRACT

Non-empty synthetic string, unique in source and stable through cohort/index outputs. Missing IDs are excluded as `INVALID_SUBJECT_ID`; duplicates fail closed.

## 18. STAY-ID CONTRACT

Non-empty synthetic string, unique, owned by exactly one subject, and stable through the index. Missing IDs are excluded; duplicates and cross-subject collisions fail closed.

## 19. INTIME CONTRACT

Required, timezone-aware UTC ISO-8601 `Z`, taken directly from the Phase-3 episode artifact. It is never inferred or rounded.

## 20. OUTTIME CONTRACT

Required, timezone-aware UTC ISO-8601 `Z`, explicitly structural and never model eligible. It is not duplicated into the structural index.

## 21. EPISODE-BOUNDARY VALIDATION

`outtime > intime` is enforced. Missing, malformed, equal, and reversed boundaries receive deterministic structural exclusions. Episode endpoints are never inferred from event timestamps.

## 22. EVENT / EPISODE CONSISTENCY RESULT

PASS. Phase-3 source validation and Phase-4 preselection checks require matching subject/stay linkage and `intime <= event_time <= outtime`. Invalid events trigger `UPSTREAM PHASE-3 DEFECT`; they are never silently dropped.

## 23. LEGAL-CUTOFF REQUIREMENT

At least one row must be returned by the existing timestamp generator. The smoke episode with 23.809363 h duration is excluded as `NO_LEGAL_PREDICTION_CUTOFF`; production cohort code contains no manual duration approximation.

## 24. VEDANT TIMESTAMP IMPLEMENTATION REUSED

PASS. Phase 4 imports and calls `RetainedICUStay`, `generate_prediction_timestamps`, and `validate_prediction_rows` from `src/data/timestamps.py`. The cutoff formula is not reimplemented.

## 25. TIMESTAMP SPEC VERSION / HASH

`timestamp_spec_v1`; implementation SHA-256 `c7bd20e6d475ab3699fad5cbf829049311b010407bb387338dc35db5ed718e2c`.

## 26. FIRST-CUTOFF RESULT

PASS. Earliest emitted cutoff is exactly `intime + 24h`; no earlier row is present.

## 27. CADENCE RESULT

PASS. Consecutive rows use the unchanged exact six-hour cadence relative to the original, unrounded `intime`.

## 28. MAX-12 RESULT

PASS. A 300-hour hostile stay produces exactly grid indices 0–11 and no thirteenth row. Smoke maximum is 12.

## 29. EXACT END-BOUNDARY RESULT

PASS. A 30-hour stay emits grid index 0 at `intime + 24h`, exactly equal to `outtime - 6h`.

## 30. RETAINED COHORT ARTIFACT PATH

Primary engineering smoke evidence: `artifacts/data/synthetic/cohorts/smoke/phase4_smoke_v2/retained_cohort.jsonl`. A separate fixture artifact exists under `cohorts/fixtures/phase4_fixture_v2/`.

## 31. RETAINED COHORT STATUS

`ENGINEERING_COHORT_SMOKE` for the smoke evidence and `ENGINEERING_COHORT_FIXTURE` for the tiny fixture. Neither is `RETAINED_SYNTHETIC_COHORT` final status.

## 32. RETAINED COHORT SHA-256

Smoke byte hash: `112dbf53b73bda2bd0c29e47009d1b07abb0c235059733864a738cd85b80a27a`. Fixture: `1ab02a9d8f31bdc013c60782d7aa66eb7d1f5e0d72fb67783643c9c3861f205e`.

## 33. COHORT SEMANTIC HASH

Smoke: `62765c8421949b14c12acb4ac71fde1e16f3331f248e05897ae8457519c84df6`. Fixture: `544c9dd7f353ff1c5b1bf2d9d522f6d8168209589381047a17481f8a85948503`.

## 34. GENERATED SUBJECT COUNT

Smoke summary: 24. Fixture summary: 4. These are engineering counts, not final cohort-size claims.

## 35. GENERATED EPISODE COUNT

Smoke summary: 24. Fixture summary: 4. Both validate one source episode per subject.

## 36. RETAINED SUBJECT COUNT

Smoke: 23. Fixture: 4.

## 37. RETAINED STAY COUNT

Smoke: 23. Fixture: 4. Every retained subject maps to exactly one retained stay.

## 38. EXCLUSION COUNTS BY REASON

Smoke: `NO_LEGAL_PREDICTION_CUTOFF = 1`; all other reasons 0. Fixture: no exclusions. Counts come from generated `exclusion_summary.json`, not manual cohort filtering.

## 39. OUTCOME-FREE STRUCTURAL INDEX PATH

Primary smoke evidence: `artifacts/data/synthetic/cohorts/smoke/phase4_smoke_v2/structural_index.jsonl`.

## 40. STRUCTURAL INDEX SCHEMA VERSION

`synthetic_structural_index_v2`; schema path `configs/synthetic/structural_index_schema_v2.json`; SHA-256 `67e9bf99509b548ba694607e410f87ab70fac56b95324079b111ab5493cf4a33`.

## 41. STRUCTURAL INDEX SHA-256

Smoke: `c5804e961711a2fc6c6c2cb101346169d7ba51601df95701a98bfa64124804d4`. Fixture: `a0d2994064f95cb8787363ba478852b5800e621be4619d7e05009ad2ff9fe351`.

## 42. STRUCTURAL INDEX SEMANTIC HASH

Smoke: `69646a05dd3c1d6d8e50d1dfcdb90b698050973c88459b609076a59fd00dd673`. Fixture: `a5588e3387d09d343d20f038639838e2cb99b010ca4a94cba61f4fd8b6bbd3a0`.

## 43. TOTAL LEGAL PREDICTION ROWS

Smoke: 136. Fixture: 27. These are identity/time rows, not model examples with features or labels.

## 44. MIN/MAX ROWS PER STAY

Smoke: minimum 1, maximum 12. Fixture: minimum 4, maximum 10. Smoke distribution is machine-generated in `cohort_summary.json`.

## 45. STRUCTURAL HORIZON AVAILABILITY COUNTS

Smoke true counts: recovery24 73, recovery48 22, support24-full-follow-up 73, ICU-time-temporally-eligible 136. Fixture: 15, 3, 15, and 27. These are Vedant’s temporal flags, never final label eligibility.

## 46. STRUCTURAL INDEX OUTCOME-FREE AUDIT

PASS. Strict schema equality and role metadata prohibit SOFA, targets, labels, remaining time, support state/outcome, censoring, model outputs, and future clinical values.

## 47. STRUCTURAL INDEX SPLIT-NEUTRALITY AUDIT

PASS. No split, fold, train, validation, test, or fake anchor-era field exists. Manifest status is `NOT_ASSIGNED_PHASE_10`.

## 48. OUTTIME MODEL-ELIGIBILITY AUDIT

PASS. `outtime` exists only in retained structural cohort metadata with role `STRUCTURAL_LABEL_ONLY` and `model_eligible=false`; it is absent from the per-cutoff index.

## 49. PREDICTION-TIME ROLE AUDIT

PASS. `prediction_time` is `INDEX_ROUTING`, outcome-free, and `model_eligible=false`.

## 50. PATIENT/STAY UNIQUENESS AUDIT

PASS. Subject IDs, stay IDs, subject-to-stay mapping, `(stay_id,prediction_time)`, and `(subject_id,stay_id,grid_index)` identities are unique.

## 51. DUPLICATE / COLLISION AUDIT

PASS. Exact/conflicting duplicate stays, duplicate subjects, multiple episodes per subject, orphan episodes, and cross-subject stay collisions fail closed.

## 52. FUTURE-OUTCOME PERTURBATION RESULT

PASS. Changing a future clinical event value while preserving subject, stay, and boundaries leaves retained cohort and structural index unchanged.

## 53. EVENT-DENSITY PERTURBATION RESULT

PASS. Removing otherwise optional observations leaves membership and index unchanged when identities and boundaries remain valid.

## 54. SOURCE-MANIFEST TAMPER RESULT

PASS by rejection. A mutated Phase-3 artifact hash prevents Phase-4 construction.

## 55. COHORT-CONFIG TAMPER RESULT

PASS by rejection. Changing the adult threshold after build causes `cohort_spec_sha256 mismatch`.

## 56. TIMESTAMP-SPEC TAMPER RESULT

PASS by rejection. A mismatched timestamp implementation hash blocks construction/validation.

## 57. REPRODUCIBILITY / REBUILD RESULT

PASS. Repeated builds with identical source, config, clock, code, command, and timestamp produce identical retained cohort, index, exclusion summary, cohort summary, byte hashes, and semantic hashes.

## 58. COHORT MANIFEST PATH

Primary smoke: `artifacts/data/synthetic/cohorts/smoke/phase4_smoke_v2/synthetic_cohort_manifest_v2.json`. Fixture equivalent exists in the fixture directory.

## 59. COHORT MANIFEST SHA-256

Smoke: `cd9e13b1536de45688d720b2845caf196d210dc288a18c4890afa7098b15d522`. Fixture: `1e52f37bb1a9663c4d093fd6747aeb301359f94682b5aa7f7c96263a00303f32`.

## 60. COHORT SUMMARY PATH

Primary smoke: `artifacts/data/synthetic/cohorts/smoke/phase4_smoke_v2/cohort_summary.json`.

## 61. COHORT SUMMARY SHA-256

Smoke: `ace47fb633d5f3e8419ed41a30261a9bf2ee1e9a45ba63d54dc5459aab167343`. Fixture: `df296687b346aff9e2ba0eb78fdb58afd9c5a4627f5f3654146a53d7bf7bd6e3`.

## 62. VEDANT RECEIVER COMPATIBILITY PRECHECK

PASS for identities/timestamps and temporal flags. Rows are generated by Vedant’s own dataclasses/validator with no fake source era, split, label, or feature. Existing MIMIC-specific acceptance bundles are intentionally not reused wholesale.

## 63. PULKIT IDENTITY / REPLAY COMPATIBILITY PRECHECK

PASS at the structural level. Stable subject ID, stay ID, prediction time, and grid index can route future replay/history work. No HistoryProvider, feature request, current-SOFA provider, support state, or PredictionPipeline input was built.

## 64. EXPECTED SYNTHETIC CONTRACT MIGRATIONS

Phase 10/13 must create the synthetic split and acceptance bundle rather than feed fake anchor years into the real/MIMIC v1 adapter. Phase 5/7 must add the canonical timeline/features without changing Phase-4 membership or cutoff identities.

## 65. FILES CREATED

Created cohort spec; retained-cohort and structural-index schemas; `cohort.py`; `cohort_manifest.py`; cohort CLI; test helper plus four focused test modules; fixture/smoke Phase-4 bundles; and this review.

## 66. FILES MODIFIED

Modified only `src/data/synthetic/__init__.py` to export Phase-4 APIs. No Vedant timestamp, split, tensor, model, evaluation, Pulkit, serving, dashboard, or historical code was modified.

## 67. TESTS ADDED

Added 39 tests spanning population, IDs, boundaries, upstream event consistency, cutoff boundaries/cap/cadence, outcome freedom, perturbations, hashes, tampering, determinism, source-status gates, overwrite protection, smoke flow, and import isolation.

## 68. FOCUSED PHASE-4 TEST RESULTS

PASS: final-tree run completed with 39 passed, 0 failed/skipped/errors in 1.10 s.

## 69. FULL INTEGRATED REGRESSION RESULTS

PASS: final-tree run completed with 1,017 passed, 0 failed, 2 skipped, 0 errors in 14.23 s; measured 16 s from 2026-09-20T16:56:03Z through 16:56:19Z. The skips remain optional dependency blocks.

## 70. STATIC / IMPORT CHECK RESULTS

`python3 -m compileall -q src/data/synthetic scripts/build_synthetic_cohort.py`: exit 0. `git diff --check`: exit 0. No configured Ruff/Black/Mypy gate was found, so none is claimed.

## 71. EXACT COMMANDS EXECUTED

Principal commands: isolated baseline/full pytest command above; focused pytest over the four Phase-4 test modules; `PYTHONPATH=src python3 scripts/build_synthetic_cohort.py --source-manifest <fixture-or-smoke-manifest> --config configs/synthetic/cohort_spec_v2.yaml --output <engineering-output> --mode engineering --generation-timestamp 2026-09-20T17:05:00Z`; manifest validation through `validate_cohort_manifest`; compileall; `git diff --check`; SHA-256 inventory generation.

## 72. UPSTREAM DEFECTS FOUND

None. Both Phase-3 engineering datasets pass manifest, schema, foreign-key, value, field-role, and event-bound validation. The short smoke episode is expected structural exclusion evidence, not a defect.

## 73. UNRESOLVED PARAMETERS

All Phase-3 final scientific parameters and human freeze remain unresolved; consequently no authorized source or final retained cohort exists. Final cardiac taxonomy values will come from the future approved generator config, not this engineering evidence.

## 74. SCIENTIFIC FEASIBILITY ISSUES

None established. Engineering smoke has 23 retained subjects and 136 legal rows, showing software/structural feasibility only. It provides no power, prevalence, realism, or clinical-validity evidence.

## 75. PHASE-4 DEFINITION-OF-DONE CHECKLIST

Implementation DoD passes: prerequisites and manifests inspected; baseline recorded; config-driven population/episode rules; IDs/bounds/events enforced; unchanged Vedant clock reused; at least one cutoff required; exact first/cadence/end/cap preserved; outcome-free pre-split index and provenance created; deterministic flow/exclusions/hashes generated; hostile/focused/full/static checks pass; no downstream scientific work performed. Final execution DoD remains blocked at source authorization.

## 76. PHASE-4 IMPLEMENTATION STATUS

`COMPLETE — ENGINEERING FRAMEWORK AND FIXTURE/SMOKE EXECUTION PASS`.

## 77. FINAL COHORT EXECUTION STATUS

`BLOCKED — AUTHORIZED RAW SYNTHETIC DATA REQUIRED`. Final mode rejects current Phase-3 sources and produces no output.

## 78. STRUCTURAL INDEX STATUS

Engineering index: `ENGINEERING_OUTCOME_FREE_STRUCTURAL_INDEX_SMOKE` / `..._FIXTURE`, validated and hash-bound. Final `OUTCOME_FREE_STRUCTURAL_INDEX` status was not claimed.

## 79. PHASE-5 READINESS

Implementation handoff is ready: scope, source manifest/schema/events, cohort spec, retained identities/bounds, legal cutoff universe, timestamp identity, exclusions, schemas, hashes, and review are available. Final Phase-5 execution remains blocked until an authorized raw dataset and final Phase-4 cohort exist.

## 80. EXPLICIT CONFIRMATION: NO FEATURE GRID WAS CREATED

Confirmed. No timeline normalization, bins, masks, TSLO, padding, feature schema, or tensors were created.

## 81. EXPLICIT CONFIRMATION: NO SOFA SCORES WERE COMPUTED

Confirmed. No SOFA value, component, availability, or observability rule was computed.

## 82. EXPLICIT CONFIRMATION: NO RECOVERY LABELS WERE GENERATED

Confirmed. Temporal follow-up flags are not recovery labels or SOFA availability.

## 83. EXPLICIT CONFIRMATION: NO REMAINING-STAY LABELS WERE GENERATED

Confirmed. No remaining hours or log-transformed target was calculated or stored.

## 84. EXPLICIT CONFIRMATION: NO ORGAN-SUPPORT LABELS WERE GENERATED

Confirmed. `support24_full_followup_available` is only a temporal horizon flag, not support state, at-risk eligibility, censoring, or label.

## 85. EXPLICIT CONFIRMATION: NO TRAIN/VALIDATION/TEST SPLIT WAS CREATED

Confirmed. The index is all-subject/pre-split and contains no fake MIMIC era.

## 86. EXPLICIT CONFIRMATION: NO PREPROCESSOR WAS FIT

Confirmed. No imputer, scaler, encoder, normalization, or forward fill was created.

## 87. EXPLICIT CONFIRMATION: NO MODEL TRAINING WAS PERFORMED

Confirmed. No model, metric, calibration, selection, inference, or feature-importance code ran.

## 88. EXPLICIT CONFIRMATION: PHASE 5+ WAS NOT IMPLEMENTED

Confirmed. Work stops at Phase 4.
