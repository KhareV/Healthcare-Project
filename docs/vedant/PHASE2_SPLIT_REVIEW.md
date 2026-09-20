# Phase 2 Subject Split — Reviewer Package

**Owner:** Vedant Khare  
**Mandatory reviewer:** Sanskruti Satish Shete  
**Contract:** `configs/split_spec_v1.yaml`  
**Contract SHA-256:** `c2257f16c3a9dc0c54bb9810bca882ae52a22f4908c6fb7b32820b8e4b1200e1`

## Review decision requested

Sanskruti should verify that the internal `subject_id`, optional `stay_id`, and exact `anchor_year_group` strings match the frozen retained-cohort output, and that the resulting subject mapping can be joined safely to features and labels before any train-derived preprocessing is fitted.

The real retained cohort is absent. Therefore final generation is **BLOCKED — REAL COHORT ARTIFACT REQUIRED**, and `artifacts/splits/split_v1.csv` plus its metadata sidecar have intentionally not been created.

## Frozen mapping

| Exact `anchor_year_group` | Split |
|---|---|
| `2008-2010` | `train` |
| `2011-2013` | `train` |
| `2014-2016` | `validation` |
| `2017-2019` | `test` |

No whitespace or alternative era normalization is performed. This is a subject-level **coarse temporal holdout**, not exact patient-by-patient chronological deployment simulation.

## Input and duplicate policy

`CohortSubjectRecord(subject_id, anchor_year_group, stay_id=None)` is an internal engineering contract. The implementation does not query MIMIC or choose the first ICU stay.

- Exact duplicate subject/era/stay rows—such as timestamp-expanded rows—deduplicate to one assignment and increment `duplicate_input_count`.
- A subject with conflicting eras or multiple distinct retained `stay_id` values fails validation.
- Null/invalid subjects, null eras, unsupported eras, mixed identifier types, and invalid optional stay identifiers fail validation.
- Extra label, outcome, missingness, timestamp, model-performance, or test-result columns are intentionally discarded before assignment.

## Authoritative API

The only implementation is `src/data/split.py` (SHA-256: `58caffd92eab18e400a657a371f4a59564acd34000ba8920fec0fa08dbbce2ca`). It provides:

- exact era mapping with `era_to_split`;
- mapping-to-contract extraction with `records_from_mappings`;
- structural auditing/deduplication with `audit_cohort_records`;
- deterministic assignment with `generate_subject_split` and `generate_subject_split_from_mappings`;
- pairwise subject-set and overlap validation;
- canonical CSV serialization and exact-byte SHA-256;
- split writing with source/spec lineage metadata;
- split loading, expected-hash assertion, and fail-closed integrity verification.

Final artifact columns are exactly `subject_id,split,anchor_year_group`, ordered deterministically by subject ID. The sidecar records split/spec versions and hashes, source cohort version/hash, counts, frozen mapping, creation timestamp, optional code commit, and the phrase “coarse temporal holdout.”

## Synthetic evidence and QA

The fixture is explicitly named `cohort_rows_synthetic_v1.json`; its expected artifact is `expected_split_synthetic_v1.csv`, not `split_v1.csv`.

Hand-counted synthetic QA:

- input rows: 5;
- unique subjects: 4;
- train subjects: 2;
- validation subjects: 1;
- test subjects: 1;
- unique subjects per era: 1 each;
- duplicate input rows: 1;
- missing/unsupported/conflicting records: 0.

Synthetic expected CSV SHA-256: `462f73979e63c7169e41f514ea4c5a9273759f27ed6263dd8ada37729f3d07d0`.

Tests prove input reordering yields identical rows, bytes, and hash; all split subject sets are pairwise disjoint; timestamp multiplicity does not affect assignment; changed labels/outcomes/missingness/performance/test fields do not affect assignment; content or metadata tampering fails closed.

## Real freeze procedure

When Sanskruti supplies the frozen real cohort:

1. Verify exact `subject_id` and `anchor_year_group` representation and source cohort SHA-256/version.
2. Run the authoritative generator; resolve any reported upstream consistency failure without dropping records arbitrarily.
3. Review structural counts and pairwise subject isolation.
4. Write `artifacts/splits/split_v1.csv` and `split_v1.metadata.json` through `write_split_artifacts`.
5. Record and independently verify the exact split SHA-256 before downstream preprocessing.
6. Downstream consumers call `assert_split_hash`/`verify_split_artifacts` and fail on replacement or incompatible metadata.

No source commit is currently recordable because the project directory is not a Git repository. No label, preprocessing, model, calibration, or test-result logic is present.

