# Phase 13 Synthetic Data QA

Synthetic methodology benchmark only; not real-patient evidence, bedside validation, causal treatment evidence, external validation, or deployment authorization. Generated support hazards/rates are project-designed, not observed treatment statistics.

| Measure | Value |
|---|---|
| scope | "SYNTHETIC_ADULT_CARDIAC_RETROSPECTIVE_FORECASTING" |
| subjects | 2000 |
| episodes | 2000 |
| retained_stays | 2000 |
| canonical_rows | 12222 |
| raw_events | 670303 |
| support_intervals | 63663 |
| condition_groups | {"SYNTHETIC_ARRHYTHMIA": 510, "SYNTHETIC_HF": 801, "SYNTHETIC_ISCHEMIC": 689} |
| cohort_exclusions | {} |
| support_dispositions | {"INTERNAL_CENSORED_EARLY_EXIT_NO_INITIATION": 4043, "INTERNAL_NEGATIVE_FULL_FOLLOWUP": 4368, "INTERNAL_NOT_AT_RISK_BOTH_COMPONENTS_ON": 352, "INTERNAL_POSITIVE_INITIATION": 3459} |
| exact_clone_count | 0 |
| feature_dimension | 21 |
| split_subject_counts | {"test": 300, "train": 1400, "validation": 300} |
| test_accessed | false |
| phase12_performance_used_for_acceptance | false |
| focused_junit | {"errors": 0, "failures": 0, "skipped": 0, "tests": 419} |

## Critical invariants

| ID | Category | Status | Evidence |
|---|---|---|---|
| A01 | generator | PASS | seed=20260921; generated_subjects=2000 |
| A02 | generator | PASS | exported_fields=['age_years', 'agent_key', 'cardiac_condition_group', 'concept_code', 'episode_sequence', 'event_id', 'event_time', 'generator_component', 'generator_subject_stream_id', 'interval_end', 'interval_start', 'intime', 'normalization_provenance_ref', 'outtime', 'rate_unit', 'rate_value', 'respiratory_category', 'sex_category', 'stay_id', 'subject_id', 'support_category', 'support_event_id', 'support_type', 'unit', 'value_numeric', 'value_text'] |
| A03 | generator | PASS | exact_identity_free_full_subject_clones=0 |
| B01 | identity | PASS | subject/stay IDs unique and one episode per subject |
| B02 | identity | PASS | all dynamic/support rows resolve |
| B03 | identity | PASS | event/support episode bounds |
| B04 | scope | PASS | adult cardiac groups={'SYNTHETIC_HF': 801, 'SYNTHETIC_ISCHEMIC': 689, 'SYNTHETIC_ARRHYTHMIA': 510} |
| B05 | cohort | PASS | generated=2000 retained=2000 exclusions={} |
| D01 | timestamps | PASS | canonical_cutoffs=12222 max_per_stay=12 |
| E01 | features | PASS | 8x21; 48h; 6h; left-open/right-closed |
| F01 | missingness | PASS | padding/mask/TSLO/support-known-state checked for every cell |
| H01 | sofa | PASS | single frozen six-domain SOFA spec; computed recovery baselines within 0..24 |
| I01 | recovery | PASS | independent horizon masks; raw deltas within -24..24 |
| J01 | icu_time | PASS | every row equals log1p((outtime-t) hours) without rounding |
| K01 | support_labels | PASS | dispositions={'INTERNAL_NEGATIVE_FULL_FOLLOWUP': 4368, 'INTERNAL_POSITIVE_INITIATION': 3459, 'INTERNAL_CENSORED_EARLY_EXIT_NO_INITIATION': 4043, 'INTERNAL_NOT_AT_RISK_BOTH_COMPONENTS_ON': 352} |
| L01 | leakage | PASS | prohibited predictor hits=[] |
| M01 | split | PASS | exact label-blind subject split regenerated |
| N01 | preprocessing | PASS | fit subjects=1400 all train |
| N02 | preprocessing | PASS | recovery scaler/support weight train-only; no ICU target scaler artifact |
| O01 | information_parity | PASS | complete train/validation XGB-GRU reconstruction |
| Q01 | lineage | PASS | Phase-12 exact governed budget and lineage |
| Q02 | lineage | PASS | all Phase-10 child hashes resolve |
| T01 | test_governance | PASS | test_accessed=false; no test artifact; no G3 marker |
| Z01 | adversarial_tests | PASS | focused JUnit={'tests': 419, 'failures': 0, 'errors': 0, 'skipped': 0} |
| R01 | reproducibility | PASS | full raw-to-pre-split plus split/preprocessor byte replay |
