# Phase 13 Structural Leakage Audit

All acceptance decisions use structural invariants, provenance, adversarial perturbation tests, and exact reconstruction. They do not use correlations, performance magnitude, prevalence targets, arbitrary missingness thresholds, or CI-width ratios.

| ID | Status | Evidence |
|---|---|---|
| D01 | PASS | canonical_cutoffs=12222 max_per_stay=12 |
| E01 | PASS | 8x21; 48h; 6h; left-open/right-closed |
| F01 | PASS | padding/mask/TSLO/support-known-state checked for every cell |
| K01 | PASS | dispositions={'INTERNAL_NEGATIVE_FULL_FOLLOWUP': 4368, 'INTERNAL_POSITIVE_INITIATION': 3459, 'INTERNAL_CENSORED_EARLY_EXIT_NO_INITIATION': 4043, 'INTERNAL_NOT_AT_RISK_BOTH_COMPONENTS_ON': 352} |
| L01 | PASS | prohibited predictor hits=[] |
| Z01 | PASS | focused JUnit={'tests': 419, 'failures': 0, 'errors': 0, 'skipped': 0} |

The model input contract excludes targets, eligibility, split, IDs, outtime, future values, latent state, hazards, and future support duration. Future events may change labels but must not change cutoff-time features; the focused hostile suite is the executable evidence.
