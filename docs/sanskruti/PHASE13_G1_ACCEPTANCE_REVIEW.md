# Phase 13 — Synthetic G1 Data/Label Acceptance Review

## Result

**ACCEPTED_SYNTHETIC_DATA_LABEL_FREEZE.** All critical structural, temporal, leakage, provenance, parity, and reproducibility invariants passed. The full 2,000-subject pipeline reproduced exact scientific bytes after unrelated global RNG perturbation. Phase-12 parents were not changed, so its governed XGBoost results remain valid.

## Governance boundaries

- Final-test state remains `NEVER_OPENED`; no test model input, prediction, metric, prevalence, or error analysis was produced.
- Model performance was not an acceptance criterion.
- No model training, GRU search, family selection, calibration, threshold selection, G2, G3, or Phase 14 work was performed.
- G1 accepts methodology and data/label consistency for a synthetic cardiac benchmark; it is not clinical validation or deployment authorization.

## Evidence

The authoritative evidence is under `docs/evidence/data/`, with machine-readable invariants and the authorizing marker under `artifacts/acceptance/`. The dataset card records intended and prohibited uses. The golden-case report contains the ten required deterministic cases. The circularity report confirms physiology/support-first generation and derived labels.
