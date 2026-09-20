# Synthetic Canonical Raw-Input Dictionary

Status: Phase-5 raw canonical contract. This is not the Phase-7 ordered model feature schema.

The authoritative machine-readable mapping is `configs/synthetic/concept_map_v1.json`; event-level lineage is `docs/sanskruti/feature_provenance_v1.csv`. All 15 Phase-2 concepts retain identity units and their source event IDs. Point observations become available at `event_time`. Urine output is a `quantity_interval`, with collection bounds preserved and availability at `interval_end == event_time`. Missing observations remain absent rows.

SOFA prerequisites are mapped but not scored: PaO2/FiO2 (respiratory), platelets (coagulation), bilirubin (liver), MAP (cardiovascular), GCS (CNS), and creatinine/urine output (renal). Ventilation and vasopressor context are deferred to Phase 9; complete cardiovascular/respiratory support-aware SOFA readiness must not be claimed.

Structural fields (`subject_id`, `stay_id`, `intime`, `outtime`, `prediction_time`) are routing/structural metadata, not model inputs. `outtime`, duration/remaining time, future SOFA/deltas, support outcomes, censoring, split, generator latent state, trajectory class, final outcome, and future summaries are prohibited.

Potential Phase-7 derivations remain `UNFROZEN`; Phase 5 creates no aggregation, feature order, bins, masks, TSLO, padding, imputation, scaling, or tensors.
