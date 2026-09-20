# Codex Project Context V2 — Active Scope Amendment

**Status:** `DRAFT COMPLETE — TEAM FREEZE REQUIRED`  
**Primary authority:** [`governance/project_scope_v2.md`](governance/project_scope_v2.md)  
**Machine contract:** [`../configs/governance/project_scope_v2.json`](../configs/governance/project_scope_v2.json)

This is the active discovery entrypoint for the Sanskruti rebuild. It amends, but does not erase, `CODEX_PROJECT_CONTEXT_V1.md` and the two Final v1.0 DOCX documents.

## Active project identity

- Final data source: `AUTHORIZED_FINAL_SYNTHETIC_DATA`; engineering fixtures are a distinct non-scientific scope.
- Population: adult cardiac/heart-disease domain; exact subtype taxonomy is deferred and heart-failure-only is not approved.
- Episode: one retained synthetic ICU-like current critical-care stay per subject with explicit `subject_id`, `stay_id`, `intime`, and `outtime`.
- Product: **RETROSPECTIVE SEQUENTIAL REPLAY — NOT REAL-TIME CLINICAL PREDICTION**.
- Clinical status: methodological software prototype evaluated on synthetic trajectories; not clinically validated.

## Retained v1 scientific architecture

- `t_k = intime + 24h + 6h*k`, `k=0..11`, retained only at `t_k <= outtime-6h`.
- `(t-48h,t]` as eight left-open/right-closed six-hour bins.
- Distinct genuine-observation mask, TSLO, structural padding, and clinical missingness.
- Independent continuous `SOFA(t+24)-SOFA(t)` and `SOFA(t+48)-SOFA(t)`.
- `log1p((outtime-t) hours)` until the current synthetic episode ends.
- Eligible OFF→ON vasopressor or invasive-ventilation initiation in `(t,t+24h]`.
- One canonical row with task-specific eligibility; XGBoost/GRU information parity.
- Naive/XGBoost/GRU candidates; one fixed LSTM sensitivity per task.
- Validation-only exact-30 searches, selection, support calibration/threshold; sealed final test.
- Stay-balanced metrics, stay-cluster bootstrap, selected-family explanations, PredictionPipeline, API, and replay.

## Superseded MIMIC-specific requirements

MIMIC access/SQL/item IDs/tables/counts, PhysioNet credentials, anchor-era split, MIMIC-Code provenance, and MIMIC source mappings do not apply to the final synthetic run. They are replaced by generator/schema/config/seed/dataset provenance, a future synthetic subject split, a synthetic six-component SOFA specification, and synthetic support-event mappings.

## Current phase boundary

Phase 1 creates governance only. It does not authorize data generation, cohort/feature/SOFA/label implementation, model training/search, runtime contract changes, artifact promotion, or final-test access. See [`sanskruti/PHASE1_SCOPE_V2_REVIEW.md`](sanskruti/PHASE1_SCOPE_V2_REVIEW.md).

## Historical/reference repository

`Heart-Failure-Recovery-and-Deterioration-Trajectory-Prediction` is historical reference material, not authoritative final scope and not an integration source.

## Required reading order

1. `docs/governance/project_scope_v2.md`
2. `configs/governance/project_scope_v2.json`
3. `docs/governance/project_scope_v2_delta.csv`
4. `docs/governance/project_scope_v2_impact.md`
5. Unchanged current Vedant/Pulkit contracts
6. `docs/CODEX_PROJECT_CONTEXT_V1.md` for retained historical baseline detail

No team approval is asserted. Until genuine review is recorded, the v2 package remains a complete draft rather than a frozen scientific contract.
