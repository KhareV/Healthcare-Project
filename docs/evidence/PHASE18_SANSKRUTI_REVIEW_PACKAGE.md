# Phase 18 Sanskruti Handoff Review Package

## Current disposition

- Acceptance framework: implemented.
- Real upstream handoff: missing.
- Real acceptance: `BLOCKED`.
- Phase 19 authorization: false.
- Protected test labels and model performance: not accessed.
- Current report SHA-256: `3a7ad8edee17a4d1844e82695498ca41e93818924828c18a3a9690a7523b93b4`.
- Registered snapshot: `artifacts/data_acceptance/history/real_data_acceptance_v1_3a7ad8edee17a4d1.json`.

## Sanskruti review checklist

Provide one hash-bound `real_data_input_manifest_v1` containing:

- canonical development artifact and processed-schema sidecar;
- ordered `feature_schema_v1` and feature provenance dictionary;
- outcome-free all-split structural index;
- real `split_v1.csv` and metadata sidecar;
- SOFA provenance, exact pinned MIMIC-Code commit, and six component flags;
- independent recovery formulas/eligibility and ICU-time labels;
- frozen valid ranges and structural leakage evidence; and
- train-only preprocessing provenance or an explicit not-yet-fit interface.

Review the generated inventory hashes, timestamp parity, feature order, mask/TSLO/padding behavior, label formula checks, split compatibility, loader projection, and XGBoost/GRU information parity. Restricted row-level clinical extracts must not be committed as review evidence.

## Pulkit dependency

The generic receiver contract is implemented and tested, including positive-before-censor and already-ON cases. Real support integration remains `BLOCKED — PULKIT REAL ORGAN-SUPPORT LABEL ARTIFACT REQUIRED`. Vedant has not recreated the vasopressor or ventilation state machine.

## Phase 19 handoff rule

Do not run scientific validation searches until the report is exactly `ACCEPTED` and the exact registered report file hash passes `require_accepted_real_data`.
