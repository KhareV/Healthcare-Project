# Pulkit Phase 3 — Composite Organ-Support Label Review

## Decision

**Composite label engine:** IMPLEMENTED AND SYNTHETICALLY TESTED.

**Real organ-support label artifact:** **BLOCKED — REAL STATE DEFINITIONS / DATA
REQUIRED.**

**Endpoint freeze and prevalence:** DEFERRED TO PULKIT PHASE 4.

The implementation composes the Phase-1 vasopressor and Phase-2 invasive-
ventilation state artifacts. It does not rederive either component state.

## Repository state before changes

- 535 tests passed.
- Phase 1 and Phase 2 were draft/synthetic and non-authorizing.
- Vedant's timestamp contract exposed temporal follow-up separately from final
  support eligibility.
- Vedant's Phase-7 consumer already accepted `SupportLabelRecord` containing
  label, eligibility, and audit-only metadata.
- The real state definitions, processed data, endpoint freeze, and reviewer
  approvals were unavailable.
- This workspace is not a Git repository.

## Frozen scientific logic implemented

For each canonical cutoff `t`:

1. Consume version/hash-bound Phase-1 and Phase-2 baseline state results.
2. Mark each component at risk only when its baseline state is OFF.
3. If both are ON, return null/ineligible with a distinct not-at-risk reason.
4. Accept only verified component OFF-to-ON evidence satisfying
   `t < onset <= t+24h` for a component that was at risk at `t`.
5. Reject post-outtime evidence. Onset exactly equal to outtime fails closed
   because the tie policy is unresolved.
6. If an eligible onset occurs before exit, return positive even without full
   24-hour follow-up.
7. Without a positive, return negative only when temporal 24-hour follow-up is
   complete, including equality at `outtime == t+24h`.
8. Otherwise return null/ineligible censored.

Only one composite binary label is produced when both components initiate.
Earliest qualifying onset evidence is retained deterministically.

## Temporal flag separation

`SupportLabelCutoff.from_prediction_timestamp()` carries Vedant's
`support24_full_followup_available` flag and validates it against outtime. The
flag is used only for the negative-versus-censored decision:

- full follow-up plus no onset can be negative;
- incomplete follow-up plus a prior positive remains positive;
- full follow-up does not override both-components-ON not-at-risk status.

`src/data/timestamps.py` was not modified.

## Audit-only onset evidence

`ComponentOnsetEvidence` is a normalized audit contract supplied by a
component-specific adapter. It carries stay, component, onset, transition kind,
source/provenance, state-rule and definition/adapter versions, dictionary
version/hash, evidence-contract version, and execution mode.

Phase 3 does not infer raw infusion or ventilation transitions. Evidence marked
`NO_COMPONENT_OFF_TO_ON` can represent continuation, a second pressor, or an
invasive mode/settings change and cannot create a positive.

## Internal reason vocabulary

The result distinguishes:

- positive initiation;
- negative with complete follow-up;
- censored early exit without initiation; and
- not at risk because both components were already ON.

These are typed internal values under
`organ_support_audit_disposition_internal_v1`. The serialized interoperability
vocabulary is **INTERNAL NON-FINAL — REVIEW REQUIRED**.

Unknown component states fail closed and are never coerced to OFF.

## Existing Phase-7 handoff

`to_phase7_support_label_record()` returns Vedant's existing
`data.support_labels.SupportLabelRecord` rather than creating a competing
format. Model-visible fields remain only:

- binary label for eligible rows, otherwise null; and
- `organ_support_eligible`.

Baseline states, onset times, reason, and lineage remain inside
`audit_metadata`. `build_phase7_handoff_records()` enforces one record per
`(stay_id, prediction_time)`, requires a split assignment, and sorts output
deterministically.

No future audit field was added to `CanonicalBatch`, the GRU, or the flattened
model feature path.

## Shared event dictionary

`configs/event_dict_v1.yaml` now documents the Phase-3 composite rule, exactly
24-hour horizon, exclusive/inclusive boundaries, positive precedence, negative
follow-up requirement, censoring, both-ON rule, unresolved policies, reviewer
status, and blocked real artifact.

Current exact-file draft SHA-256:

`717f6183ad8014602b3cc32eeaa3a810eab8a26a19937bda8e214200bb2f9620`

This is not a frozen endpoint hash. The validator actively rejects a Phase-4
freeze attempt in Phase 3.

## Unresolved real-label policies

- **BLOCKED — SUPPORT ONSET / ICU-EXIT SAME-TIMESTAMP POLICY REQUIRED.**
- **BLOCKED — UNKNOWN SUPPORT-STATE LABEL POLICY REQUIRED.**
- **UNLOCKED/BLOCKED — FINAL AUDIT REASON VOCABULARY REQUIRES REVIEW.**
- Phase-1 real vasopressor mapping remains blocked.
- Phase-2 real MIMIC-Code concept and SOFA alignment remain blocked.
- Real canonical state/onset inputs and final artifact do not exist.

## Synthetic truth table

`tests/fixtures/organ_support/truth_table_v1.json` covers the ten required core
cases across all baseline state combinations, component-specific initiations,
dual initiation, full negative follow-up, early censoring, positive before
early exit, and both-ON not-at-risk status.

Additional tests cover exact horizon boundaries, post-outtime evidence,
onset/outtime ties, temporal-flag separation, transition continuity/escalation,
other stays, ordering, duplicates, version attacks, unknown state, canonical
uniqueness, Phase-7 adaptation, future feature parity, and scope exclusions.

No real prevalence, censoring rate, at-risk rate, positive rate, class weight,
calibration, threshold, or model metric was calculated.

## Vedant review — pending

Vedant must verify:

1. Phase-1/2 baseline semantics are consumed rather than reimplemented;
2. at least one OFF component gives at-risk possibility;
3. both ON is ineligible/not at risk;
4. the future horizon is exactly `(t,t+24h]`;
5. continuation/escalation is not initiation;
6. positive before early exit remains positive;
7. early exit without a positive is censored;
8. complete follow-up without a positive is negative;
9. the temporal flag is not final eligibility;
10. audit-only fields do not enter model inputs; and
11. generated records match the Phase-7 consumer interface.

## Sanskruti compatibility — pending

Sanskruti must verify event-time and ICU-episode provenance, absence of
post-outtime dynamic events, state/onset adapter provenance, canonical
stay/cutoff identity, and eventual processed-schema compatibility. No upstream
timeline was changed.

## Phase-18 status

The framework defines the artifact shape and state-machine behavior needed by
the Phase-18 dependency, but does not resolve it. Phase 18 remains:

**BLOCKED — PULKIT REAL ORGAN-SUPPORT LABEL ARTIFACT REQUIRED.**

Real Phase-1 and Phase-2 definitions, Phase-4 freeze, deterministic real
generation, hashing, and Vedant spot-check are still required.

## Handoff to Phase 4

Phase 3 hands off a tested composite engine, internal audit contract, Phase-7
adapter, truth table, draft event-dictionary section, and explicit blocker
list. Phase 4 owns formal team review, endpoint freeze, frozen hash, and only
then prevalence inspection.

