# Pulkit Phase 1 — Vasopressor State Review

## Decision

**Support-state engine:** IMPLEMENTED AND SYNTHETICALLY TESTED.

**Real vasopressor event dictionary:** **BLOCKED — ACTUAL MIMIC SCHEMA / MAPPING REQUIRED.**

**Real vasopressor state definition:** **BLOCKED.** No real prevalence, label generation,
training, or registry promotion is authorized by the checked-in draft.

The Master Plan is the scientific authority. It fixes a binary OFF-to-ON
vasopressor-support component and says the vasopressor set/state rule is frozen
after schema verification. The Work Division assigns the implementation to
Pulkit and requires active-at-cutoff state to come from verified valid infusion
intervals/rates, reviewed by Sanskruti and Vedant. Neither document supplies the
concrete MIMIC mapping or interval/rate policy.

## Repository audit before changes

- Vedant's `src/data/timestamps.py` already exposes only temporal
  `support24_full_followup_available`; it was not modified or reinterpreted.
- `src/data/support_labels.py` is explicitly a manual synthetic downstream
  handoff and states that no OFF/ON derivation is implemented.
- No prior `src/labels/support_state.py`, `src/labels/event_dict.py`, or
  `configs/event_dict_v1.yaml` existed.
- `docs/CODEX_PROJECT_CONTEXT_V1.md` already marked the qualifying vasopressor
  set, raw source fields, units/rates, interval semantics, and final OFF/ON rule
  unresolved.
- Baseline: 464 tests passed before Phase-1 changes.
- This workspace is not a Git repository, so no code commit or worktree status
  can be recorded.

## Added contracts

### Draft event dictionary

`configs/event_dict_v1.yaml` is `DRAFT_NON_AUTHORIZING`. Its exact-file SHA-256
at implementation time is:

`fec7432cafab7e3fe9a76c24908a931d04e2e9a97b837fe77fb52cf83091e425`

This is a **draft configuration hash**, not the final scientific
event-dictionary hash. The validator rejects unknown schema/rule versions,
draft authorization, empty frozen mappings, missing frozen provenance,
unverified boundary/activity/duplicate policies, pending reviewer approvals,
and placeholders in a frozen mapping.

The file contains no ventilation derivation, composite label, censoring rule,
real agent identifier, MIMIC table/field, unit, or rate threshold.

### Normalized internal support-event contract

`NormalizedActiveInterval` has internal engineering fields:

- `stay_id`
- `agent_key`
- `interval_start`
- `interval_end`
- `source_event_ref`
- `normalization_provenance_version`

These names are **not claims about raw MIMIC columns**. Each interval asserts
that a separate, not-yet-implemented source-normalization layer has already
verified qualifying activity across its span. The state layer never derives
activity from raw rate, amount, stop, pause, or order records.

Malformed timestamps, missing identifiers/provenance, end-at-or-before-start,
timezone inconsistencies, repeated source references, and exact duplicate
normalized intervals fail closed. This is validation of the normalized input,
not a scientific raw-source deduplication policy.

## State-query algorithm

1. Require an explicit `SYNTHETIC` or `SCIENTIFIC` execution mode.
2. Scientific mode requires a frozen, authorizing, provenance-complete event
   dictionary. The checked-in draft therefore refuses scientific execution.
3. Validate every normalized interval without silently repairing it.
4. Restrict to the requested stay and the mode-specific qualifying keys.
5. Evaluate activity at the cutoff using the mode-specific boundary policy.
6. Collapse any number of active qualifying intervals to one binary `ON`
   state; otherwise return `OFF`.
7. Set `vaso_at_risk = false` for `ON` and `true` for `OFF`.
8. Return sorted active interval references plus cutoff, dictionary version and
   hash, state-rule version, mapping provenance, boundary policy, and mode.

The query never generates a future outcome, counts drugs, calculates dose
intensity, or interprets a second pressor as a new support initiation.

## Boundary and activity status

- **Real boundary convention:** **BLOCKED — VASOPRESSOR INTERVAL-BOUNDARY
  SEMANTICS REQUIRED.**
- **Real rate/activity rule:** **BLOCKED — INFUSION ACTIVITY/RATE RULE
  REQUIRED.**
- **Real duplicate/conflict policy:** **BLOCKED — SUPPORT EVENT DEDUPLICATION
  RULE REQUIRED.**
- The synthetic profile explicitly uses `[start, end)` (`CLOSED_OPEN`) only to
  test the engine. It is tagged synthetic and cannot authorize real execution.
- Exact start, exact end, and interior behavior are tested under that synthetic
  convention. Scientific boundary tests instead verify that the draft blocks.

## Synthetic fixtures and tests

The fixture `tests/fixtures/support_state/state_cases_v1.json` uses only
symbolic `QUALIFYING_AGENT_A` / `NONQUALIFYING_AGENT_X` records and is marked
`NOT_MIMIC`.

Tests cover no event; active/past/future/non-qualifying events; multiple and
overlapping qualifying events; already-ON risk status; future perturbation;
order invariance; other-stay isolation; non-qualifying attacks; explicit
boundaries; malformed/missing timestamps; invalid intervals/stays; duplicate
and conflicting normalized events; timezone mismatch; config hash/provenance;
unknown versions; invalid freeze attempts; and scientific-mode denial.

No real prevalence, class balance, support incidence, calibration, threshold,
or test-set result was inspected.

## Sanskruti review checkpoint — pending

Sanskruti must supply and verify, against the actual processed/MIMIC schema:

1. the exact qualifying vasopressor source mapping;
2. actual source and field names;
3. the clinical event-time field;
4. rate/unit and conversion semantics;
5. valid active-infusion derivation, including zero/missing/paused/stopped cases;
6. interval start/stop and exact-boundary semantics;
7. duplicate/conflict handling;
8. mapping/derivation provenance versions; and
9. compatibility with the shared timeline and feature-provenance layer.

Approval must include a repository-relative evidence reference. Until then the
dictionary remains draft.

## Vedant review checkpoint — pending

Vedant must verify:

1. the query consumes canonical cutoff `t` without changing timestamp code;
2. `support24_full_followup_available` remains temporal-only;
3. ON at `t` implies not at risk for a new vasopressor initiation;
4. post-cutoff events do not alter baseline state;
5. output metadata can feed later organ-support eligibility/label logic;
6. dictionary version/hash can be carried through registry lineage; and
7. draft/synthetic configurations cannot authorize real training, prevalence,
   or label generation.

## Unlock conditions

The real Phase-1 freeze requires all of the following through reviewed,
versioned change control: actual source schema, exact qualifying mapping,
event-time field, unit/rate semantics, activity derivation, boundary convention,
duplicate/conflict policy, full provenance, Sanskruti approval, and Vedant
approval. Only then may `status`, authorization, verified fields, approvals,
and the final hash be updated before prevalence inspection.

## Scope exclusions preserved

Phase 1 does not implement ventilation state, future OFF-to-ON outcome logic,
composite organ-support labels, censoring/eligibility, prevalence, class
weights, model changes, serving, explanations, API/dashboard work,
calibration/thresholding, or final test access. SOFA+2 and dose-escalation
semantics are absent.

