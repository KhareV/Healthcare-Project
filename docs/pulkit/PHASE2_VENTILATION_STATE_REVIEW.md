# Pulkit Phase 2 — Invasive-Ventilation State Review

## Decision

**Ventilation state engine:** IMPLEMENTED AND SYNTHETICALLY TESTED.

**Real MIMIC-Code adapter:** **BLOCKED — PINNED CONCEPT REQUIRED.**

**Real invasive-ventilation state:** **BLOCKED.**

No composite organ-support label, censoring, eligibility, prevalence, model,
serving, API, dashboard, or Phase-3 behavior was implemented.

## Repository state before changes

- 497 tests passed.
- Phase 1 provided the shared draft dictionary/parser and vasopressor state.
- No ventilation state module, pinned concept, SOFA concept version, processed
  ventilation schema, or real adapter existed.
- Existing project context explicitly listed the MIMIC-Code ventilation commit
  and concrete source mapping as unresolved.
- This workspace is not a Git repository.

## Phase-1 compatibility

The Phase-1 vasopressor section and state query were preserved. The existing
parser was extended with a nested `VentilationEventContract`; no second event
dictionary parser was created. All 33 Phase-1 tests pass after the extension.

The shared config remains globally `DRAFT_NON_AUTHORIZING`. Its scope now names
both state foundations, but it still cannot authorize real support labels.

## Normalized state query

The Phase-2 query:

1. requires explicit synthetic or scientific mode;
2. blocks scientific mode unless the ventilation section is `FROZEN_REAL`;
3. validates internal categories, interval structure, timezone form, concept
   version, adapter version, and provenance;
4. rejects duplicate/conflicting normalized records;
5. isolates the requested stay;
6. evaluates only records active at cutoff `t`;
7. returns invasive ON when any active normalized category is `INVASIVE`;
8. leaves NIV, HFNC, and ordinary oxygen as invasive OFF; and
9. returns sorted baseline evidence and dictionary/concept lineage.

Future intervals are not returned or used to change baseline state.

## State and risk semantics

- Invasive ON at `t` → `vent_at_risk = false`.
- Invasive OFF at `t` → `vent_at_risk = true` only for a validly classified
  synthetic/frozen concept state.
- Unknown categories fail closed and never become OFF.
- Continued invasive ventilation is baseline ON; Phase 2 emits no initiation.
- Simultaneous synthetic categories are allowed; verified invasive presence
  makes invasive state ON. Real overlap resolution remains concept-dependent.

## Boundary, duplicate, and SOFA blockers

- **BLOCKED — VENTILATION INTERVAL BOUNDARY SEMANTICS REQUIRED.**
- **BLOCKED — VENTILATION CONCEPT DUPLICATE POLICY REQUIRED.**
- **BLOCKED — SOFA / SUPPORT VENTILATION CONCEPT ALIGNMENT REQUIRED.**

Synthetic tests use `[start, end)` only and test exact start/end, adjacent
intervals, zero duration, overlap, and transitions. This convention cannot
authorize real execution.

## State-transition fixture

`tests/fixtures/ventilation_state/state_transitions_v1.json` models synthetic:

`NON_INVASIVE → INVASIVE → NON_INVASIVE`

It proves OFF before invasive support, ON during it, and OFF after it without
generating a future 24-hour label.

Additional tests cover no support, NIV, HFNC, ordinary oxygen, future invasive
start, prior stop, already-ON continuation, other-stay isolation, future-row
perturbation, record ordering, unknown categories, version mismatch, boundary
cases, duplicates, Phase-3 field absence, model-input safety, and absence of a
prevalence path.

## Event-dictionary validation

The ventilation section is component-level `DRAFT_NON_AUTHORIZING`. Frozen-real
validation requires:

- verified source repository, exact commit, concept and output schema;
- verified adapter and raw-to-internal category mapping;
- explicit invasive category and NIV/HFNC/oxygen exclusion semantics;
- hard failure for unknown values;
- verified event/boundary and duplicate/overlap semantics;
- verified SOFA/support version alignment; and
- Sanskruti approval evidence.

Placeholder, unknown, synthetic, unverified, malformed-commit, and mismatched
SOFA-version attacks fail.

## Sanskruti review — pending

Sanskruti must verify:

1. exact MIMIC-Code source and commit;
2. exact derived ventilation concept;
3. actual output schema and representation;
4. event/interval and boundary semantics;
5. raw category mapping;
6. invasive versus NIV distinction;
7. invasive versus HFNC distinction;
8. invasive versus ordinary oxygen distinction;
9. duplicate, overlap, start, and stop semantics;
10. compatibility with SOFA respiratory context; and
11. complete provenance and evidence references.

## Vedant compatibility notes

- Query accepts canonical cutoff `t`; timestamp generation was not changed.
- `vent_at_risk` is baseline-only and is not `organ_support_eligible`.
- Future changes do not affect baseline state.
- Dictionary version/hash and concept/adapter versions are returned.
- Phase-7 target and synthetic support handoff were not changed.
- No ventilation evidence or future onset was added to `CanonicalBatch`.

## Scope audit

No Phase-2 code computes `Y_support`, `organ_support_eligible`, censor reason,
future-onset label, prevalence, class balance, calibration, threshold, or model
input features. No Vedant/Sanskruti scientific module was modified.

## Handoff status

The normalized interface, state query, evidence fields, draft dictionary
section, fixtures, provenance blockers, and reviews are ready as a framework
handoff. Phase 3 must wait for the reviewed real concept if it requires real
scientific execution.

