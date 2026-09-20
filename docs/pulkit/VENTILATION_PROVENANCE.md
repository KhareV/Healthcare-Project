# Invasive-Ventilation Concept Provenance

## Current scientific status

**REAL INVASIVE-VENTILATION STATE: BLOCKED — PINNED MIMIC-CODE CONCEPT
REQUIRED.**

The repository contains no verified MIMIC-Code source, commit, derived concept,
output schema, category mapping, interval/event semantics, or processed-schema
adapter for invasive ventilation. This document records required provenance; it
does not fill absent scientific values from memory.

## Authority and intended reuse

The Final Implementation Master Plan requires a pinned, versioned,
MIMIC-Code-derived concept that distinguishes invasive ventilation from
non-invasive ventilation, HFNC, and ordinary supplemental oxygen. The same
scientific concept is intended to be reviewed for both:

- organ-support invasive-ventilation state; and
- SOFA respiratory-component invasive-ventilation context.

No SOFA concept version is presently available, so alignment is:

**BLOCKED — SOFA / SUPPORT VENTILATION CONCEPT ALIGNMENT REQUIRED.**

No change was made to SOFA code.

## Real provenance record — unresolved

| Required item | Current value |
|---|---|
| Concept name | UNRESOLVED — REQUIRES VERIFICATION |
| Source repository | UNRESOLVED — REQUIRES VERIFICATION |
| Exact commit/hash | UNRESOLVED — REQUIRES VERIFICATION |
| Concept/derivation version | UNRESOLVED — REQUIRES VERIFICATION |
| Output schema/version | UNRESOLVED — REQUIRES VERIFICATION |
| Output representation | UNRESOLVED — point/event/interval form unknown |
| Event/interval semantics | UNRESOLVED — REQUIRES VERIFICATION |
| Raw-category mapping | UNRESOLVED — REQUIRES VERIFICATION |
| Invasive definition | UNRESOLVED — REQUIRES VERIFICATION |
| NIV treatment | Must be explicitly verified as non-invasive for this endpoint |
| HFNC treatment | Must be explicitly verified as non-invasive for this endpoint |
| Ordinary oxygen treatment | Must be explicitly verified as non-invasive for this endpoint |
| Unknown-value policy | Must fail closed; real adapter policy still requires review |
| Duplicate/overlap policy | UNRESOLVED — REQUIRES VERIFICATION |
| Adapter version | UNRESOLVED — real adapter not implemented |
| SOFA compatibility evidence | UNRESOLVED — REQUIRES VERIFICATION |
| Sanskruti review | PENDING |

## Internal normalized contract

`NormalizedVentilationInterval` is an **INTERNAL ENGINEERING CONTRACT**, not a
claim about MIMIC or MIMIC-Code output columns. It contains:

- stay identity;
- internal respiratory category;
- normalized interval start/end;
- source-state provenance reference;
- concept version;
- adapter version; and
- normalization-provenance reference.

The real adapter remains blocked until the actual concept output is inspected.
Unknown categories fail; they are never interpreted as OFF.

## Synthetic-only contract

Synthetic tests use symbolic internal categories:

- `INVASIVE`
- `NON_INVASIVE`
- `HFNC`
- `ORDINARY_OXYGEN`

Only `INVASIVE` produces invasive ON. The synthetic interval convention is
`[start, end)` and is non-authorizing. It does not establish the real concept's
boundary convention.

Synthetic metadata is deliberately marked `NOT_MIMIC` / `NOT_MIMIC_CODE` and
cannot pass the frozen-real validator.

## Shared dictionary lineage

The Phase-2 draft section is stored in `configs/event_dict_v1.yaml`.

Phase-2 exact-file SHA-256 at implementation time:

`486cb500fa6d9b39805f122c983c28cc41efbef1fb7432bc089deee481ae51ca`

This is a historical draft-development hash, not a frozen scientific concept
hash. Phase 3 subsequently extended the shared dictionary, so this hash is no
longer the current exact-file identifier. Each phase review records its own
draft lineage without presenting any of those hashes as a scientific freeze.

## Freeze requirements

Before the ventilation section can become `FROZEN_REAL`, the validator requires
verified concept metadata, an exact Git hash, explicit raw-to-internal mapping,
NIV/HFNC/ordinary-oxygen exclusion semantics, output and boundary semantics,
duplicate/overlap policy, SOFA alignment, and Sanskruti approval evidence.

Freeze must occur before any real support prevalence inspection.
