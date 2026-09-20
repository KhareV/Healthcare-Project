# Phase 18 Real-Data Acceptance

## Purpose and boundary

Phase 18 validates an upstream, already-built canonical dataset against frozen ML contracts. It does not extract MIMIC data, build features, derive SOFA, implement organ-support state logic, fit scientific models, compare performance, select models, calibrate probabilities, or open protected test outcomes.

Only a report whose `overall_status` is exactly `ACCEPTED`, whose file and content hashes validate, and whose isolation flags remain false can authorize Phase 19 through `require_accepted_real_data`.

## Input handoff

The input is `artifacts/data/real_data_inputs_v1.json`. Every entry supplies a repository-relative path, owner, artifact/schema versions, creation commit when available, consumer contract, and SHA-256. Required inputs include:

- Sanskruti canonical development dataset;
- processed-schema sidecar containing actual field mappings and dtype declarations;
- ordered feature schema and feature provenance dictionary;
- outcome-free structural index for train, validation, and test membership/timestamps;
- frozen split plus metadata;
- SOFA provenance and pinned MIMIC-Code commit;
- label specification;
- tensor and timestamp contracts; and
- optionally Pulkit's independently produced support handoff.

Protected test labels may be inventoried with `PROTECTED_NOT_OPENED_PHASE18`. Their declared reference/hash is recorded, but the path is never opened or hash-recomputed in Phase 18.

The current engineering adapter accepts the versioned `canonical_json_v1` exchange representation used by the acceptance tests. The production storage representation and dependency must be supplied and frozen by the upstream sidecar. A different Parquet/Arrow representation requires an explicitly reviewed versioned decoder; it is never silently coerced.

## Acceptance checks

The audit checks exact artifact hashes and versions, sidecar-defined field mapping, feature order, declared and runtime-compatible types, `[N,8,F]` shape, prefix padding, genuine-observation masks, frozen TSLO sentinel/propagation, ordered statics, frozen valid ranges, and six SOFA component-observed flags.

It regenerates every legal cutoff from `intime` and `outtime` and requires exact parity, including the first 24-hour cutoff, six-hour spacing, at most 12 cutoffs, and legal equality at `outtime - 6h`. Every canonical key is `stay_id + prediction_time`; duplicates are rejected rather than removed.

The structural index is checked against `split_v1` and its metadata. It contains no outcome fields. The development canonical artifact must contain exactly the non-test structural keys and may not contain test rows.

Recovery deltas are checked against one baseline SOFA, ICU time against `(outtime-t)` and `log1p`, and eligibility against available follow-up where provable. SOFA provenance must declare one function for all cutoffs and a pinned MIMIC-Code commit.

If Pulkit's handoff exists, the adapter only validates and joins supplied evidence. It checks already-ON behavior, positive-before-censor, full-follow-up negatives, early-exit censoring, and one-to-one canonical keys. It never reconstructs medication or ventilation states.

Preprocessing must be train-only or explicitly not yet fitted with a Phase-19-ready interface. Missingness summaries cover train/validation inputs only. Test reporting is limited to structural subject/stay/example counts; test labels, eligibility, prevalence, feature distributions, predictions, and performance remain uninspected.

## Commands

```bash
PYTHONPATH=src python3 -m experiments.real_data_acceptance_cli
```

Exit code `0` means `ACCEPTED`; exit code `2` means `BLOCKED`. Both write the machine-readable report at `artifacts/data_acceptance/real_data_acceptance_v1.json`. A blocked report is evidence, not authorization.

Each invocation also registers an immutable, content-addressed copy under `artifacts/data_acceptance/history/` in the Phase-16 artifact index. A later changed decision creates a new snapshot rather than rewriting registered history.

## Hash semantics

The report carries:

- exact upstream artifact hashes for byte-level lineage;
- canonical semantic hashes for deterministic row-order-independent comparison;
- an acceptance-decision hash that excludes raw inventory byte hashes; and
- a report-content hash over the full report content.

Phase 19 must bind the exact report file SHA-256 in addition to validating its internal content hash and accepted status.

## Current status

The current real report is `BLOCKED`. No real canonical dataset, processed/feature schemas, feature dictionary, split artifact, SOFA provenance, label specification, or Pulkit support handoff is present. In particular:

- `BLOCKED — INFORMATION REQUIRED`
- `BLOCKED — PULKIT REAL ORGAN-SUPPORT LABEL ARTIFACT REQUIRED`
- `UNLOCKED ENGINEERING PARAMETER — PRODUCTION CANONICAL STORAGE DECODER` until the actual upstream Parquet/Arrow/agreed representation and dependency are supplied.

No synthetic fixture can authorize Phase 19, and no protected test outcome was opened.
