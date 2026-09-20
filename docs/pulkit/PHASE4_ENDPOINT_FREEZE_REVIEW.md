# Phase 4 endpoint-freeze review

## Decision

`ENDPOINT FREEZE = BLOCKED` for real scientific execution and real prevalence access. This is an evidence-backed governance outcome, not an implementation failure. No real label or prevalence data were loaded.

The blocked audit is `artifacts/labels/support_endpoint_freeze_v1.json` (SHA-256 `f2b90e23806f98d068b284eb7348b57193d18cc93cdd00b9429cb005bfdbeba8`). It binds the current event dictionary, composite implementation, timestamp contract, proposed signoff record, and repository-commit status.

## Real blockers

- The event dictionary is not globally frozen and does not authorize real execution.
- Phase 1 vasopressor definitions remain synthetic/draft rather than verified real MIMIC-IV mappings.
- Phase 2 ventilation definitions remain draft rather than frozen against a pinned reviewed MIMIC-Code concept and aligned SOFA semantics.
- Phase 3 composite tie, unknown-state, audit-vocabulary, and reviewer states are not frozen.
- Sanskruti, Vedant, and Pulkit-after-team-agreement approvals are genuinely pending in `configs/support_endpoint_signoff_v1.json`; no approvals were fabricated.
- Reviewed state-machine test evidence and the pre-freeze noninspection attestation are pending.
- This workspace is not a Git repository, so an immutable code commit cannot be bound.
- The real reporting population and headline denominator are unresolved.

Until all blockers are resolved through reviewed, versioned change control, `create_real_endpoint_freeze` fails closed. The prevalence guard validates the exact freeze bytes and current dependency hashes before calling a supplied label loader. A synthetic freeze cannot authorize real data, and protected test access additionally requires a valid matching G3 marker plus explicit endpoint-freeze authorization.

## Synthetic governance rehearsal

The synthetic chain is non-scientific, non-authorizing, and unmistakably marked as such:

- Freeze: `artifacts/labels/synthetic/support_endpoint_freeze_synthetic_v1.json` (`8124d4d74246d2b24643e5c85f1f763fca84d3832c63176b55565f4ba31464e6`)
- Labels: `artifacts/labels/synthetic/support_labels_phase4_synthetic_v1.json` (`e3cd7c28f08f5c762348cc17389fcdfbf2436111b45cefbfd27b5fabe0868cd0`)
- Raw-count report: `artifacts/labels/synthetic/support_prevalence_synthetic_v1.json` (`88bc645e88a541f5dce9775ff595ad330bbebbf1f529992ebccf08cca6474e58`)

The synthetic raw counts are five canonical rows across four stays: three eligible rows, two positive rows, one negative row, one censored row, and one not-at-risk row. There are two eligible stays and two positive stays. No percentage is reported because the headline denominator is intentionally unresolved. Censored and not-at-risk rows remain separate from eligible negatives.

## Version and lineage bindings

- Event dictionary SHA-256: `717f6183ad8014602b3cc32eeaa3a810eab8a26a19937bda8e214200bb2f9620`
- Composite implementation SHA-256: `216f066ed837a634b66cb5aa0e7741c5dcfb914243e1b0eb03c8198ed67c3058`
- Timestamp contract SHA-256: `de3a424fcd60affda178afbde49a0c83fd51610f7682cf73306d124454b18a9b`
- Signoff schema SHA-256: `e360928acd5634ffadccf56424e1a3b982e73d84aa2ea2ac848a15bdb926a0d2`
- Prevalence governance config SHA-256: `aef489fcaf8b311a6063b3c131440547834c446d5dcce232a130778c5eb7fc48`
- Code commit: unavailable because the workspace is not a Git repository.

All five Phase-4 records are registered in `experiments/artifacts.csv`, with freeze→label→report parent relationships for the synthetic chain. Artifact paths and identifiers are immutable.

## Scope boundary

No class weights, model training, tuning, thresholds, calibration, acceptance ranges, or Phase-5 work were performed. Prevalence is descriptive only and cannot mutate the endpoint definition. Real Phase-5 handoff remains blocked; only the governance interface and synthetic rehearsal are ready for review.
