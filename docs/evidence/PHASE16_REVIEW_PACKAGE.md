# Phase 16 Registry and Lineage — Pulkit Review Package

## Scope

This package covers Vedant-owned registry and lineage governance only. It does
not implement serving, open final test data, or create the G3 marker.

## Review interfaces

- Authoritative runs: `experiments/registry.csv`
- Artifact lineage: `experiments/artifacts.csv`
- Schema/write validation: `src/vedant_infra/registry.py`
- Artifact graph/tracing: `src/experiments/lineage.py`
- Audit: `src/experiments/audit.py`
- CLI: `src/experiments/registry_cli.py`
- Policy: `docs/EXPERIMENT_REGISTRY.md`

## What Pulkit should verify

1. A finalized run cannot be rewritten under the same ID.
2. Candidate IDs remain distinct from retry attempt run IDs; failed attempts
   remain present.
3. Checkpoint, prediction, result, selected-manifest, calibrator, threshold,
   ablation, error-analysis, evidence, and future serving-build artifact types
   can be represented without changing the run registry.
4. Prediction/checkpoint, calibrator/model, threshold/calibrator, and
   task/family/split/feature/label mismatches fail closed.
5. Result and reverse traces expose config, split, feature, label, and commit
   provenance without loading model internals.
6. Hash drift is reported and never silently adopted.
7. `artifact_descendants` identifies downstream artifacts affected by change.
8. Synthetic runs cannot enter final-scientific queries.
9. Explicit restricted-data fields are rejected.

## Historical audit

Seven synthetic smoke runs remain in the single registry. Sixteen exact files
were indexed. Seven checkpoints and two preprocessors are verified. Seven
historical metric rows are marked `legacy_incomplete` because prediction parents do not exist. The
audit also lists unregistered historical smoke/search/supporting files as
orphans; none are classified as final-scientific artifacts.

The Phase 15 synthetic error-analysis table remains a synthetic historical
orphan because its generating execution was not registered. It must not be
retroactively assigned a fabricated run.

## Phase 17 handoff

Phase 17 can query validated runs, exact artifact hashes, selected/calibration/
threshold compatibility, final-scientific eligibility, broken parents,
untraceable results, and whether any final artifact is orphaned. Phase 16 does
not decide or write the G3 freeze state.
