# Phase 19 Full Validation Governance

## Current scientific status

The orchestration framework is implemented, but the real validation program has not started. The registered Phase-18 acceptance report is `BLOCKED`; therefore every real preprocessing, baseline, search, model-selection, calibration, threshold, ablation, and error-analysis action remains prohibited.

The Phase-19 preflight is intentionally broader than the first blocker. It reports all presently known frozen-input gaps while guaranteeing `scientific_commands_executed: 0` and `test_accessed: false`.

## Command surface

```bash
PYTHONPATH=src python3 -m experiments.validation_suite_cli plan
PYTHONPATH=src python3 -m experiments.validation_suite_cli audit
PYTHONPATH=src python3 -m experiments.validation_suite_cli run
PYTHONPATH=src python3 -m experiments.validation_suite_cli finalize
```

All four commands run the same fail-closed preflight. `run` and `finalize` cannot reach a scientific executor unless every prerequisite passes. There is no force, skip-acceptance, test, candidate-31, or unsafe option.

## Preflight requirements

- Exact registered Phase-18 report path and SHA-256, internally valid and `ACCEPTED`.
- Final-test state `NEVER_OPENED`, no active G3 marker, and train/validation-only partition policy.
- Frozen GRU and XGBoost search spaces.
- Six real scientific candidate manifests: three tasks times two families.
- Exactly 30 unique candidate identities and configuration hashes in each manifest.
- Frozen search-space and candidate-list hashes.
- Approved real project seed policy for selected-GRU reruns.
- Approved real validation-F1 threshold policy.
- Sanskruti's real XGBoost entrypoint.
- Clean Phase-16 critical registry lineage.

Synthetic dry-run manifests do not satisfy real preflight. Scientific candidates are loaded from predeclared manifests; Phase 19 never generates or replaces them after results are visible.

## Search accounting and resume

Candidate identity is distinct from execution attempt identity. Retries remain attached to the same candidate and never increase the exact-30 scientific count. Candidate 31 or an unknown candidate ID is rejected.

A complete run is skipped only after its candidate config hash, checkpoint hash, and validation-prediction hash validate. Failed or incomplete work may only return `RETRY_SAME_CANDIDATE`; incompatible artifacts cannot be reused.

## Stage ordering

The frozen orchestration order is preprocessing, naive baselines, searches, within-family selection, selected-GRU reruns, LSTM sensitivity, cross-family selection, support calibration, support threshold, selected manifest, comparison table, ablations, error analysis, and validation freeze.

Calibration cannot precede support selection; thresholding cannot precede calibration. Exactly one selected-config GRU rerun and one non-search LSTM sensitivity run are required per task. Naive and LSTM rows are never serving candidates.

After `VALIDATION_FROZEN`, any extra candidate or family change requires a scientific reset.

## Selection and post-processing

- Recovery: lowest validation stay-balanced MAE24; MAE48 remains secondary.
- ICU time: lowest validation stay-balanced weighted median absolute error in hours.
- Support: highest validation stay-balanced AUPRC on raw, uncalibrated probabilities.
- Serving candidates: XGBoost and GRU only.
- Routing metadata: XGBoost to TreeSHAP; GRU to Integrated Gradients.
- Calibration: selected support model only, validation partition only.
- Threshold: calibrated validation probabilities and validation F1 only.

Model, calibrator, and threshold hashes must form one compatible chain. Every comparison-table row must resolve to both a registered run and registered checkpoint with the same hash.

## Current blockers

- `BLOCKED — REQUIRED VALIDATION DEPENDENCY MISSING`: Phase-18 report is not accepted.
- Six real scientific candidate manifests are missing.
- `BLOCKED — PROJECT SEED POLICY REQUIRED`.
- `BLOCKED — REAL THRESHOLD POLICY REQUIRED`.
- `BLOCKED — REQUIRED VALIDATION DEPENDENCY MISSING: Sanskruti XGBoost entrypoint`.
- Real Pulkit support labels remain absent through the Phase-18 blocker.
- Official G3 remains blocked by validation prerequisites, Pulkit explanation adapters, and all-member signoff.

No performance value, winner, selected model, calibrator, or threshold is reported while these blockers remain.
