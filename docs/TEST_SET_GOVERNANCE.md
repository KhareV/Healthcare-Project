# Test-Set Governance and G3 Freeze

## Scope

Phase 17 installs the hard boundary between development/validation and the one final test evaluation. It does not perform real-data integration, validation experiments, or final-test evaluation. The implementation may be complete while the scientific G3 freeze remains blocked.

The active real authorization path is `artifacts/governance/g3_freeze.json`. Absence is the safe default. `g3_audit_report.json` is a diagnostic report with `authorization: false`; it can never authorize access.

## Commands

Run the current-state audit without activating G3:

```bash
PYTHONPATH=src python3 -m experiments.g3_cli audit
```

Attempt a real freeze only after every upstream prerequisite and all reviews exist:

```bash
PYTHONPATH=src python3 -m experiments.g3_cli freeze
```

There is no force, skip-validation, unsafe, or environment-variable bypass. The freeze command never reads test labels, predictions, prevalence, or metrics.

## Prerequisite audit

The fail-closed audit validates:

- immutable split artifact and, for real scope, its metadata;
- all three selected task entries and mixed-family-safe per-task routing;
- exactly 30 unique completed validation candidates for XGBoost and GRU per task, with frozen search hashes;
- exactly one non-search LSTM sensitivity run per task, parented to the selected GRU;
- selected checkpoint and training-fit preprocessor existence, hashes, versions, and registry producers;
- support isotonic calibrator binding to the selected support model;
- support validation-F1 threshold binding to that calibrator;
- Phase-16 critical registry/artifact lineage;
- TreeSHAP and Integrated Gradients adapter review required by the official G3 gate;
- all-member signoff; and
- structural test nonuse through scope, partition, run-type, lineage, and persistent access-state checks.

Test performance fields are rejected recursively from the selected manifest. G3 does not require or inspect test performance.

## Active marker schema

`g3_freeze_marker_v1` is an **UNLOCKED ENGINEERING PARAMETER — G3 MARKER SCHEMA** pending all-member approval. An active marker contains the gate and active status, real or synthetic-test scope, creation time, split hash, exact selected-manifest path/hash, per-task run/family/checkpoint/preprocessor/version/explanation bindings, support calibrator and threshold bindings, an exact dependency list, reviewers, audit version, a false pre-freeze-access assertion, and a canonical marker content hash.

The marker is validated by content and current dependency hashes, not by filename presence. The prerequisite audit is rerun during every access authorization. A synthetic-test marker cannot be written to the real path and cannot authorize the real scope.

## Access order and state

The only Phase-17 final-test wrapper is `experiments.final_test_entrypoint.run_guarded_final_test`. Its order is:

1. validate marker schema, status, scope, and canonical hash;
2. validate every bound dependency and rerun the G3 audit;
3. require `AUTHORIZED_NOT_RUN`;
4. persist consumption as `FINAL_RUN_COMPLETED`;
5. only then call the supplied test loader.

The persistent state vocabulary is **UNLOCKED ENGINEERING PARAMETER — FINAL TEST ACCESS STATE REPRESENTATION**:

- `NEVER_OPENED`: no active authorization and no final access;
- `AUTHORIZED_NOT_RUN`: valid freeze created, designated run not consumed;
- `FINAL_RUN_COMPLETED`: authorization consumed; a second run is denied;
- `INVALIDATED_BY_RESET`: prior result invalidated and history retained; a new formal freeze is required.

Consumption is recorded before invoking the loader. This is deliberately conservative: even a loader failure does not silently permit a retry. The project therefore remains **BLOCKED — FINAL TEST RETRY POLICY REQUIRED** before Phase 20 if a non-observational technical failure is to be treated differently.

Test access includes loading test labels, loading label-linked predictions, computing or inspecting aggregate/slice metrics or prevalence, and loading test features for frozen final inference. Partial inspection still counts.

## Mutation, post-test changes, and reset

Any bound split, selected manifest, checkpoint, preprocessor, calibrator, threshold, registry, or artifact-index mutation invalidates authorization. The marker is never silently regenerated.

Model selection, calibration, threshold, feature, or label changes are denied while a freeze is active or after the final run is consumed. After a genuine result-affecting defect:

1. record a reason and invoke formal invalidation;
2. archive the prior marker under `artifacts/governance/history/`;
3. remove the active marker and append an immutable-style history event;
4. mark the state `INVALIDATED_BY_RESET`;
5. version-bump affected contracts/artifacts and identify stale descendants;
6. return to development/validation and establish a new reviewed freeze.

The original access and invalidation history is retained. Premature access or tuning from test results invalidates the untouched-test claim and must use the same formal reset process.

## Current repository status

The Phase-17 framework is implemented. The current real audit is `BLOCKED`: real split metadata, completed scientific searches, selected models, LSTM sensitivities, support calibration/threshold, and explanation-adapter review evidence are not present. The official gate explicitly requires Pulkit-owned explanation adapters, so this also remains **BLOCKED — PULKIT EXPLANATION ADAPTER G3 DEPENDENCY**. The active real marker must remain absent.

## All-member review checklist

- Vedant: test isolation, search/selection/calibration freeze, registry lineage, marker/guard/state/reset logic.
- Sanskruti: split, feature, label, SOFA/data compatibility, preprocessing provenance, and XGBoost artifact bindings.
- Pulkit: support endpoint compatibility, both explanation adapters, routing metadata, and serving-manifest compatibility.

All three must approve the real signoff artifact before activation. Review of this engineering mechanism is not evidence that the later scientific prerequisites are complete.
