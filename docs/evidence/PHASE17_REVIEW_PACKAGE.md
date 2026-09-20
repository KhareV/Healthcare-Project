# Phase 17 Review Package

## Outcome

- Governance framework: implemented and adversarially tested.
- Real scientific G3 freeze: `BLOCKED`.
- Active real marker: absent by design.
- Real test access: not performed.
- Phase 18+ work: not implemented.

## Review surfaces

- `src/vedant_infra/g3.py`: audit, marker, integrity bindings, access state, guard, and reset.
- `src/experiments/g3_cli.py`: non-authorizing audit and fail-closed real freeze commands.
- `src/experiments/final_test_entrypoint.py`: sole guarded Phase-20 integration seam.
- `docs/TEST_SET_GOVERNANCE.md`: operating and reset policy.
- `artifacts/governance/g3_audit_report.json`: current machine-readable blocked audit; never authorization.

## Required reviewer decisions

1. All members must approve `g3_freeze_marker_v1`, currently **UNLOCKED ENGINEERING PARAMETER — G3 MARKER SCHEMA**.
2. All members must approve the persistent access-state vocabulary, currently **UNLOCKED ENGINEERING PARAMETER — FINAL TEST ACCESS STATE REPRESENTATION**.
3. A non-permissive failure/retry policy is required before Phase 20: **BLOCKED — FINAL TEST RETRY POLICY REQUIRED**.
4. Pulkit must deliver and review TreeSHAP and Integrated Gradients adapters before official G3: **BLOCKED — PULKIT EXPLANATION ADAPTER G3 DEPENDENCY**.

## Mandatory reviews

- Vedant: verify exact-30 accounting, selected-model and post-processing bindings, registry lineage, guard ordering, one-run consumption, and reset behavior.
- Sanskruti: verify split hash/metadata, feature and label versions, preprocessing training-only provenance, and selected XGBoost/data compatibility.
- Pulkit: verify support bindings, explanation implementations and routing, and downstream manifest compatibility.

Approval must be recorded in a real-scope `g3_review_signoffs.json`; prose acknowledgement alone is insufficient.

## Adversarial evidence

Synthetic-only tests demonstrate that missing/malformed/inactive markers fail before the loader, all-task and explanation routing rules are enforced, a complete mixed-family state can freeze, test metrics are rejected, mutations to manifest/model/calibrator/threshold/split deny access, the authorization is consumed once, reset archives history, post-freeze changes require reset, and synthetic authorization cannot unlock a real path.

## Handoff condition

Phase 18 may integrate real canonical artifacts into these existing contracts. It must not create the active marker, weaken the audit, regenerate the split, or open test data. Real activation remains contingent on Phases 18–19 evidence and all-member approval.
