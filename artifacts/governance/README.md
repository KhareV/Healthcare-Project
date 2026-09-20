# Test-Set Governance Artifacts

`g3_freeze.json` is the only active G3 authorization artifact. It must be absent until every real prerequisite passes and all three members sign off. Its `g3_freeze_marker_v1` schema remains an unlocked engineering parameter pending that review.

`g3_audit_report.json` is a diagnostic snapshot. It explicitly carries `authorization: false`; its existence, even with passing items, never grants test access.

`test_access_state.json` is created only by a successful freeze and records authorization consumption and reset history. Invalidated markers are retained under `history/`.

Authorization requires marker content validation, exact dependency hashes, a fresh prerequisite audit, correct real scope, and the `AUTHORIZED_NOT_RUN` state. Filename presence is never sufficient. See `docs/TEST_SET_GOVERNANCE.md` for the complete policy.
