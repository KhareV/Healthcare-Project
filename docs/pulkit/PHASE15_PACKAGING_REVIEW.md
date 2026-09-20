# Pulkit Phase 15 Packaging and Demo-Fixture Review

## 1. PULKIT PHASE 15 SUMMARY

The packaging documentation, privacy guardrails, observed-runtime audit, official synthetic input fixture, validator, and replay acceptance tests are implemented. The final environment lock and clean installed live demo remain blocked because no package manager/version policy is approved and the observed runtime lacks required real-stack dependencies.

## 2. REPOSITORY STATE BEFORE CHANGES

The pre-change suite passed 887 tests with two skips. No root README, runbook, package declaration, dependency lock, `.gitignore`, `.env.example`, or `data/demo` fixture existed. The directory is not a Git checkout.

## 3. PACKAGE/ENVIRONMENT MANAGER AUDIT

No `pyproject.toml`, Poetry/uv lock, requirements input/lock, Conda definition, setup metadata, Dockerfile, or task-runner convention exists. Phase-21 explicitly records the exact environment as absent.

## 4. FINAL ENVIRONMENT DECISION STATUS

`BLOCKED — PACKAGE/ENVIRONMENT MANAGER DECISION REQUIRED`. A raw pip freeze was not presented as a final lock and no manager was silently selected.

## 5. FILES CREATED

Created the root README, RUNBOOK, `.gitignore`, non-final observed environment snapshot, `data/demo/demo_patient_v1.json` and metadata, `src/demo` validator, Phase-15 tests, and this review.

## 6. FILES MODIFIED

The Phase-14 integration test helper was minimally parameterized so the official fixture's validated current-SOFA provider and dashboard catalog can traverse the same pipeline/API/dashboard harness. No scientific production contract or artifact was changed.

## 7. ENVIRONMENT LOCK PATH

None. `configs/reproducibility_v1.json` remains truthfully unbound with `environment_lock_ref: null`.

## 8. ENVIRONMENT LOCK SHA-256

None. No candidate/development snapshot was registered or labeled as a final lock.

## 9. PYTHON VERSION

Observed runtime: CPython 3.9.6. This is recorded in `observed_environment_phase15.json` as non-final and does not establish an approved interpreter policy.

## 10. PACKAGE LOCK / HASH STATUS

Blocked. Exact observed versions are recorded, but package-manager integrity hashes and a reproducible install graph cannot be generated until the manager and platform policy are approved.

## 11. CPU/GPU/CUDA STATUS

Observed development runtime is macOS arm64, PyTorch 2.8.0 CPU-only, `torch.version.cuda = null`, and CUDA unavailable. This is an observation, not a final training device policy.

## 12. README STATUS

Complete as a truthful repository entry point: scope, status, layout, tested development commands, fail-closed app entrypoints, privacy boundary, and documentation links are included.

## 13. RUNBOOK STATUS

Complete for the currently supportable operations. It separates synthetic input validation, fail-closed local startup, and blocked authorized real mode; documents tests, audits, shutdown, troubleshooting, privacy, and change control.

## 14. DEMO FIXTURE PATH

`data/demo/demo_patient_v1.json` with `data/demo/demo_patient_v1.metadata.json`.

## 15. DEMO FIXTURE PROVENANCE

The fixture is manually constructed synthetic point-event data with unmistakably synthetic identifiers. It is not derived from a MIMIC patient and claims no legal/privacy certification.

## 16. DEMO FIXTURE SHA-256

`625289d3eaff3482cc94d4190bed61adb8f81c1a8eb24f2e148397b6a85d1a43`. The metadata sidecar SHA-256 at validation was `eb448d75b50f633ded0466037728dfa5f3895308ab02e581bffa980805ded3da`.

## 17. DEMO SCHEMA VALIDATION

The validator reuses existing `TimelineContract`, `StoredStayTimeline`, `InMemorySyntheticHistoryProvider`, canonical truncator, timestamp generator, current-SOFA contract, and hash utility. Phase-14 integration proves the fixture is accepted by the same pipeline boundary.

## 18. LEGAL REPLAY CUTOFFS

The frozen approved cutoffs are 2030-01-02 00:00Z, 06:00Z, and 12:00Z: admission +24 hours followed by six-hour cadence, all within the canonical cap.

## 19. T1/T2/T3 REPLAY RESULT

The three cutoffs expose respectively two, three, and four events. A fifth event after t3 remains hidden. Each selection calls the API/pipeline path with the exact timestamp.

## 20. PREDICTION-LOOKUP NONUSE

The fixture contains inputs, current-SOFA state, and provenance only. It contains no stored t1/t2/t3 predictions or prediction lookup table.

## 21. PRIVACY / RESTRICTED-DATA AUDIT

The fixture contains no raw MIMIC identifiers, item IDs, hospital-admission IDs, caregiver IDs, names, binary exports, or credentials. This guardrail is not presented as legal certification.

## 22. SECRET / ABSOLUTE-PATH AUDIT

Packaging, demo, README, runbook, and observed-environment files pass deterministic scans for personal absolute paths and obvious credential assignments.

## 23. CLEAN LOCAL ENVIRONMENT INSTALL RESULT

`BLOCKED — NO APPROVED LOCK`. A new environment was not mislabeled clean by inheriting global packages. Captum and SHAP are absent from the observed runtime, confirming that the current environment cannot be frozen as the complete declared real stack.

## 24. PHASE-14 INTEGRATION UNDER LOCK

Not applicable because no approved lock exists. Phase-14 integration is rerun in the observed development runtime and reported separately, without claiming locked-environment reproduction.

## 25. API CLEAN-ENVIRONMENT SMOKE

Blocked with the clean install. The documented default API startup is tested only as a fail-closed development entrypoint; `/health` is available while metadata/prediction return unavailable until an approved serving composition exists.

## 26. DASHBOARD CLEAN-ENVIRONMENT SMOKE

Blocked with the clean install. The default dashboard starts fail closed and preserves the retrospective warning; a packaged live synthetic/real replay server is not fabricated.

## 27. REGISTRY AUDIT

Run read-only. Packaging introduces no registered scientific artifacts, hash drift, fabricated producers, or final-scientific synthetic promotion. The official fixture is an input file, not a model/result artifact.

## 28. REPRODUCIBILITY AUDIT

The Phase-21 report builder is invoked read-only. Environment remains blocked; artifact integrity, path portability, restricted-data checks, and synthetic CI remain separately reported without overwriting unrelated blockers.

## 29. FINAL-TEST NONACCESS STATUS

`NEVER_OPENED`. No final-test entrypoint or loader is invoked and audit state remains false.

## 30. G3 NON-AUTHORIZING AUDIT STATUS

Real G3 remains blocked. The audit is invoked without freeze/write authority and no marker is created.

## 31. TESTS ADDED

Added fixture schema/hash/replay/pipeline tests; environment-blocker and observed-version tests; README/runbook/reference tests; `.gitignore` tests; and privacy, credential, path, and raw-field scans.

## 32. EXACT COMMANDS EXECUTED

Executed `PYTHONPATH=src:tests:. python3 -m demo.fixture validate --root .`; the focused Phase-15 and combined Phase-14/15 pytest selections; `PYTHONPATH=src:tests:. python3 -m experiments.registry_cli audit --root .`; direct read-only `build_report(...)` and `audit_g3(..., scope='real')`; local uvicorn API/dashboard startup with curl smoke checks; portability/privacy searches; `PYTHONPATH=src:tests:. python3 -m compileall -q api dashboard src tests`; and `PYTHONPATH=src:tests:. python3 -m pytest -q`. No final-test, G3-freeze, release-evidence, package-install, or lock-generation command was executed.

## 33. FULL TEST RESULTS

Phase-15 focused tests passed 13/13; the combined Phase-14/15 selection passed 36/36; portability regression tests passed 16/16; compilation passed; and the complete repository suite passed 900 tests with two skips. API and dashboard uvicorn entrypoints started successfully and returned the documented fail-closed states. Performance metrics were not acceptance criteria.

## 34. REAL SCIENTIFIC ARTIFACT STATUS

Blocked: real selected manifest, task models, train-fit preprocessors, support calibration chain, verified current-SOFA handoff, approved timeline, and complete real lineage remain unavailable.

## 35. BLOCKED / UNLOCKED PARAMETERS

Blocked decisions: package manager, final Python/packages, platform/device policy, final environment approval, and live serving bundle. The demo uses the already established JSON synthetic timeline convention; local ports 8000/8501 are presentation-only development defaults.

## 36. VEDANT ENVIRONMENT REVIEW NOTES

Review is required for manager choice, exact versions, hashes, platform/device policy, future-training commitment, compatibility of PyTorch/XGBoost/SHAP/Captum/evaluation/API packages, Phase-14 execution from lock, and synthetic classification.

## 37. SANSKRUTI DEMO/PRIVACY REVIEW NOTES

Review is required for timeline schema, event-time semantics, units, feature identity, current SOFA, legal cutoffs, future-row hiding, absence of raw MIMIC mappings, and synthetic provenance.

## 38. DEFINITION-OF-DONE CHECKLIST

Packaging framework tasks that do not require the unresolved manager are complete: audit, docs, fixture/version/hash, validator, pipeline replay tests, privacy scans, ignore rules, Phase-14 rerun, registry/reproducibility/G3 read-only checks, and no Phase-16 evidence. Final lock and clean install boxes remain unchecked.

## 39. FINAL ENVIRONMENT FREEZE STATUS

`BLOCKED — PACKAGE/ENVIRONMENT MANAGER REQUIRED` and `BLOCKED — FINAL ENVIRONMENT VERSION DECISION REQUIRED`. Git commit provenance is also unavailable because this directory is not a Git checkout.

## 40. OFFICIAL DEMO FIXTURE STATUS

The official Phase-15 synthetic **input fixture** is frozen at version `demo_patient_v1` and its recorded hash. Schema, timestamp, privacy, mutation, three-cutoff, pipeline, API, and dashboard-contract tests pass. A clean-installed live demo remains blocked by environment and serving-composition prerequisites.

## 41. READINESS FOR PULKIT PHASE 16

Not fully ready for final evidence: Phase 16 must wait for an approved lock, clean installation, complete local live demo, reviews, and real or explicitly approved serving composition. The documentation, official input fixture, validator, and blocker package are ready for that handoff.
