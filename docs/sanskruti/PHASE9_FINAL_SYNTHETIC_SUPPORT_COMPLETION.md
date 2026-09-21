# Phase 9 Final Synthetic Support Completion

Status: `COMPLETE — FINAL PRE-SPLIT SYNTHETIC PACKAGE`  
Decision authority: `USER_DELEGATED_AI_PROJECT_DECISION`  
Clinical validation: none; this is a project-designed synthetic benchmark.

Phase 9 froze the five-coordinate hourly latent process, support initiation/cessation/rate/effect process, synthetic event dictionary, interval policies, seed `20260921`, and the deterministic accepted-cohort size of 2,000. The support contract is atomically bound to the latent-process SHA-256. Final generation integrates support inside the latent transition loop with a two-hour feedback lag.

The full replay produced 2,000 subjects, 2,000 retained stays, 670,303 raw observations, 63,663 one-hour support state/rate segments, and 12,222 legal prediction rows. A second full raw replay reproduced the exact semantic hashes of subjects, episodes, raw observations, and support intervals.

The Phase 7 successor schema has 21 temporal channels: the original 15 plus current qualifying vasopressor state, current invasive-ventilation state, and four agent-specific rates. Active-bin support masks are known, TSLO is zero, padding remains structural, and Pulkit's existing state engines are reused. Future starts and future cessation bounds are not exposed as model inputs.

The final pre-split QA contains 7,827 support-eligible rows: 3,459 positives and 4,368 negatives. It contains 7,030 eligible 24-hour recovery targets, 2,589 eligible 48-hour recovery targets, and 12,222 ICU-time targets; every continuous target family has nonzero variance. Production SOFA is frozen under the delegated synthetic authority without changing its numerical thresholds.

No train/validation/test split, learned preprocessor, model, hyperparameter search, calibration, threshold selection, or final-test evaluation was performed. The governed suite passed 1,193 tests with two optional-dependency skips. Evidence audit passed with 17 indexed outputs and zero orphans; registry integrity reported no missing artifacts, hash mismatches, broken parents, incompatible dependencies, scientific orphan artifacts, final-test access, or G3 marker.

Bulk production JSONL artifacts are deterministic local build outputs and are excluded from Git because files exceed GitHub's per-file limit. Their builders, frozen configs, hashes, audit entry point, and small governed fixture/evidence artifacts are version controlled.
