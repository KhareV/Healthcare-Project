# Sanskruti Rebuild Phase 1 Completion Report

## 1. SANSKRUTI REBUILD PHASE 1 SUMMARY

Phase 1 is implemented as a governance-only migration. It establishes one human-readable and one machine-readable `PROJECT_SCOPE_V2`, a 53-row requirement-delta matrix, a Vedant/Pulkit contract-impact matrix, integrity hashes, semantic tests, future-agent discovery pointers, and human review checklists. It does not generate data, build features or labels, implement SOFA, train or select models, calibrate a classifier, access final-test data, or change scientific runtime behavior.

The authorized changes are limited to: final MIMIC-IV data source → final synthetic project data; general ICU population → adult cardiac/heart-disease population; MIMIC-era split → unresolved synthetic subject-split policy; MIMIC provenance → synthetic generator/schema/config/seed/dataset provenance; and MIMIC event mappings → future versioned synthetic mappings.

## 2. REPOSITORY STATE BEFORE CHANGES

The integrated project root `/Users/Home/Downloads/HealthCareProject` is not itself a Git working tree. The nested historical Sanskruti repository `Heart-Failure-Recovery-and-Deterioration-Trajectory-Prediction` was on branch `main`, commit `dc95164b808becd6e822bccd28c0135303753e40`, with a clean worktree. It was inspected only; it was neither executed as production nor modified, merged, imported, or registered.

The root already contained mature Vedant/Pulkit code, configs, phase reviews, evidence, serving components, and 906 passing integrated tests. It had no authoritative `project_scope_v2` artifact and still contained active v1/MIMIC wording alongside historical MIMIC implementation dependencies.

## 3. BASELINE TEST RESULT

Before Phase-1 edits:

```text
906 passed, 2 skipped, 0 failed, 0 errors in 10.70s
```

The skips were the already-declared optional Captum and SHAP/XGBoost integration paths.

## 4. AUTHORITATIVE SOURCES INSPECTED

- Final Implementation Master Plan DOCX, SHA-256 `a819b6144061780fac070537b2a29a50d8b3ea24d7a82d4cb7235985d0ab5588`.
- Final Work Division DOCX, SHA-256 `3d07c6283896b468f9cb013a669412dd8e1e96ff8aabe350c0f8cb7601b636a0`.
- Explicit later user/team decision in the Phase-1 request.
- `docs/CODEX_PROJECT_CONTEXT_V1.md` and active repository rules.
- `docs/SYNTHETIC_HEART_DISEASE_INTEGRATION_AUDIT.md`, SHA-256 `9296e63f192b23f52a4f640b30df0abe51dd4e7c5964d007f0b1def379a972fe`.
- Current timestamp, tensor, prediction, split, event, label, evaluation, registry, final-test, serving, replay, packaging, and evidence contracts.
- Historical Sanskruti candidate repository, as reference only.

## 5. LATER SCOPE-CHANGE EVIDENCE

The explicit later decision states that MIMIC authorization was requested but not obtained, the final academic implementation uses synthetic data, and the population is cardiac/heart-disease focused. No credible later approved artifact was found that changed the prediction clock, history window, three tasks, model families, selection/calibration rules, serving contract, or retrospective replay semantics. No competing approved Scope v2 existed.

## 6. PROJECT_SCOPE_V2 PATH

Authoritative human scope: `docs/governance/project_scope_v2.md`.

Authoritative machine contract: `configs/governance/project_scope_v2.json`.

Future-agent entrypoint: `docs/CODEX_PROJECT_CONTEXT_V2.md`.

## 7. PROJECT_SCOPE_V2 STATUS

`DRAFT COMPLETE — TEAM FREEZE REQUIRED`; machine value `DRAFT_COMPLETE_TEAM_FREEZE_REQUIRED`; approval status `REVIEW_REQUIRED`; `approved_by` is empty. File existence and passing tests do not imply approval.

## 8. PROJECT_SCOPE_V2 SHA-256

- Human scope: `09c2b15a05a87f78cc03b8be3f4e98475e34fdc01787fe6c58191e338e86a880`.
- Machine scope: `4fcab7045b30b631bdbb9e8219226c93288d6ef1c2550125d3456ab431d04994`.
- Delta matrix: `f04e2a151fe278d16f15f43d02f0684bc6c9c380f3d2766b8e2587df511d5a5b`.
- Impact matrix: `53b8dbaf953cb714721264c8e7d84e8ca967af716c68d8d0258290e510ef7ec3`.
- V2 context: `d74de3b2cc89dea44da187efc0a059076afd29f3b9659d44ec25c9fa48805f8c`.

The values are stored and verified in `docs/governance/project_scope_v2_hashes.json`; the manifest itself hashes to `b21a868110f851ce91cdaff36539f16c431ebba85e2611a1d8bd1725acd9170c`.

## 9. ACTIVE DATA SOURCE

`AUTHORIZED_FINAL_SYNTHETIC_DATA`. No real MIMIC patient data are in the final study and no MIMIC-derived experimental result may be claimed. `ENGINEERING_SYNTHETIC_FIXTURE` is a separate scope and cannot be promoted without formal acceptance.

## 10. ACTIVE POPULATION DOMAIN

Adult synthetic patients in `CARDIAC_HEART_DISEASE`. The active scope is not unrestricted ICU and is explicitly not heart-failure-only.

## 11. EPISODE DEFINITION

The unit is one canonical synthetic ICU-like/current critical-care episode per subject, identified by `subject_id`, `stay_id`, explicit `intime`, and explicit `outtime`. If later generation creates multiple episodes, the core retains the chronologically first eligible episode. First/last measurement times cannot substitute for episode bounds.

## 12. RETAINED PREDICTION CLOCK

`t_k = intime + 24h + 6h*k`, `k=0..11`; emit only if `t_k <= outtime - 6h`; at most 12 rows per stay. Event-time information must satisfy `event_time <= t` and all lookback rules.

## 13. RETAINED HISTORY CONTRACT

`(t-48h,t]` in eight disjoint six-hour, left-open/right-closed bins. Values at or before `t-48h` cannot populate or seed the window. Pre-episode positions are structural padding, which remains distinct from clinical missingness. Genuine observation mask, TSLO, and padding mask are separate concepts; train-fitted transformations use training subjects only.

## 14. RETAINED RECOVERY TARGET

Continuous, independent, non-chained `SOFA(t+24h)-SOFA(t)` and `SOFA(t+48h)-SOFA(t)`. Display reconstruction may clip `SOFA(t)+delta` to 0–24; raw-delta evaluation remains unclipped. Trajectory classes and 72-hour deterioration classification are excluded.

## 15. RETAINED REMAINING-STAY TARGET

Raw target: hours from `t` until the current synthetic ICU-like episode's explicit `outtime`. Model target: `log1p((outtime-t) hours)`. It is not discharge ETA, survival, recovery time, remaining bins, or last-measurement time.

## 16. RETAINED THIRD TASK

Twenty-four-hour new organ-support initiation: eligible OFF→ON qualifying vasopressor or invasive mechanical ventilation in `(t,t+24h]`. Support already active at `t` is continuation. An event before episode exit is positive; incomplete negative follow-up is censored.

## 17. RETAINED MODEL FAMILIES

Core: Naive, XGBoost, GRU. Sensitivity: one fixed LSTM per task using selected GRU settings where meaningful. XGBoost and GRU receive identical canonical information; XGBoost only flattens it. Generator latents, future summaries, target classes, and privileged full-episode features are prohibited.

## 18. RETAINED SELECTION / CALIBRATION GOVERNANCE

Exactly 30 validation configurations per task for XGBoost and 30 for GRU; one fixed LSTM sensitivity run. Per-task validation selection remains: recovery lowest stay-balanced +24h MAE; remaining time lowest stay-balanced weighted median absolute error in hours; support highest stay-balanced AUPRC. Only the selected support classifier receives validation-fitted isotonic calibration and a validation-selected predefined-F1 threshold. Final test is not involved.

## 19. RETAINED SERVING / REPLAY CONTRACT

`PredictionPipeline` recomputes from history truncated at `t`; model/explanation routing remains family-specific (TreeSHAP for XGBoost, Integrated Gradients for GRU). Product mode remains `RETROSPECTIVE SEQUENTIAL REPLAY — NOT REAL-TIME CLINICAL PREDICTION`.

## 20. MIMIC REQUIREMENTS SUPERSEDED

For the final scientific run: PhysioNet credentials, MIMIC SQL, item IDs, tables, anchor-year groups, MIMIC cohort counts, and MIMIC-Code-specific SOFA/ventilation provenance. Historical records remain preserved and identifiable as v1 history.

## 21. SYNTHETIC EQUIVALENTS REQUIRED

Future phases must create governed generator version, schema version, config hash, seed policy/provenance, dataset hash, population/cohort spec, split artifact/hash, explicit episode bounds, cutoff-safe six-component severity inputs/flags, and versioned support events/mappings compatible with Pulkit's state machines.

## 22. SPLIT POLICY STATUS

The 2008–2019 MIMIC anchor-era split is `SUPERSEDED`. Retained principles are deterministic subject-level train/validation/final-test separation, immutable artifact/hash, validation-only development, G3-equivalent final-test seal, and no duplicates/clones across splits. Percentages, seed, stratification, and generator-batch holdout are explicitly unresolved before Phase 10.

## 23. SOFA / SEVERITY STATUS

The contract remains a versioned six-component 0–24 SOFA-style score, one cutoff-safe `sofa_at(T)` for `t`, `t+24h`, and `t+48h`, plus six component-observed flags. MIMIC-Code provenance is superseded. The old 0–12 three-component demo proxy is not accepted as SOFA. Implementation is deferred; inability to meet the contract requires a renamed measure and reviewed scientific version change.

## 24. HEART-DISEASE DOMAIN STATUS

Frozen at adult cardiac/heart-disease domain. This is the required boundary for Phase-2 generator design and cannot silently broaden to unrestricted ICU or narrow to heart failure.

## 25. EXACT SUBTYPE TAXONOMY STATUS

`DEFERRED TO PHASE 2 GENERATOR SPECIFICATION`. No subtype names, proportions, or diagnostic taxonomy were invented.

## 26. SCOPE-DELTA MATRIX

`docs/governance/project_scope_v2_delta.csv` contains 53 rows and all 11 required columns. Every row uses one of `RETAINED`, `SUPERSEDED`, `SYNTHETIC_EQUIVALENT`, `DEFERRED_PARAMETER`, or `NOT_APPLICABLE`, identifies affected owners/contracts, states regeneration implications, and remains `REVIEW_REQUIRED`.

## 27. VEDANT IMPACT MATRIX

The matrix covers timestamps, split, tensor schema/loaders, training, all three GRUs, fixed LSTM, metrics, bootstrap, search, selection, calibration/threshold, registry/lineage, G3 guard, data acceptance, reproducibility, and evidence. Core algorithms largely remain unchanged; split/data acceptance/provenance require later successors or adapters; MIMIC-specific lineage/evidence requires regeneration or wording migration.

## 28. PULKIT IMPACT MATRIX

The matrix covers vaso/vent states, composite support label, prediction schema, `PredictionPipeline`, resolver, history/preprocessing seam, explanation routing, IG, TreeSHAP, FastAPI, current-SOFA/recovery reconstruction, dashboard, replay, integration, packaging, and release evidence. Task/serving behavior remains; MIMIC mappings/provenance require future synthetic adapters/configs and final artifacts require regeneration.

## 29. ACTIVE CONTRACTS THAT REMAIN V1

Timestamp semantics (`timestamp_spec_v1`), canonical tensor information semantics (`tensor_contract_v1`), prediction response/task semantics (`prediction_schema_v1`), recovery/remaining/support equations, evaluation estimators, model-family/search policy, calibration/threshold governance, explanation-family routing, `PredictionPipeline`, and replay semantics remain authoritative unless a later reviewed change proves a genuine semantic need for a successor.

## 30. CONTRACTS EXPECTED TO REQUIRE V2

Project/cohort/data scope, synthetic generator and schema, final-data acceptance, synthetic subject split and provenance, synthetic feature dictionary/timeline provenance, synthetic severity input mapping, synthetic support event dictionary/mapping, registry fields that currently demand MIMIC identifiers, reproducibility inputs, and scientific evidence/claims. Exact successors belong to later phases.

## 31. SCIENTIFIC CLAIM LIMITATIONS

Allowed: methodological prototype, software/research framework, evaluation on versioned synthetic cardiac trajectories, retrospective sequential forecasting, integration, and reproducibility demonstrations. Prohibited: MIMIC validation, external clinical validation, bedside accuracy, deployment readiness, patient-specific clinical utility, causal treatment recommendations, real-world prognosis accuracy, or real-time clinical prediction.

## 32. STALE ACTIVE-SCOPE REFERENCES FOUND

The audit classified references into four groups. Historical v1 docs/reviews/evidence were retained; active `AGENTS.md`, README, runbook, and context discovery were minimally amended; actual runtime MIMIC dependencies in real adapter/schema/split/event/label/registry/evidence/history code were documented and deferred rather than patched; the nested candidate remained historical/reference. A final nonhistorical-scope search found no competing declaration that final data are MIMIC, population is unrestricted ICU, or targets are trajectory classification.

## 33. FILES CREATED

- `configs/governance/project_scope_v2.json`
- `docs/governance/project_scope_v2.md`
- `docs/governance/project_scope_v2_delta.csv`
- `docs/governance/project_scope_v2_impact.md`
- `docs/governance/project_scope_v2_hashes.json`
- `docs/CODEX_PROJECT_CONTEXT_V2.md`
- `docs/sanskruti/PHASE1_SCOPE_V2_REVIEW.md`
- `docs/sanskruti/PHASE1_SCOPE_V2_COMPLETION_REPORT.md`
- `tests/test_project_scope_v2.py`

## 34. FILES MODIFIED

- `AGENTS.md`: active scope precedence and synthetic cardiac governance.
- `docs/CODEX_PROJECT_CONTEXT_V1.md`: prominent historical/superseded pointer, without erasing v1 content.
- `README.md`: minimal active-scope and fixture/final-data distinction.
- `RUNBOOK.md`: active amendment and historical MIMIC-mode classification.
- `docs/evidence/models/evidence_manifest_v1.json`: refreshed source hashes only.
- `docs/evidence/system/system_evidence_manifest_v1.json`: refreshed source hashes only.
- `experiments/artifacts.csv`: rehashed the changed Phase-22 evidence manifest registration.
- `docs/evidence/PHASE22_REVIEW_PACKAGE.md`: synchronized that manifest hash.

No file under `src/`, `api/`, `dashboard/`, existing v1 scientific configs, original DOCX sources, generated data, or the candidate repository was modified.

## 35. TESTS ADDED

Fifteen semantic tests validate artifact presence and hashes, non-fabricated approval, final-data/fixture separation, cardiac-not-HF-only population, exact clock/history/tasks, model parity and final-test governance, unresolved synthetic split, replay/nonclinical claims, Phase-1 boundaries, delta completeness and vocabulary, owner impact coverage, and human/machine agreement.

## 36. FOCUSED TEST RESULTS

`tests/test_project_scope_v2.py`: **15 passed in 0.02s**. Evidence/packaging repair subset: **11 passed**. Registry/system-evidence propagation subset: **12 passed**.

## 37. FULL REGRESSION TEST RESULTS

Final: **921 passed, 2 skipped, 0 failed, 0 errors in 10.72s**. The two skips are unchanged optional Captum and SHAP/XGBoost dependency integrations. Python compilation across `src`, `api`, `dashboard`, and `tests` passed.

## 38. EXACT COMMANDS EXECUTED

Core verification commands (run from the integrated root) were:

```bash
PYTHONPYCACHEPREFIX=<temporary-directory> PYTHONPATH=src:tests:. python3 -m pytest tests -q -rs -p no:cacheprovider
PYTHONPYCACHEPREFIX=<temporary-directory> PYTHONPATH=src:tests:. python3 -m pytest tests/test_project_scope_v2.py -q -p no:cacheprovider
PYTHONPYCACHEPREFIX=<temporary-directory> PYTHONPATH=src:tests:. python3 -m pytest tests/test_evidence_traceability.py tests/test_packaging.py tests/test_system_evidence.py -q -p no:cacheprovider
PYTHONPYCACHEPREFIX=<temporary-directory> PYTHONPATH=src:tests:. python3 -m pytest tests/test_system_evidence.py tests/test_repro_registry_trace.py tests/test_serving_lineage_e2e.py -q -p no:cacheprovider
PYTHONPYCACHEPREFIX=<temporary-directory> PYTHONPATH=src:tests:. python3 -m compileall -q src api dashboard tests
shasum -a 256 configs/governance/project_scope_v2.json docs/governance/project_scope_v2.md docs/governance/project_scope_v2_delta.csv docs/governance/project_scope_v2_impact.md docs/CODEX_PROJECT_CONTEXT_V2.md
git -C Heart-Failure-Recovery-and-Deterioration-Trajectory-Prediction status --short
git -C Heart-Failure-Recovery-and-Deterioration-Trajectory-Prediction rev-parse --abbrev-ref HEAD
git -C Heart-Failure-Recovery-and-Deterioration-Trajectory-Prediction rev-parse HEAD
```

Repository discovery used `pwd`, bounded `find`, `rg`, `sed`, `ls`, JSON parsing, CSV parsing, and SHA-256 checks. Two intermediate full-suite runs correctly failed on stale evidence hashes after governance-document edits; the hash chain was updated without weakening tests, and the final full run passed.

## 39. REVIEW REQUIRED ITEMS

Humans must review the full scope/delta/impact package, confirm that the two authorized changes and retained architecture match the team decision, decide whether the functional project title remains sufficient, and explicitly record approval before changing status. Later owners must resolve split and generator parameters through versioned successors rather than editing this draft implicitly.

## 40. VEDANT REVIEW CHECKLIST

- [ ] Exact timestamp grid matches implementation.
- [ ] Synthetic split changes era policy only, retaining isolation.
- [ ] Tensor information contract remains valid.
- [ ] All three tasks remain unchanged.
- [ ] Search/evaluation/selection/calibration governance remains valid.
- [ ] Final-test isolation remains fail-closed.
- [ ] Synthetic provenance can enter registry without fabricated MIMIC fields.
- [ ] No engineering fixture can be promoted to final data.

## 41. PULKIT REVIEW CHECKLIST

- [ ] Retrospective replay wording/semantics remain valid.
- [ ] `PredictionPipeline` remains the sole orchestration path.
- [ ] Support state/composite semantics remain valid.
- [ ] Only MIMIC source mapping/provenance is superseded.
- [ ] Future synthetic events can feed existing state engines.
- [ ] `CurrentSOFAProvider` remains usable with synthetic provenance/flags.
- [ ] Prediction schema remains v1 while task semantics are unchanged.
- [ ] API/dashboard make no clinical-validity or real-time claim.

## 42. SANSKRUTI REVIEW CHECKLIST

- [ ] Generator targets adult cardiac/heart-disease, not HF-only.
- [ ] Explicit episode `intime`/`outtime` are produced.
- [ ] Latent trajectory/target variables cannot leak into inputs.
- [ ] Generator/schema/config/seed/dataset provenance is versioned and hashed.
- [ ] Canonical builder consumes Vedant timestamps.
- [ ] Exact 48-hour/eight-bin semantics are used.
- [ ] Six-component 0–24 SOFA inputs/flags are generated.
- [ ] Support events satisfy Pulkit state interfaces.
- [ ] Preprocessing remains unfitted until split and train-only fitting.
- [ ] XGBoost consumes only the flattened canonical view.

## 43. UNRESOLVED PARAMETERS RESERVED FOR LATER PHASES

Exact cardiac subtype taxonomy/mix, cohort size, generator latent dynamics, missingness mechanism, support frequency, episode-duration distribution, dynamic/static feature inventories, split percentages/seed/stratification/batch policy, generator seed policy, detailed severity inputs/mappings, final package manager, and final environment versions. No arbitrary values were chosen.

## 44. PHASE 1 DEFINITION-OF-DONE CHECKLIST

- [x] Repository and nested candidate audited before modification.
- [x] Baseline integrated test result recorded.
- [x] Human and machine Project Scope v2 artifacts created.
- [x] Synthetic final data and cardiac population declared.
- [x] MIMIC final-run requirements explicitly superseded; history preserved.
- [x] Clock, history, padding/missingness/TSLO, three tasks, censoring, parity, models, search, evaluation, calibration, test isolation, and replay retained.
- [x] MIMIC-era split superseded; replacement principles retained; exact parameters unresolved.
- [x] Synthetic scientific limitations frozen.
- [x] Vedant/Pulkit impact matrices and three review checklists completed.
- [x] No final dataset generated; no training; no final-test access.
- [x] No runtime scientific module or downstream v1 contract changed.
- [x] Actual hashes computed and machine-verified.
- [x] Focused and full tests pass; compilation passes.
- [x] Approval/signoff not fabricated.

## 45. PHASE 1 IMPLEMENTATION STATUS

`PASS` — all governance artifacts, traceability updates, tests, and completion evidence required for Phase 1 are present and verified.

## 46. PROJECT_SCOPE_V2 FREEZE STATUS

`DRAFT COMPLETE — TEAM FREEZE REQUIRED`. It is not approved, frozen, or signed off. Vedant, Pulkit, and Sanskruti approvals remain genuinely absent.

## 47. READINESS FOR SANSKRUTI PHASE 2

`CONDITIONALLY READY`. The technical handoff package is complete and supplies Phase 2 with scope, deltas, impact classification, hashes, retained contracts, limitations, and unresolved generator decisions. Phase-2 implementation must not begin under a claim of frozen authority until the team reviews this package and records an actual freeze/signoff.
