# Sanskruti Rebuild Phase 6 — Synthetic Standard SOFA / Severity Review

## 1. Executive outcome

Phase 6 implemented one deterministic, cutoff-parameterized, six-component SOFA engine and a Pulkit-compatible current-SOFA adapter. The engine, typed results, component flags, provenance binding, support injection, and fixture QA pass. Final production SOFA execution is deliberately blocked because there is no authorized final timeline, the Phase-9 production ventilation/vasoactive-dose contract does not exist, and two project-specific score conventions still require genuine team review. No Phase-7 feature tensor, Phase-8 label, Phase-9 support generator, split, preprocessor, or model was created.

Status summary:

| Deliverable | Status | Evidence |
|---|---|---|
| SOFA threshold source | PASS | Original Vincent et al. 1996 Table 3, DOI `10.1007/BF01709751` |
| Versioned machine spec | PASS FOR ENGINEERING; REVIEW REQUIRED | `configs/synthetic/sofa_spec_v1.json` |
| Six-component engine | PASS | `src/data/synthetic/sofa.py::sofa_at` |
| Six observed flags | PASS | Immutable `ComponentResult` and `SOFAResult` |
| Cutoff/stay/window safety | PASS | Focused boundary and perturbation tests |
| Pulkit CurrentSOFAProvider compatibility | PASS | `SyntheticCurrentSOFAProvider` delegates to `sofa_at` |
| Pulkit state-engine reuse | PASS AT ADAPTER LEVEL | `PulkitStateSOFASupportProvider` calls existing ventilation query logic |
| Fixture full-six-component execution | PASS | Three-case non-scientific QA artifact |
| Final production execution | **BLOCKED** | Phase 9 support/dose provenance, final data, and team review required |

## 2. Active scientific scope

The population remains adult, broader cardiac/heart-disease-focused, synthetic critical-care episodes. It is not renamed or narrowed to heart-failure prediction. SOFA is a generic organ-dysfunction recovery quantity within that synthetic cardiac cohort. The implementation supplies methodological score consistency only; it does not validate synthetic trajectories clinically and makes no real-patient prognostic claim.

## 3. Authority inspection

The implementation followed the repository authority order:

1. `docs/governance/project_scope_v2.md` retains a six-component, 0–24, cutoff-safe SOFA-style calculation and one `sofa_at(T)` for all horizons.
2. No reviewed current repository threshold table existed.
3. Phase-5 canonical concept and provenance contracts supplied all nonsupport raw domains.
4. `src/serving/recovery.py` supplied Pulkit's actual `CurrentSOFAState` and `CurrentSOFAProvider` receiver shape.
5. Existing Pulkit state engines supplied the reusable invasive-ventilation state seam.
6. The original 1996 SOFA publication supplied numeric thresholds.
7. The historical Sanskruti proxy was inspected only as rejection evidence, never as scientific authority.

Project Scope v2 path/hash: `docs/governance/project_scope_v2.md`, SHA-256 `09c2b15a05a87f78cc03b8be3f4e98475e34fdc01787fe6c58191e338e86a880`.

## 4. Prerequisite Phase 1–5 inspection

Repository reality, not prompt-suggested filenames, was used. The inspection found Project Scope v2; generator spec/schema; raw fixture and smoke manifests; Phase-4 retained cohort and structural indices; Phase-5 processed schema, concept map, feature provenance, dictionary, canonical statics/timeline, summary, and processed manifests.

Both Phase-5 bundles were revalidated through `validate_processed_manifest`:

- fixture: `ENGINEERING_CANONICAL_TIMELINE_FIXTURE`, 4 stays, 1,222 events, zero nonretained rows;
- smoke: `ENGINEERING_CANONICAL_TIMELINE_SMOKE`, 23 stays, 7,708 events, 162 nonretained source rows filtered.

The smoke processed manifest is `e73456990afdac79b155276f88ea5a1fb10a8c316ec35a18738f0664d2314c9b`; the canonical timeline is `71b5eaabaf65d5a99df3646e749d2ac38cbd7a17d6131f3b855e59c3b92cd233`.

## 5. Authoritative SOFA source

Thresholds are transcribed from Vincent JL, Moreno R, Takala J, et al., “The SOFA (Sepsis-related Organ Failure Assessment) score to describe organ dysfunction/failure,” *Intensive Care Medicine* 1996;22:707–710, Table 3, DOI `10.1007/BF01709751`. Verification used the [ESICM-hosted paper](https://www.esicm.org/wp-content/uploads/2018/03/esicmworking5-2.pdf) on 2026-09-20. No numeric threshold was accepted from the old proxy or silently recalled as project truth.

The paper defines the named score table; it does not settle this project's arbitrary-cutoff event-pairing and endpoint closure. Those project-specific decisions remain conspicuously review-required rather than being attributed to the paper.

## 6. Versioned specification

`configs/synthetic/sofa_spec_v1.json` defines `synthetic_standard_sofa_v1` and hashes to `900504be9b2d1186c15b5893fcd1ea5e4aa3d3c54c60b44d7b4b0079c79e4c13`. It contains the reference, range, ordered components, threshold rules, concepts, units, Phase-5 domains, window, selection, support needs, missingness, observed semantics, prohibited shortcuts, and provenance policy.

Its status is `DRAFT_REVIEW_REQUIRED_NOT_AUTHORIZED_FOR_FINAL_PRODUCTION`. That status is substantive, not clerical. The team must review `(T-24h,T]` and exact-time PaO2/FiO2 pairing, and Phase 9 must freeze support evidence.

## 7. Exactly one function

`src/data/synthetic/sofa.py` contains one public scoring function: `sofa_at(history, stay, cutoff, *, spec, bindings, support_provider)`. It has no horizon argument. Static tests reject `baseline_sofa`, `future_sofa`, `dashboard_sofa`, `serving_sofa`, and `label_sofa` alternatives.

Phase 8 must call exactly:

```text
baseline = sofa_at(t)
future24 = sofa_at(t + 24h)
future48 = sofa_at(t + 48h)
Delta24 = future24 - baseline
Delta48 = future48 - baseline
```

The deltas are independent. Phase 6 does not compute or persist them.

## 8. Time and episode contract

The engine permits any timezone-aware cutoff inside `[intime,outtime]`, not only a prediction-grid row. This is necessary because `t+24h` and `t+48h` are label evaluation times. The draft score window is `(T-24h,T]`: exact left is excluded, a microsecond inside is included, `T` is included, and anything after `T` is ignored before scientific validation.

Rows are first isolated to exact subject/stay. Scoreable rows must lie inside the episode. Events from another stay/subject, at/before the left edge, or after the cutoff cannot affect scores or evidence.

## 9. Component result contract

Each immutable `ComponentResult` carries component name, integer score, observed boolean, selected evidence references, selected source timestamps, and specification version. Validation rejects unknown names, booleans/nonintegers, values outside 0–4, and nonzero/evidenced unobserved components.

`observed=true` means sufficient genuine score evidence existed. It does not mean abnormal. Tests prove an all-normal case has six zero scores and six true flags.

## 10. Total result contract

Immutable `SOFAResult` carries subject, stay, cutoff, total, ordered component objects, spec identity, implementation version, processed schema identity, concept-map identity, provenance identity, timeline source identity, support-contract identity, and deterministic binding hash. Validation requires the exact component order, total equality to the six scores, and range 0–24. It fails rather than clipping.

## 11. Respiratory scoring

The scorer uses canonical `pao2` in mmHg and `fio2` as a fraction. The engineering contract pairs only observations with the same availability timestamp and queries support state at that time. It applies the reference P/F ratio categories and the invasive-support requirement for scores 3 and 4. Multiple pairs are scored and the worst score retained.

Missing PaO2, missing FiO2, or no pair makes the component clinically unobserved only when the support provider itself is working. A pair without known support state is a pipeline dependency error. Wrong units, nonfinite values, and Phase-5 domain violations fail closed.

Important blocker: Phase-3/5 PaO2 and FiO2 observations are asynchronous in existing engineering outputs. Exact-time pairing avoids inventing carry-forward, but must be reviewed or matched by a revised authorized generator before final scoring.

## 12. Coagulation scoring

Canonical `platelet_count`, unit `10^9/L`, maps to reference cutoffs below 150, 100, 50, and 20 for scores 1–4. Every in-window point is evaluated; the highest component score is selected. Exact boundaries and epsilon-below cases are tested.

## 13. Liver scoring

Canonical `bilirubin_total`, unit `mg/dL`, maps at 1.2, 2.0, 6.0, and strictly above 12.0 for scores 1–4, matching the cited table's comparison symbols. Every candidate is scored, the worst is retained, and unit/domain errors fail.

## 14. Cardiovascular scoring

Canonical `mean_arterial_pressure` below 70 mmHg scores 1. Versioned vasoactive evidence supports dobutamine, dopamine, epinephrine, and norepinephrine reference categories in `ug/kg/min`. At least 60 minutes of exposure overlapping the 24-hour window is required. Exposure ending after cutoff is rejected rather than reading a future cessation time. The worse MAP or dose score is retained.

No dose equivalence and no generic `on_pressor` approximation exists. Pulkit's endpoint state is binary and intentionally insufficient for standard SOFA dose categories. Phase 9 must deliver normalized agent/rate/unit/interval evidence.

## 15. Central nervous system scoring

Canonical `glasgow_coma_scale` must be an integer from 3 through 15. Reference bands are 15, 13–14, 10–12, 6–9, and below 6. No sedation correction, imputation, or undocumented clinical adjustment is introduced.

## 16. Renal scoring and interval correctness

Creatinine uses canonical mg/dL reference cutoffs. Urine volume remains a quantity-over-interval observation available at interval end; it is never treated as a point or full-stay total. Only a nonoverlapping, exactly contiguous set covering the entire `(T-24h,T]` window is eligible for the 24-hour urine thresholds. No partial interval is prorated. The component takes the worse valid creatinine or complete urine score.

Incomplete urine coverage does not create an artificially severe low-output score. Creatinine can independently make renal observed. A complete urine series can independently make it observed.

## 17. Worst-value selection and deterministic ties

The engine does not use generic latest/mean/min/max values. It scores each scientifically valid candidate using component-specific rules and then selects the maximum score. Tied determining evidence is sorted by source identity. Same-timestamp input row order therefore cannot alter the result.

## 18. Missing-component convention

The retained convention is exactly zero points plus `observed=false` for a clinically unobserved component. There is no old `>50% missing`, majority-observed, feasibility threshold, forward fill, population mean, median imputer, or learned estimate. The all-missing fixture produces total zero with six false flags and is explicitly not interpreted as good data quality.

## 19. Dependency-unavailable versus clinical missingness

This distinction is executable:

- working support provider, no MAP, and known absence of pressor exposure: cardiovascular score zero, observed false;
- absent provider or unknown support coverage: `SOFADependencyUnavailable`, final score blocked.

Unknown is never coerced to inactive. A support version/hash mismatch also fails closed.

## 20. Pulkit state-engine integration

`PulkitStateSOFASupportProvider` delegates invasive-state queries to `labels.ventilation_state.query_invasive_ventilation_state`. It does not implement a competing production ventilation state machine. It combines that result with injected normalized Phase-9 dose segments because existing binary vasopressor state lacks the rate information SOFA requires. The adapter rejects an interval whose not-yet-observed end extends past the score cutoff; Phase 9 must provide a cutoff-truncated state/evidence representation for ongoing support rather than expose future cessation.

The current `configs/event_dict_v1.yaml` remains draft/MIMIC-era and hashes to `717f6183ad8014602b3cc32eeaa3a810eab8a26a19937bda8e214200bb2f9620`; it cannot authorize final v2 synthetic support semantics. Engineering synthetic-mode reuse is tested; final data execution remains blocked.

## 21. Pulkit CurrentSOFAProvider integration

`SyntheticCurrentSOFAProvider` validates the complete Phase-5 processed manifest, reloads canonical history/statics, cross-checks SOFA concept names/units/domains/event kinds against the Phase-5 concept map, and calls `sofa_at`. It then returns Pulkit's actual `CurrentSOFAState`: stay, exact requested prediction time, total, SOFA version, source version/hash, and six observed flags.

The adapter optionally accepts Phase-4 legal cutoffs and rejects requests outside them. Direct score and adapter parity are tested. The adapter never reads recovery labels, future SOFA, or model predictions.

## 22. Cutoff-safety evidence

Tests cover exact left edge, just inside it, exact cutoff, immediately after cutoff, future high-severity rows, old high-severity rows, other-stay rows, future-only support intervals, cutoff outside episode, and provider parity. Adding a severe platelet result immediately after `T` or bilirubin older than the window leaves the entire immutable result unchanged.

This is structural: filtering occurs before component evaluation. Evidence references and timestamps are therefore also limited to the permitted window.

## 23. Canonical units and value validation

The spec repeats Phase-5 unit/domain identities and the provider validates them against `concept_map_v1`. Runtime scoring rejects wrong units, nonfinite values, GCS nonintegers, invalid FiO2, and any Phase-5 domain violation. Values are not clipped to a threshold.

Processed schema SHA-256: `7e0ba3d3789b5b469198773a5b1a7af2062ad973466a4af0135fdf97383688d9`. Concept map SHA-256: `24de193ea0041c128d921d1585dff17cd461af5fc7c415853449aa400e7bf74d`. Provenance CSV SHA-256: `ed6c06c04533bd8b07cef585c178e3d9d17fb3745d3aaba0cb0a736ef9b7757a`.

## 24. Historical proxy rejection

The nested historical candidate's `src/preprocessing/sofa.py` is not reused. Prior acceptance evidence establishes that it is a demo 0–12 proxy, activates only respiratory/cardiovascular/renal approximations, pins three domains to zero, uses SpO2 and heart-rate shortcuts, lacks the standard rolling 24-hour six-component logic, has no component-observed flags, and is not compatible with the current receiver. Phase-6 source imports none of it.

## 25. Fixture QA

`artifacts/data/synthetic/sofa/fixture/phase6_fixture_v1/sofa_fixture_qa.json` is explicitly `ENGINEERING_SYNTHETIC_FIXTURE_NON_SCIENTIFIC`. Three recomputable golden cases cover all-observed normal, all-six maximum, and all-missing with known support coverage. Scores range 0–24; totals are zero twice and 24 once; every component is observed in two of three cases; all-six is observed in two of three.

The stored summary is recomputed in tests through `summarize_sofa_results`; it contains no model metric, recovery delta, or scientific production claim.

## 26. Production availability QA

Not run. There is no authorized final canonical timeline and no production Phase-9 support evidence. Running smoke data with fabricated “inactive” support would confuse absent implementation with clinical absence, so the implementation deliberately refuses to present that as production availability.

Feasibility threshold: **NOT FROZEN / NOT REQUIRED FOR CODE CORRECTNESS**. No numeric availability gate was invented and the recovery target was not changed.

## 27. Lineage artifact

`artifacts/data/synthetic/sofa/phase6_sofa_lineage_v1.json` binds the active scope, Phase-5 manifest/timeline, processed schema, concept map, provenance dictionary, SOFA spec, engine, current-SOFA provider, existing support dictionary status, fixture QA, and parent code commit. It explicitly lists later artifacts not created.

The lineage reports final production as `BLOCKED_PHASE9_SUPPORT_DEPENDENCY_AND_TEAM_REVIEW_REQUIRED`. Hash-verification tests recompute every listed file identity.

## 28. Baseline test result

The repository's bare `pytest` executable is not installed (`command not found`), which is an environment diagnostic rather than a test failure. The documented isolated command was then used:

```text
PYTHONPYCACHEPREFIX=<temporary> PYTHONPATH=src:tests:. python3 -m pytest tests -q -rs -p no:cacheprovider
```

Before edits: **1,061 passed, 2 skipped, 0 failed, 0 errors in 16.90 s**. The skips were optional Captum and SHAP/XGBoost integrations.

## 29. Focused Phase-6 test result

Six focused Phase-6 modules contain **89 passing tests**. They cover spec structure, six component/range/sum invariants, all golden missingness cases, every reference boundary, worst selection, urine-versus-creatinine, window closure, future/old/other-stay attacks, invalid values/units, dependency tri-state, support hash mismatch, same-function API guard, Pulkit state reuse, current-provider parity/legal cutoff/stay isolation, QA recomputation, and lineage hashes.

Command result: **89 passed in 1.56 s**, zero skipped/failures/errors.

## 30. Full regression result

After implementation, the isolated integrated suite reported **1,150 passed, 2 skipped, 0 failed, 0 errors in 18.17 s**. The two unchanged skips are:

- Captum absent for optional Integrated Gradients integration;
- SHAP or XGBoost absent for optional TreeSHAP integration.

No final-test access, model run, split, calibration, or freeze state was invoked.

## 31. Static checks

`python3 -m compileall -q src tests scripts` passed with an isolated bytecode directory. `git diff --check` passed. Repository search found no configured `pyproject.toml`, `setup.cfg`, `tox.ini`, flake8, mypy, or Ruff configuration at the project root, so no formatter/linter/type-check result is claimed.

## 32. Engine Definition of Done

- [x] Phase 1–5 artifacts and scope inspected.
- [x] SOFA recovery retained; no scope conflict.
- [x] Authoritative original threshold reference identified and versioned.
- [x] Exactly six 0–4 components; exact 0–24 sum.
- [x] One cutoff-only `sofa_at` implementation.
- [x] Six independent observed flags; zero-versus-missing tested.
- [x] Missing component contributes zero; no 50-percent rule.
- [x] Respiratory, coagulation, liver, cardiovascular, CNS, renal logic implemented.
- [x] Urine interval availability and complete-window rule enforced.
- [x] Future, old, other-stay, post-episode, unit, and domain safeguards.
- [x] Missing dependency is not clinical missingness.
- [x] Support interface injected; Pulkit ventilation state reused by adapter.
- [x] Spec and source identities bound; result provenance typed.
- [x] Pulkit current-SOFA adapter uses the exact engine.
- [x] Fixture QA, focused tests, full regression, compile, and diff checks pass.
- [x] No recovery labels, feature grid, split, preprocessing, or model work.

## 33. Production execution Definition of Done

- [ ] Authorized final canonical timeline exists — **not available**.
- [x] Nonsupport raw dependencies represented in Phase 5.
- [ ] Production respiratory-support evidence available — **Phase 9 blocked**.
- [ ] Production agent/rate/unit exposure evidence available — **Phase 9 blocked**.
- [ ] Synthetic support contract/version frozen — **not frozen**.
- [ ] Project window and respiratory-pairing conventions reviewed — **review required**.
- [ ] Final current-SOFA execution on retained stays — **not authorized**.
- [ ] Production availability QA — **not generated**.

Final production status: **BLOCKED — PHASE-9 SUPPORT DEPENDENCY AND TEAM REVIEW REQUIRED**. This does not invalidate the passing engine or fixture implementation.

## 34. Phase-7 handoff

Phase 7 receives the active scope identity, Phase-5 processed manifest/timeline/schema/concept/provenance identities, `synthetic_standard_sofa_v1` and its computed hash, `sofa_at`, typed component flags, current-SOFA provider status, support blocker, lineage, and this review. Phase 7 must not reimplement SOFA or inject its total into model features without its own frozen schema decision.

Phase 7 remains responsible for `(t-48h,t]`, eight 6-hour bins, missingness masks, TSLO, and padding. None is implemented here. The SOFA 24-hour score window is independent of that future model-history representation.

## 35. Phase-8 handoff caveat

The engine supports arbitrary valid episode cutoffs and therefore the required future calls, but Phase 8 must wait for the production dependencies/reviews above. It must compute both deltas independently from the same implementation, apply horizon eligibility, and create label artifacts itself. There is no 24-to-48 chaining and no recovery dataset in Phase 6.

## 36. Blocker-resolution sequence

Before final production SOFA:

1. review/freeze the SOFA score-window endpoint convention;
2. review the PaO2/FiO2 pairing policy or align authorized generation to paired facts;
3. have Phase 9 freeze synthetic invasive-ventilation and vasoactive agent/rate/unit/interval provenance;
4. expose cutoff-truncated support evidence through the adapter without future cessation leakage;
5. authorize a final Phase-5 canonical timeline after upstream data freeze;
6. rerun component availability QA and store its computed hashes;
7. review/freeze the spec status through versioned change control.

No workaround should mark unknown support as inactive or fabricate final events merely to close these boxes.

## 37. Final Phase-6 boundary statement

Phase 6 stops here. The completed work is the specification-backed engine, evidence contracts, Pulkit adapters, fixture validation, provenance, and explicit blockers. It does not include Phase 7 feature construction, Phase 8 recovery/remaining-time labels, Phase 9 event generation, Phase 10 split/preprocessing, or any training/evaluation activity.
