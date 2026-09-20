# 1. EXECUTIVE DECISION

> **Sanskruti Acceptance Audit — final, audit-only review**  
> Audit date: 2026-09-20 (Asia/Kolkata)  
> Integrated root: `/Users/Home/Downloads/HealthCareProject`  
> Candidate root: `/Users/Home/Downloads/HealthCareProject/Heart-Failure-Recovery-and-Deterioration-Trajectory-Prediction`

| Decision question | Answer |
|---|---|
| Recommendation | **REBUILD (category D)** |
| Would I merge/use Sanskruti's implementation in the integrated project today? | **NO** |
| Should we build Sanskruti's track ourselves instead? | **YES — REBUILD IS SAFER** |
| Confidence | **HIGH** |

The candidate is not a slightly different implementation of the frozen project. It implements an older heart-failure-only, admission-oriented, 1-hour, 24/48/72-hour deterioration-classification project. The current project requires an all-adult, first-ICU-stay, cutoff-indexed, eight-bin forecasting dataset with continuous independent SOFA deltas, log remaining-current-ICU time, and a Pulkit-owned organ-support endpoint. The mismatch is foundational.

The decision rests on code, not names or performance. The candidate hard-codes the wrong cohort (`src/data/cohort_builder.py`), wrong history construction (`src/preprocessing/grid_builder.py`), proxy rather than pinned SOFA (`src/preprocessing/sofa.py`), wrong labels (`src/targets/generate_targets.py`), all-data imputation (`src/preprocessing/missing_values.py`), a competing random split and test evaluation (`src/models/xgboost_baseline.py`), and incompatible/demo serving. A compatibility adapter would have to reinterpret scientific meaning, not merely rename fields.

Quality is deliberately separated:

| Quality dimension | Assessment |
|---|---|
| Scientific correctness for the final Master Plan | **FAIL** — wrong cohort, time axis, SOFA, targets, split, and information set |
| Software quality/reproducibility in isolation | **PARTIAL** — modular source and tests exist, but real mode is incomplete, versions are unpinned, artifacts lack lineage, and the suite does not collect in the current environment |
| Integration compatibility | **FAIL** — no `real_data_input_manifest_v1`, canonical example, receiver-compatible timeline, exact tensor views, SOFA provider, or loadable selected-model artifact |

Confidence is high because the decisive failures are explicit code paths and configs and do not depend on access to MIMIC. Confidence is limited only for claims requiring real-data/schema inspection: no authorized MIMIC connection was available; Parquet payloads could not be decoded because `pyarrow` is absent; and candidate tests could not collect under the current Python/XGBoost runtime. Those limitations are marked **BLOCKED**, not inferred as failures.

# 2. REPOSITORY TOPOLOGY

| Property | Integrated root | Candidate root |
|---|---|---|
| Absolute path | `/Users/Home/Downloads/HealthCareProject` | `/Users/Home/Downloads/HealthCareProject/Heart-Failure-Recovery-and-Deterioration-Trajectory-Prediction` |
| Git repository | No | Yes; the only nested `.git` found |
| Branch/HEAD | Not applicable | `main`, `dc95164b808becd6e822bccd28c0135303753e40` |
| Recent log | Not applicable | `dc95164 cleanup`; `66b3376 Initial commit` |
| Worktree | Not applicable | Clean after removal of audit-created SQLite WAL sidecars |
| Package layout | `src/`, `tests/`, `configs/`, `docs/`, `scripts/` | `src/`, `tests/`, `config/`, `backend/`, `frontend/`, `data/`, `outputs/`, `notebooks/` |

The candidate is a separate Git repository nested physically inside a non-Git integration directory. It has 187 tracked files, including 67 Python files, two notebooks, 14 Parquet files, 11 JSON files, four YAML files, and no SQL. The largest tracked files are synthetic/demo Parquet outputs; the largest two are 3,145,332 bytes each.

Import isolation is unsafe unless commands explicitly choose a root. Running parent discovery without restricting the test path traversed the nested suite and produced 12 collection errors. Four parent dashboard tests were then affected by the candidate's top-level `tests` package shadowing. With `cwd` at the parent and `PYTHONPATH=src:tests:.`, `data.timestamps` and `serving.pipeline` resolved to the integrated tree. With `cwd` at the candidate and `PYTHONPATH=.`, candidate imports began in the candidate tree but failed in `src/utils/helpers.py` before a meaningful integration probe.

Current runtime: Python 3.9.6, pandas 2.3.3, NumPy 2.0.2, scikit-learn 1.6.1, PyYAML 6.0.3, FastAPI 0.128.8, and Pydantic 2.13.5. `pyarrow`, Captum, SHAP, Black, Ruff, and mypy are unavailable. `xgboost` is discoverable but import fails because the platform OpenMP library is missing. The candidate README claims Python 3.10+, and `str | Path` annotations in `src/utils/helpers.py` are evaluated under 3.9, causing collection failure. Requirements use lower bounds rather than an exact lock; there is no `pyproject.toml`, lock file, tox config, or configured static-analysis policy.

# 3. AUTHORITATIVE DOCUMENTS FOUND

| Authority | Exact file | SHA-256 | Use in this audit |
|---|---|---|---|
| Scientific | `/Users/Home/Downloads/Personalized_Patient_Recovery_Trajectory_FINAL_Implementation_Master_Plan.docx` | `a819b6144061780fac070537b2a29a50d8b3ea24d7a82d4cb7235985d0ab5588` | Cohort, cutoffs, lookback, SOFA, labels, eligibility, model comparison, calibration, metrics |
| Ownership | `/Users/Home/Downloads/Personalized_Patient_Recovery_Trajectory_FINAL_Work_Division_3_Members (1).docx` | `3d07c6283896b468f9cb013a669412dd8e1e96ff8aabe350c0f8cb7601b636a0` | Sanskruti/Vedant/Pulkit packages and handoffs |
| Repository context | `docs/CODEX_PROJECT_CONTEXT_V1.md` | `ae61235ad7f71863819911bc131731113e7fd3e02b876139bdc9c365bb2bfe22` | Versioned restatement of final documents and phase/gate controls |
| Sanskruti receiver package | `docs/evidence/PHASE18_SANSKRUTI_REVIEW_PACKAGE.md` | Repository file inspected | Concrete `real_data_input_manifest_v1` acceptance requirements |

The Master Plan and Work Division agree on the material invariants. Candidate README/config claims conflict with them: README lines 20–27 specify heart failure, improving/stable/deteriorating classes, and 24/48/72-hour horizons; `config/config.yaml` lines 20–31 specifies 1-hour resolution, a 24-hour observation window, those three horizons, and a separate fixed 8×6 grid. Candidate documentation is therefore historical project documentation, not authority.

# 4. CURRENT VEDANT RECEIVER CONTRACT

The current receiver is concrete and versioned:

- `configs/timestamp_spec_v1.yaml` freezes `t = intime + 24h + 6h*k`, `k=0..11`, retaining only `t <= outtime-6h`; the history is `(t-48h,t]`, with pre-ICU positions as padding.
- `src/data/timestamps.py` implements the canonical timestamp path; candidate code does not call it.
- `configs/tensor_contract_v1.json` requires `[B,8,F]` dynamic values plus observation masks, TSLO, padding, statics, identities, targets, and eligibility.
- `src/data/schema.py:36-49` defines `FeatureSchemaReference`; lines 52–65 define task eligibility/targets; lines 68–93 define `CanonicalExample` with `subject_id`, `stay_id`, `prediction_time`, `grid_index`, values, observation mask, TSLO, padding, statics, eligibility, and targets.
- `src/data/schema.py:536-574` creates XGBoost and recurrent views from the same canonical information and supports unflattening. XGBoost is allowed to flatten, not enrich.
- `src/data/real_adapter.py:40-82` fixes manifest version and required artifacts/concepts; `load_real_bundle()` at lines 372–457 checks hashes, inventory, versions, mappings, order, finite sentinel, same/pinned SOFA, leakage evidence, exact formulas, and train-only preprocessing provenance.
- `docs/evidence/PHASE18_SANSKRUTI_REVIEW_PACKAGE.md:15-26` requires the manifest, canonical/schema artifacts, provenance, structural index, split compatibility, SOFA pin/flags/formulas, leakage evidence, and receiver reviews before Phase 19.

This receiver intentionally leaves raw-data production injectable but does not leave scientific semantics open. The candidate provides no object or artifact satisfying the contract.

# 5. CURRENT PULKIT RECEIVER CONTRACT

- `src/serving/history.py` defines production timeline/truncation protocols; the production history source remains intentionally injected.
- `src/serving/preprocessing.py:33+` defines `CanonicalHistoryInputProvider`, which accepts a builder and requires a real `CanonicalExample`; it obtains model views through the shared `temporal_information_views()` path.
- `src/serving/recovery.py:31-56` defines `CurrentSOFAState`: stay and cutoff identity, score in `[0,24]`, SOFA/source versions and hashes, and six component-observed flags. `CurrentSOFAProvider` is at lines 85–89.
- `src/serving/artifacts.py` expects `selected_models_calibrated_v1`, strict hashes, feature/task contracts, recovery output `independent_delta_24h_delta_48h`, ICU output `log1p_remaining_current_icu_hours`, and an uncalibrated support probability before the frozen calibrator.
- `PredictionPipeline`, model/explanation routing, replay, and the API consume those contracts; they are not alternate places to reconstruct scientific features.

The candidate supplies neither a cutoff-aware timeline builder nor a current-SOFA object with provenance/flags. Its separate backend is not an implementation of these seams.

# 6. SANSKRUTI IMPLEMENTATION INVENTORY

The candidate contains a coherent older-project scaffold:

| Area | Principal files | Actual role |
|---|---|---|
| Cohort/loading | `src/data/cohort_builder.py`, `src/data/load_data.py` | Heart-failure admissions/demo data; partial MIMIC loader |
| Extraction | `src/data/extract_features.py` | Hard-coded item-ID extraction; real pipeline incomplete |
| Preprocessing | `src/preprocessing/time_alignment.py`, `clean_data.py`, `missing_values.py`, `feature_engineering.py` | Admission-keyed 1-hour processing and engineered summaries |
| Grid | `src/preprocessing/grid_builder.py` | First 48 hours from first observed timestamp, fixed 8×6-hour bins |
| SOFA | `src/preprocessing/sofa.py`, `docs/SOFA_DEFINITION.md` | Demo proxy score, range 0–12, only three active proxy components |
| Labels | `src/targets/generate_targets.py` | 24/48/72-hour deterioration and categorical/proxy delta targets |
| XGBoost | `src/models/xgboost_baseline.py`, `src/models/xgboost_validation_search.py` | Random-split classification and one-task validation search |
| Applications | `backend/`, `frontend/` | Separate demo API/dashboard; overlaps Pulkit ownership |
| Entrypoints | `run_pipeline.py`, `run_sofa_pipeline.py` | Demo-oriented; real execution is not complete |
| Evidence | `data/`, `outputs/`, notebooks | Tracked synthetic Parquet/figures/metrics without scientific lineage |

Key file hashes include: README `471e9c…09fca`; config `605caf…7ffa`; cohort builder `85c8ff…bd74`; grid builder `60d69a…21f6`; SOFA `baf457…011c`; target generator `a3857e…e681`; XGBoost baseline `e1ad2c…7290`; requirements `e9c025…34ded9`.

# 7. WORK PACKAGE A — EXTRACTION

**Status: FAIL; real schema verification BLOCKED.**

`src/data/load_data.py:72-82` checks only a narrow subset of real files. Lines 124–179 read admissions, diagnoses, patients, and all chart events, but return an admission/demographic merge; diagnoses and chart events are not converted into the required canonical timeline. Lines 50–66 silently fall back to demo mode when real data are absent, which is unsafe for a scientific run.

`src/data/extract_features.py:29-51` hard-codes “commonly used” item IDs without a verified MIMIC-IV v2.2 source manifest. Lines 132–174 mix chart/lab extraction assumptions; final long output drops the source item identifier needed for traceability and keys rows by patient/admission/timestamp rather than ICU stay. The real orchestration path raises `NotImplementedError` around lines 247–253. There is no SQL, extraction manifest, source schema snapshot, code/config hash binding, sample trace, or declared MIMIC-Code commit.

No authorized MIMIC root/connection was present. Therefore whether individual table/column/item-ID assumptions are valid is **BLOCKED — REAL MIMIC SCHEMA VERIFICATION REQUIRED**. No table, field, or ID is accepted from memory. This blocked verification does not rescue the implementation: reproducible extraction and required provenance are absent in code.

# 8. WORK PACKAGE B — FIRST-STAY COHORT

**Status: FAIL (CRITICAL).**

`src/data/cohort_builder.py:104-193` selects heart-failure-coded admissions. The ICU join around lines 165–173 uses `drop_duplicates` over patient/admission without chronological sorting and does not implement chronologically first ICU stay per subject. It does not enforce the adult cohort, deterministic tie handling, anchor-year validity, `outtime > intime`, or at least one legal canonical cutoff. It renames hospital admission/discharge fields around lines 179–186 and computes hospital LOS. The real build path remains `NotImplementedError` around lines 267–290.

There is no evidence of the required “no global exclusion of hospital deaths” logic because the required cohort is not implemented. The synthetic summary reports 100 demo heart-failure patients, not a reproducible MIMIC cohort flow.

# 9. WORK PACKAGE C — CANONICAL TIMELINE

**Status: FAIL (CRITICAL).**

The candidate has no `processed_schema_v1` equivalent. `src/preprocessing/time_alignment.py:63-117` floors to 1-hour timestamps and groups by patient/admission rather than `stay_id`. `src/data/extract_features.py` reduces provenance while pivoting by mean. `src/preprocessing/clean_data.py:53-65` drops timestamp duplicates keeping “last” without a stable source ordering; lines 88–105 clip ranges without a source-level trace.

The output does not retain the required source table/field/ID, event-time interpretation, normalized-unit conversion, current-stay identity, or explicit post-outtime exclusion. It is not sufficient for Pulkit's history provider.

# 10. WORK PACKAGE D — PROVENANCE

**Status: FAIL (MAJOR).**

`data/processed/data_dictionary.csv` has feature name, broad unit, description, generic source, and engineered flag. It does not record source table, source field, event-time field, item/concept identifier, source unit, conversion, aggregation, validity rule, cutoff/leakage status, schema version, or immutable provenance ID. Its SHA-256 is `4d0a760c…8d2f2e` but that audit-computed hash is not producer lineage.

No `feature_provenance_v1`, feature schema sidecar, or artifact manifest exists. Engineered rolling fields in `src/preprocessing/feature_engineering.py` are not mapped to the receiver's ordered feature schema.

# 11. WORK PACKAGE E/F — SOFA + SOFA DEPENDENCIES

**Status: FAIL (CRITICAL).**

`src/preprocessing/sofa.py:1-26` explicitly describes a demo proxy. It activates only proxy respiratory, cardiovascular, and renal components and pins the other three to zero. Respiratory uses SpO2 rather than PaO2/FiO2 plus ventilation (`:53-60`, `:81-88`); cardiovascular uses MAP/heart rate rather than verified vasoactive exposure (`:62-69`, `:91-103`); renal is creatinine-only. Missing values are treated as normal. It is not a rolling 24-hour `sofa_at(T)` and has no six observed-component flags.

The version is `1.0-demo` (`:122-142`). `docs/SOFA_DEFINITION.md` documents a 0–12 proxy and cites a paper rather than an immutable MIMIC-Code commit. Repository search found no MIMIC-Code pin. There is no executable dependency provenance for platelets, bilirubin, GCS, urine output, PaO2/FiO2, invasive ventilation, or vasoactive doses. No sampled comparison against a pinned reference exists.

Ventilation alignment with Pulkit is **BLOCKED/FAIL**: there is no candidate pinned ventilation concept to compare, and the proxy does not supply Pulkit's invasive/non-invasive category and interval semantics.

# 12. WORK PACKAGE G/H — FEATURE GRID + STRICT LOOKBACK

**Status: FAIL (CRITICAL).**

`src/preprocessing/grid_builder.py` builds the first 48 hours of each admission. It sets `t0` to the first observed timestamp around line 134, retains elapsed hours `<48` around lines 137–143, and uses left-closed/right-open bins such as `[0,6)`. The frozen receiver requires a different grid for every legal cutoff and bins `(t-48,t-42] … (t-6,t]`.

It has no `prediction_time`, canonical `grid_index`, `subject_id`, or `stay_id`. It creates trailing empty bins and calls them padding, while the frozen contract uses padding only for nominal pre-ICU history. Feature absence and structural padding are conflated through `*_padded`/`bin_padded`. TSLO is derived from bin end rather than true last event time; an observed value commonly receives TSLO zero, and the no-observation behavior is an increasing bin-end/48 value rather than the frozen sentinel.

`src/preprocessing/missing_values.py:11-15` states that it is called before splitting and fitted on all data. Forward-fill at lines 106–116 is patient/admission-wide and can be seeded by observations at or before `t-48h`. Thus strict lookback fails by construction. No mandated 60-hour-old-value test, future-row perturbation equality, raw-history truncation equality, genuine-observation-mask test, or correct padding-vs-missingness test exists.

# 13. WORK PACKAGE I — RECOVERY LABELS

**Status: FAIL (CRITICAL).**

`src/targets/generate_targets.py:1-29` centers the older three-class/binary deterioration problem at 24/48/72 hours. `generate_delta_sofa_targets()` around lines 279–435 operates on bin scores rather than a pinned `sofa_at(T)`. It converts continuous deltas into categories around lines 369–386; the 48-hour target uses an end-of-stay/max-bin proxy around lines 388–405 and is generated only from an initial bin. Synthetic fallback labels depend on trajectory categories around lines 323–357.

This is not `SOFA(t+24)-SOFA(t)` and independently `SOFA(t+48)-SOFA(t)`, and there is no per-horizon eligibility tied to availability in the same ICU episode. Baseline and future scores are not proven to call one pinned function.

# 14. WORK PACKAGE J — ICU-TIME LABELS

**Status: FAIL (CRITICAL).**

The target generator approximates remaining time as `(max_bin-current_bin)*6` around lines 407–411. It does not use current ICU `outtime`, does not calculate hours from the legal prediction cutoff, and does not apply `log1p`. `cohort_builder.py` also computes hospital LOS, which is a prohibited substitute. There are no non-negativity, exact-unit, transform/inverse, or current-stay identity tests.

# 15. WORK PACKAGE K — ORGAN-SUPPORT SOURCE COLLABORATION

**Status: FAIL for handoff; source verification BLOCKED.**

No source mapping package provides verified qualifying vasopressor agents, source table/fields, dose/rate units, event and active interval semantics, pause/stop/resume behavior, duplicate policy, or provenance. No pinned ventilation output provides invasive/NIV/HFNC/ordinary-oxygen categories, interval boundaries, or overlap policy. The candidate's generic deterioration/escalation concepts cannot replace Pulkit's OFF→ON state machine.

The candidate therefore does not resolve the Phase-18 real support adapter gap. Actual source validity cannot be judged without MIMIC access, but the required mapping contract is absent regardless.

# 16. WORK PACKAGE L — TRAIN-ONLY PREPROCESSING INPUTS

**Status: FAIL (CRITICAL).**

`src/preprocessing/missing_values.py:11-15` explicitly endorses fitting before train/test split “for simplicity”; `handle()` fits medians on all rows around lines 208–213. This is real preprocessing leakage in the candidate production pipeline. `src/models/xgboost_baseline.py:98-136` independently shuffles patients into 70/15/15 splits instead of consuming Vedant `split_v1`. Scaling in the XGBoost module is fitted on its local train rows, but it occurs after leaked upstream imputation.

No fit-subject inventory, split version, preprocessor version/hash, or train-only provenance is recorded. The candidate should expose unfitted canonical data to Vedant, not define another scientific split.

# 17. WORK PACKAGE M — XGBOOST

**Status: FAIL for the final project; limited mechanics reusable after redesign.**

`src/models/xgboost_baseline.py` trains 24/48/72-hour classifiers on the older targets. It consumes engineered summaries unavailable to the GRU and lacks canonical masks, true TSLO, padding, statics, and task eligibility. It selects/evaluates its own random test partition around lines 214–216, outside Vedant governance. Metrics are not stay-balanced.

`src/models/xgboost_validation_search.py` searches only the older 24-hour target by default and ranks weighted F1. Although an output enumerates 30 configurations, it is not exactly 30 configurations **per final task**, does not use the specified task metrics, and is not bound to canonical data/split/config hashes. Its merge/split fallback around lines 155–187 can align by patient or row rather than a strict canonical identity.

No tracked model artifact closes the serving gap. Metrics/figures are not models; `.gitignore` excludes model outputs. Any JSON/pickle conventions lack the integrated artifact manifest, feature contract, preprocessor hashes, calibration state, and task output domain.

# 18. WORK PACKAGE N — DATA QA / LEAKAGE

**Status: PARTIAL for generic QA; FAIL for mandatory leakage foundations.**

Range clipping and duplicate handling exist in `clean_data.py`, and some tests exercise shapes/ranges. These are useful generic checks. They do not establish cutoff safety, post-outtime exclusion, source provenance, stay isolation, train-fit provenance, or canonical schema order.

The structural leakage suite required by the Master Plan is absent. Existing “no leakage” assertions generally check that columns exist, that bin ends do not exceed 48, or that summary values remain within a broad range. They do not perturb future/pre-boundary events and compare exact outputs. All-data imputation is affirmative evidence of leakage.

# 19. WORK PACKAGE O — REPORT / EVIDENCE

**Status: PARTIAL as demo reporting; FAIL as scientific evidence.**

The repository contains figures, metrics JSON, notebooks, a cohort summary, and Parquet outputs. `outputs/reports/heart_failure_cohort_summary.json` explicitly states `data_source: DEMO`, `is_synthetic: true`, and 100 patients/admissions/stays. XGBoost metrics are for old 24/48/72-hour classification targets and include test metrics without receiver lineage.

Artifacts lack producer code commit, exact config hash, source/cohort/schema/feature/label/split versions, input hashes, and deterministic manifest. The Parquet files are tracked but could not be decoded in the current environment because `pyarrow` is absent; their naming and report metadata identify them as the demo pipeline, not accepted real evidence.

# 20. MASTER-PLAN SCIENTIFIC INVARIANT AUDIT

| Frozen invariant | Candidate result | Status |
|---|---|---|
| MIMIC-IV v2.2, adult, first ICU stay | Heart-failure admissions; no correct first-stay/adult path | FAIL |
| No global hospital-death exclusion | Not demonstrably implemented in required cohort | UNVERIFIED |
| `t=intime+24h+6h*k`, `k=0..11`, legal by `outtime-6h` | No canonical cutoff generator | FAIL |
| `(t-48h,t]`, eight disjoint right-closed bins | First 48h from first observation; left-closed bins | FAIL |
| No pre-lookback seed; pre-ICU padding | Admission-wide imputation and wrong padding | FAIL |
| Genuine observation mask and frozen TSLO | Conflated padded flags and approximate TSLO | FAIL |
| One pinned rolling `sofa_at(T)` plus six flags | Demo instantaneous proxy, no pin/flags | FAIL |
| Independent continuous 24h/48h recovery deltas | Categorical/proxy targets | FAIL |
| `log1p(outtime-t hours)` | Bin-count approximation; no `log1p` | FAIL |
| Support OFF→ON within `(t,t+24h]`, censoring | No compatible source handoff; final endpoint is Pulkit-owned | FAIL/BLOCKED |
| One canonical grid with task masks | No cutoff-indexed canonical dataset | FAIL |
| Frozen subject/year split | Competing random patient split | FAIL |
| XGB and GRU identical information | XGB-only engineered features | FAIL |
| Validation-only search/selection/calibration | Old-task search; baseline reads local test metrics | FAIL |
| Stay-balanced metrics/cluster bootstrap | Absent | FAIL |
| Family-matched explanation | No integrated TreeSHAP contract/artifact | FAIL |

# 21. PROHIBITED-PATTERN SEARCH RESULTS

| Pattern | Evidence/classification |
|---|---|
| Heart-failure-only cohort | README and `cohort_builder.py`; actual critical violation |
| Hospital LOS/discharge semantics | `cohort_builder.py`; actual wrong endpoint foundation; not shown as final model feature but used in old project |
| Random scientific split | `xgboost_baseline.py`; actual critical violation |
| Preprocessing before split | `missing_values.py`; explicit actual leakage |
| XGBoost-only extra information | `feature_engineering.py` plus XGB input path; actual parity failure |
| Forward-fill across lookback | Admission-wide fill with no cutoff boundary; actual critical risk/violation |
| Wrong bin boundaries | `grid_builder.py`; actual critical violation |
| Future/proxy SOFA target | `generate_targets.py`; actual critical violation |
| Mortality/deterioration/old endpoints | Candidate docs/targets; old-project implementation, not harmless prose |
| Hard-coded `/Users`, Downloads, Desktop paths | No tracked production hit found |
| SOFA+2 organ-support endpoint | No accepted final implementation found; no basis for support handoff |
| Notebook-only mandatory logic | Two notebooks exist, but decisive code paths are modules; failure is semantics, not notebook-only implementation |

# 22. SANSKRUTI -> VEDANT HANDOFF AUDIT

| Handoff item | Status | Concrete mismatch |
|---|---|---|
| Canonical dataset | FAIL | No `real_data_input_manifest_v1`/`CanonicalExample` |
| Subject/stay/cutoff identities | FAIL | patient/admission/time; no canonical stay/cutoff/index |
| Eight bins | FAIL | One first-48h admission grid, not per-cutoff history |
| Feature order/schema | FAIL | Candidate CSV/config does not match versioned ordered receiver schema |
| Observation masks | FAIL | `*_padded` is not genuine-observation state |
| TSLO | FAIL | Bin-end approximation; wrong no-observation representation |
| Padding | FAIL | Trailing empty bins; no correct pre-ICU structural mask |
| Statics | FAIL | No receiver-compatible statics serialization |
| Task eligibility | FAIL | No canonical task masks |
| Recovery labels | FAIL | Categorical/proxy instead of independent continuous deltas |
| ICU labels | FAIL | Bin approximation instead of `log1p(outtime-t)` |
| Provenance versions | FAIL | No source/schema/SOFA/feature/label/split binding |
| Loader parity | FAIL | Candidate cannot instantiate receiver object |
| Formula spot-checks | FAIL | Required formulas absent, so no honest spot-check is possible |

# 23. SANSKRUTI -> PULKIT HANDOFF AUDIT

| Handoff item | Status | Concrete mismatch |
|---|---|---|
| Processed schema/raw timeline | FAIL | No source-preserving, stay-keyed `processed_schema_v1` |
| Event timestamp/units/provenance | FAIL | Generic timestamp/source; mappings and conversions incomplete |
| Cutoff-aware `build_features` | FAIL | First-48h grid has no legal cutoff argument |
| Truncation equality | FAIL | No test or compatible builder API |
| Shared feature-builder reuse | FAIL | Candidate has separate training/API/dashboard logic |
| Preprocessing interface | FAIL | All-data fitted imputation; no frozen artifact metadata |
| Current SOFA | FAIL | Proxy lacks stay/cutoff identity, hashes, and score range contract |
| Six component flags | FAIL | Absent |
| Support source mapping | FAIL/BLOCKED | No verifiable infusion/ventilation mapping contract |
| XGB artifact/flattening | FAIL | No selected manifest/model/order/hash contract |

# 24. DIRECT BUILDER / SERVING EQUIVALENCE RESULT

**Result: BLOCKED at the first semantic/API boundary; no equivalence claim is possible.**

The candidate grid builder accepts admission-aligned data and builds one fixed first-48-hour grid. Pulkit expects `(raw history, stay, legal cutoff) -> CanonicalExample`, then uses the same canonical builder in serving. There is no candidate function with those inputs or outputs. A temporary adapter would need to invent ICU stay identity, generate cutoffs, discard/reshape history, redefine bin closure, recreate observation masks/TSLO/padding, choose feature order/statics, and supply provenance. That is a scientific rewrite, not an adapter, so the dry run correctly stopped rather than faking success.

# 25. XGBOOST / GRU INFORMATION-PARITY RESULT

**Result: FAIL statically; runtime unflattening not applicable.**

Candidate flattening excludes `bin_padded`, includes per-feature padded indicators/approximate TSLO, and lacks the receiver's canonical padding/statics/eligibility representation. Its XGBoost path also adds rolling engineered summaries. Vedant's `temporal_information_views()` can prove parity only from a valid `CanonicalExample`; the candidate cannot construct one. Therefore no same-sample unflatten equality can be run without rebuilding the candidate feature semantics.

# 26. SOFA / CURRENT-SOFA INTEGRATION RESULT

**Result: FAIL.** Candidate SOFA cannot back `CurrentSOFAProvider`. It is a 0–12 demo proxy rather than a pinned 0–24 score, lacks current stay/cutoff identity, source and implementation hashes, and all six observed-component flags. Its instantaneous proxy is also unsuitable for the rolling 24-hour label path. A dashboard-specific recomputation must not be introduced; the replacement must be the rebuilt shared `sofa_at(T)` implementation.

# 27. SUPPORT-MAPPING INTEGRATION RESULT

**Result: FAIL/BLOCKED.** The mapping-gap inventory is entirely open: qualifying vasoactive agents; table/field names; event, start, stop, pause, and resume timestamps; rate units/conversions; zero-rate and duplicate rules; ventilation concept pin; invasive/NIV/HFNC/oxygen categories; interval overlap and boundary rules; provenance. Real source verification is blocked without authorized data, but the candidate also lacks the required declared contract.

# 28. XGBOOST SERVING / TREESHAP COMPATIBILITY

**Result: FAIL.** No tracked candidate model is available for Pulkit's artifact loader. The training code does not emit `selected_models_calibrated_v1`, exact feature names/order/count, canonical schema/preprocessor hashes, final task identity/output domain, selection lineage, or immutable model hash. Old 24/48/72 classification metrics and feature-importance PNGs cannot substitute for a model.

TreeSHAP additionally needs the exact model object and a reversible flattened feature map with time-bin/static identities. Candidate engineered features and noncanonical flattening prevent that mapping. Current environment lacks SHAP and a working XGBoost runtime, but compatibility already fails at the contract level.

# 29. TEST QUALITY AUDIT

The candidate contains 57 statically discovered `def test_...` functions, but none executed because collection failed. Assertion review shows many tests validate the candidate's old semantics rather than the final invariants.

| Structural invariant | Candidate test/evidence | Executed? | Does it prove the invariant? |
|---|---|---:|---|
| Future-row perturbation | A feature “no future leakage” assertion bounds a mean by a past maximum | No | **No**; future values can lie in range |
| Strict 48h lookback/60h-old value | No perturbation fixture | No | **No** |
| Exact bin boundary | Grid tests exercise candidate `[0,6)` behavior | No | **No**; they prove the wrong closure |
| Padding vs missingness | Shape/padded-column tests | No | **No**; states are conflated |
| Observation mask | No genuine-vs-imputed assertion | No | **No** |
| TSLO | Candidate approximate behavior tests | No | **No**; no true event/no-observation contract |
| SOFA cutoff safety | Proxy score component tests | No | **No** |
| Same SOFA function | No identity/config call test | No | **No** |
| Recovery formula | Delta tests expect candidate categorical behavior | No | **No**; wrong output |
| ICU formula | No exact outtime-hours/log1p test | No | **No** |
| Post-outtime exclusion | No structural test | No | **No** |
| Provenance cutoff | No source trace/cutoff test | No | **No** |
| Subject/stay isolation | Demo cohort uniqueness checks | No | **No** for first ICU/cross-stay contamination |
| Train-fit provenance | No ID-set/provenance assertion | No | **No**; code says all-data fit |
| Feature schema/order lock | Column presence/shape only | No | **No** for versioned receiver order |
| XGB/GRU parity | No receiver parity test | No | **No** |
| Raw-history truncation equality | No equality test | No | **No** |

Examples of cosmetic/weak evidence include tests that only assert an output column exists, forward-fill tests that do not assert the filled value, and a “beyond 48h” test that checks `bin_end <= 48` rather than perturbing excluded input. These tests may be useful for the old demo application but cannot be counted toward scientific acceptance.

# 30. EXISTING TEST RESULTS

| Suite | Working directory / environment | Command | Result |
|---|---|---|---|
| Integrated, correctly isolated | Parent; `PYTHONPATH=src:tests:.`; bytecode redirected to `/tmp` | `python3 -m pytest tests -q -rs` | **906 passed, 2 skipped**, exit 0, 10.01s |
| Candidate | Candidate; `PYTHONPATH=.`; bytecode redirected to `/tmp` | `python3 -m pytest -q -rs` | **8 collection errors, 0 executed**, exit 1, 1.73s |
| Parent with unsafe broad discovery | Parent; nested tree discoverable | `python3 -m pytest -q -rs` | **12 collection errors**; diagnostic only, not the parent baseline |

The two integrated skips are dependency-gated Captum and SHAP/XGBoost tests. Candidate collection errors arise from Python 3.9 evaluating `str | Path` annotations in `src/utils/helpers.py` and XGBoost failing to load its OpenMP dependency. No dependency was installed to force a green result.

All 67 candidate Python files passed a syntax-only `compile()` walk under Python 3.9. This does not establish import/runtime compatibility. Black/Ruff/mypy were neither installed nor configured, so their status is **NOT RUN / UNAVAILABLE**, not PASS.

# 31. TEMPORARY INTEGRATION PROBE RESULTS

1. Parent import probe resolved `data.timestamps` and `serving.pipeline` to integrated `src/` when cwd/PYTHONPATH were controlled.
2. Candidate import probe resolved toward the nested candidate but stopped in `src/utils/helpers.py` on Python-version syntax evaluation.
3. No canonical feature probe followed because the candidate exposes no cutoff-indexed canonical builder; proceeding would require semantic invention.
4. No XGB/GRU parity probe followed because no `CanonicalExample` can be produced.
5. No serving-builder comparison followed because candidate input/output types do not meet `CanonicalHistoryInputProvider`.
6. No current-SOFA call followed because the candidate cannot construct `CurrentSOFAState`.

Temporary bytecode directories were outside both roots and removed. Importing the candidate backend created `backend/auth.db-shm` and `backend/auth.db-wal`; only those two untracked sidecars were moved to a temporary directory and removed. `git status` was clean afterward.

# 32. SECURITY / PRIVACY / RESTRICTED-DATA AUDIT

No MIMIC credentials, PhysioNet token, obvious production secret, hard-coded personal absolute path, or confirmed restricted raw extract was found. Because Parquet payloads could not be decoded, restricted-data absence inside them is not fully verified; surrounding metadata says demo/synthetic.

The repository tracks `backend/auth.db`, a SQLite database. Read-only schema/count inspection found demo-oriented users, patients, hospitals, mappings, and audit rows; source seeding identifies demo data. Committing a mutable authentication database is poor release hygiene even when synthetic. Backend code contains demo credentials and a fallback development JWT secret; values are intentionally not reproduced here. These should never ship as production defaults.

No raw patient rows are printed in this report. The audit did not access a protected final test dataset or real MIMIC data.

# 33. REPRODUCIBILITY AUDIT

**Overall: FAIL for final-project reproducibility.** A non-author can likely rerun portions of the synthetic older project after assembling a compatible Python/OpenMP/Parquet environment, but cannot reconstruct the final project's extraction, cohort, timeline, SOFA, features, or labels.

Deficiencies: no exact environment lock; no MIMIC v2.2 manifest/schema verification; no SQL or complete real extraction command; real pipeline `NotImplementedError`; demo fallback; no artifact lineage manifest; no code/config/input hashes embedded in outputs; no canonical schema/feature order contract; no pinned SOFA concept; and no receiver acceptance entrypoint. `run_sofa_pipeline.py` forces demo mode; `run_pipeline.py` calls demo extraction unconditionally and has undefined/incomplete real-path dependencies.

# 34. CRITICAL FINDINGS

| ID | Finding | Evidence | Consequence |
|---|---|---|---|
| C1 | Wrong research cohort | `cohort_builder.py:104-193` heart-failure admissions | Cannot represent all-adult first ICU cohort |
| C2 | Wrong time semantics | `grid_builder.py` first 48h, first observation, `[start,end)` | Cannot adapt to per-cutoff `(t-48,t]` without rewrite |
| C3 | Preprocessing leakage | `missing_values.py:11-15,208-213` all-data fit | Validation/test information affects features |
| C4 | Pre-lookback carry risk | admission-wide forward fill `:106-116` | Values at/before `t-48` can seed history |
| C5 | Wrong SOFA | `sofa.py` demo 3-component proxy | Recovery labels/current SOFA scientifically invalid |
| C6 | Wrong recovery targets | `generate_targets.py:279-435` categorical/proxy | Does not estimate final continuous deltas |
| C7 | Wrong ICU-time target | bin-count proxy, no `outtime`/`log1p` | Different clinical endpoint |
| C8 | Wrong split | `xgboost_baseline.py:98-136` random 70/15/15 | Violates frozen subject/year split |
| C9 | Information advantage | engineered XGB features, no canonical parity | XGB/GRU comparison unfair |
| C10 | Development test use | XGB baseline evaluates local test during training | Competing methodology; protected integrated test was not shown accessed |
| C11 | Stale project scope | README/config/targets 24/48/72 deterioration | Repository targets older specification |

# 35. MAJOR FINDINGS

- M1: real extraction is incomplete and not reproducibly bound to MIMIC-IV v2.2.
- M2: no `real_data_input_manifest_v1`, `processed_schema_v1`, `feature_schema_v1`, or `feature_provenance_v1` equivalent.
- M3: no verified infusion/ventilation mapping to close Pulkit's support-source gap.
- M4: no receiver-compatible canonical builder, current-SOFA provider, or serving/training equivalence path.
- M5: no integrated model artifact, selected-model manifest, feature map, hash, or TreeSHAP-ready contract.
- M6: scientific evidence is synthetic/old-task and lacks lineage.
- M7: mandatory structural leakage/formula/parity tests are absent.
- M8: candidate suite cannot collect in the current integrated environment.
- M9: a separate backend/frontend duplicates Pulkit ownership and includes simulated predictions presented as a real model path (`backend/app/services/prediction_service.py:20-69`).

# 36. MODERATE / MINOR FINDINGS

**Moderate:** Python version incompatibility; lower-bound-only dependencies; no packaging/static-check configuration; silent demo fallback; tracked mutable auth database; generic rather than executable data dictionary; unsafe row-alignment fallback in validation search; duplicate timestamp resolution without stable provenance.

**Minor:** mixed `config/` naming relative to integrated `configs/`; historical names and documentation increase confusion; style/static tools are absent. These are not decision drivers.

# 37. FULL COMPLIANCE MATRIX

| ID | Requirement / authority | Candidate implementation & evidence | Status / severity | Receiver consequence | Required fix |
|---|---|---|---|---|---|
| A | WP-A: MIMIC v2.2 extraction | Partial loader; hard-coded IDs; real path incomplete | FAIL / MAJOR | Manifest cannot validate sources | Rebuild verified extraction + manifest |
| B | WP-B: adult first ICU | HF admission cohort; nondeterministic ICU dedupe | FAIL / CRITICAL | Wrong population/identities | Rebuild from `icustays` contract after schema verification |
| C | WP-C: canonical timeline | Admission-keyed 1h data, provenance reduced | FAIL / CRITICAL | History provider cannot consume | Emit stay-keyed source-preserving schema |
| D | WP-D: feature provenance | Generic CSV only | FAIL / MAJOR | Feature/schema audit impossible | Build versioned provenance dictionary |
| E | WP-E: pinned pure `sofa_at(T)` | Demo proxy | FAIL / CRITICAL | Wrong labels/current state | Implement one pinned cutoff-safe function |
| F | WP-F: SOFA dependencies | No executable component lineage | FAIL / MAJOR | Six components unverifiable | Record every source/conversion/pin |
| G | WP-G: 8×6h grid | First-48h fixed grid | FAIL / CRITICAL | Wrong tensor semantics | Build each legal cutoff from receiver spec |
| H | WP-H: strict lookback | Global fill; no perturbation proof | FAIL / CRITICAL | Temporal leakage | Enforce `(t-48,t]` before any state |
| I | WP-I: recovery labels | Categorical/proxy deltas | FAIL / CRITICAL | Wrong outputs | Independent continuous deltas using same SOFA |
| J | WP-J: ICU time | Bin-count proxy | FAIL / CRITICAL | Wrong endpoint | `log1p((outtime-t).hours)` |
| K | WP-K: support collaboration | No verified mapping | FAIL/BLOCKED / MAJOR | Pulkit real adapter remains blocked | Provide verified raw mappings only |
| L | WP-L: train-only inputs | All-data imputation + random split | FAIL / CRITICAL | Leakage/governance violation | Expose unfitted data; use `split_v1` |
| M | WP-M: XGBoost | Old classifiers/engineered inputs | FAIL / CRITICAL | No fair candidate or serving model | Train from shared canonical flatten only |
| N | WP-N: QA/leakage | Generic ranges; structural suite absent | PARTIAL/FAIL / MAJOR | G1 cannot pass | Add mandated adversarial tests |
| O | WP-O: reports/evidence | Synthetic reports without lineage | PARTIAL/FAIL / MAJOR | No acceptance evidence | Script versioned evidence from accepted bundle |
| MP1 | Canonical cutoffs / Master Plan | None | FAIL / CRITICAL | No row universe | Reuse Vedant generator |
| MP2 | Right-closed 48h history | Opposite/fixed semantics | FAIL / CRITICAL | Values assigned differently | Implement exact boundaries |
| MP3 | Padding ≠ missingness | Conflated | FAIL / CRITICAL | Model sees wrong state | Separate masks |
| MP4 | Genuine obs + TSLO | Approximate | FAIL / CRITICAL | Information semantics wrong | Use event times and frozen sentinel |
| MP5 | Subject/year split | Random split | FAIL / CRITICAL | Invalid evaluation | Consume frozen split |
| MP6 | Stay-balanced estimation | Absent | FAIL / MAJOR | Metrics incomparable | Hand results to Vedant evaluation |
| MP7 | Validation-only selection | Local test evaluated | FAIL / CRITICAL | Governance contamination | Remove selection/test logic from Sanskruti path |
| V1 | `CanonicalExample` fields | Not emitted | FAIL / MAJOR | Loader failure | Target exact dataclass/schema |
| V2 | XGB/GRU parity | Extra engineered features | FAIL / CRITICAL | Unfair family comparison | Use `temporal_information_views()` |
| V3 | Manifest/hashes | Absent | FAIL / MAJOR | Real adapter fail-closed | Produce honest v1 manifest |
| P1 | Pulkit history seam | No compatible timeline/builder | FAIL / MAJOR | Serving path remains injected only | Implement shared builder contract |
| P2 | Current SOFA state | Proxy/no flags/hashes | FAIL / CRITICAL | Recovery output invalid | Back exact provider from shared SOFA |
| P3 | Artifact loader | No compatible model manifest | FAIL / MAJOR | Cannot serve candidate | Emit governed selected artifact after G3 |
| T1 | Structural tests | Weak/absent, none executable | FAIL / MAJOR | Claims unproved | Add exact perturbation/formula tests |
| R1 | Rebuild by another member | Real route incomplete | FAIL / MAJOR | Not reproducible | Configured entrypoint + lineage |
| S1 | No restricted data/secrets | No confirmed real data; dev auth defaults | PARTIAL / MODERATE | Release hygiene concern | Remove DB/default credentials from production package |

# 38. FILE-BY-FILE MERGE MAP

| Candidate file/module | Purpose | Existing target/interface | Action |
|---|---|---|---|
| `README.md` | Old project description | Final docs/context | DISCARD / DO NOT USE |
| `config/config.yaml` | Old horizons/grid/models | Frozen configs | REBUILD |
| `config/clinical_ranges.yaml` | Generic bounds | Canonical QA config | ADAPT after clinical/source verification |
| `config/features.yaml` | Candidate feature list | `feature_schema_v1` | REBUILD |
| `src/data/load_data.py` | Demo/partial MIMIC loading | Real extraction/manifest | REBUILD |
| `src/data/cohort_builder.py` | HF cohort | First-ICU cohort artifact | REBUILD |
| `src/data/extract_features.py` | Item-ID feature extraction | Canonical raw timeline | REBUILD; individual mappings UNVERIFIED |
| `src/preprocessing/time_alignment.py` | 1h admission alignment | Pulkit timeline/canonical builder | DISCARD / DO NOT USE |
| `src/preprocessing/clean_data.py` | Generic cleaning | QA layer | ADAPT only after provenance-safe ordering/rules |
| `src/preprocessing/missing_values.py` | Imputation | Vedant train-only preprocessor | DISCARD / DO NOT USE |
| `src/preprocessing/feature_engineering.py` | Extra rolling summaries | Canonical parity views | DISCARD / DO NOT USE |
| `src/preprocessing/grid_builder.py` | First-48h grid | `CanonicalExample` builder | REBUILD |
| `src/preprocessing/sofa.py` | Demo proxy | shared `sofa_at(T)`/CurrentSOFA | DISCARD / DO NOT USE |
| `src/targets/generate_targets.py` | Old endpoints | Recovery/ICU task targets | REBUILD |
| `src/models/xgboost_baseline.py` | XGB training mechanics | Vedant task trainer/registry | REUSE AFTER FIX only for low-level wrapper ideas |
| `src/models/xgboost_validation_search.py` | Old search | Vedant governed 30/task search | DISCARD / DO NOT USE |
| `run_pipeline.py` | Demo orchestrator | Configured real-data entrypoint | REBUILD |
| `run_sofa_pipeline.py` | Forced demo run | Shared SOFA evidence runner | DISCARD / DO NOT USE |
| `backend/` | Separate API/auth/predictions | Pulkit serving/API | DISCARD / DO NOT USE |
| `frontend/` | Separate dashboard | Pulkit replay/dashboard | KEEP SEPARATE as old demo; do not integrate |
| `tests/` | Old-project tests | Integrated semantic tests | KEEP SEPARATE; extract generic fixture ideas only |
| `notebooks/` | Exploration | None | KEEP SEPARATE; never authoritative |
| `data/demo/` | Synthetic demo | Non-scientific fixtures | REUSE AFTER FIX in an explicitly isolated demo namespace |
| `data/interim`, `data/processed` | Generated old pipeline outputs | Real bundle | DISCARD / DO NOT USE scientifically |
| `outputs/` | Demo metrics/figures | Versioned evidence | DISCARD / DO NOT USE scientifically |
| `src/utils/logger.py` | Logging utility | Integrated logging | SAFE TO REUSE conceptually; copying adds little value |
| `src/utils/helpers.py` | Generic helpers | Integrated utilities | REUSE AFTER FIX for Python compatibility, if needed |

# 39. REUSABLE COMPONENTS

**SAFE TO REUSE:** no scientific production component as-is. Generic logging patterns are harmless but unnecessary.

**REUSE AFTER FIX:** range-validation ideas from `clean_data.py`; unit-conversion scaffolding from `extract_features.py` only after real schema verification and source-retaining provenance; isolated synthetic-data ideas; low-level XGBoost `save_model`/fit wrapper patterns after replacing inputs, tasks, split, metrics, search, and artifact metadata.

**DO NOT REUSE:** cohort selection, fixed grid, missing-value pipeline, engineered feature path, demo SOFA, target generator, random split, old validation search, separate API/prediction service, dashboard serving logic, and old scientific outputs.

**UNVERIFIED:** every MIMIC item/concept/table mapping and every candidate Parquet payload. Reuse requires authoritative schema/concept verification, not code-name similarity.

# 40. REQUIRED PATCHES OR REBUILD SCOPE

This is a rebuild against existing interfaces, not another isolated architecture:

1. Verify MIMIC-IV v2.2 tables/fields/event timestamps and create a source-retaining extraction manifest without silent demo fallback.
2. Build the adult, chronologically first ICU-stay cohort, legal-cutoff structural index, and deterministic flow evidence.
3. Emit a versioned stay-keyed raw timeline and full feature/support provenance dictionary.
4. Implement one pure rolling-24h `sofa_at(T)` from an immutable verified MIMIC-Code commit, with all six observed flags and Pulkit-compatible ventilation source semantics.
5. Reuse Vedant's canonical timestamp generator; build exact `(t-48,t]` bins with genuine observation masks, frozen TSLO, and structural padding.
6. Generate independent continuous 24h/48h recovery targets and exact `log1p` current-ICU-time target with explicit eligibility.
7. Supply verified vasoactive/ventilation raw mappings to Pulkit; do not implement a second endpoint.
8. Emit `real_data_input_manifest_v1` and every required sidecar/hash/version; keep preprocessing unfitted or prove train-only fit IDs.
9. Implement XGBoost as a thin consumer of Vedant's canonical flattened view; delegate split, 30-per-task search, metrics, selection, calibration, and test governance to Vedant.
10. Add all adversarial structural tests, then prove direct/training/serving builder equality and current-SOFA compatibility.
11. Only after G1/G2 acceptance, generate governed XGBoost artifacts compatible with Pulkit's loader and TreeSHAP map.

# 41. G0 READINESS

**FAIL.** The authoritative project contracts are frozen, but the candidate does not implement them and instead freezes conflicting old-project semantics. Candidate code cannot be admitted as the Sanskruti implementation at G0. The rebuild can start immediately against the already-versioned receiver contracts; any necessary source mapping uncertainty must remain explicitly unresolved until verified.

# 42. G1 READINESS

**FAIL.** Required real cohort, split compatibility, canonical features, masks, SOFA, labels, provenance, and structural QA are absent or incorrect. Phase-18 explicitly says the real handoff remains missing/blocked and that Phase 19 is not thereby complete. Candidate synthetic artifacts do not change that status.

# 43. G2 INTEGRATION READINESS

**FAIL.** Existing Vedant/Pulkit paths cannot consume the candidate without scientific reinterpretation. No accepted canonical bundle, shared builder, current-SOFA provider, support mapping, XGB artifact, or parity/equivalence proof exists. G3–G5 were not assessed or declared.

# 44. WHAT WAS NOT POSSIBLE TO VERIFY

- Actual MIMIC table/column/item-ID correctness: **BLOCKED — REAL MIMIC SCHEMA VERIFICATION REQUIRED**.
- Candidate Parquet internal schemas/rows: blocked by absent `pyarrow`; metadata identifies demo outputs, but payloads were not opened.
- Candidate test outcomes under its claimed Python 3.10+/OpenMP environment: blocked; current suite fails collection.
- External existence/correctness of a pinned SOFA commit: not applicable because no immutable candidate pin was found.
- Absence of restricted data inside binary Parquet/SQLite at row level: not exhaustively proven; safe metadata/schema-level inspection found demo framing.
- Real cohort counts, label availability, missingness, and support prevalence: no accepted real artifacts and deliberately not invented.
- Final XGBoost numerical compatibility/performance: no compatible model artifact; performance was not used as an acceptance proxy.
- Protected final-test history outside repository evidence: no evidence of integrated protected-test access was found. Candidate's own random “test” metrics are a governance violation in its methodology but are not asserted to be the protected final test.

# 45. EXACT COMMANDS EXECUTED

Representative exact audit commands (all read-only except the final report and cleanup of audit-created temporary sidecars):

```bash
pwd
find . -name .git -type d -prune -print
git -C Heart-Failure-Recovery-and-Deterioration-Trajectory-Prediction status --short --branch
git -C Heart-Failure-Recovery-and-Deterioration-Trajectory-Prediction rev-parse --show-toplevel
git -C Heart-Failure-Recovery-and-Deterioration-Trajectory-Prediction rev-parse HEAD
git -C Heart-Failure-Recovery-and-Deterioration-Trajectory-Prediction log --oneline -5
git -C Heart-Failure-Recovery-and-Deterioration-Trajectory-Prediction diff --stat
git -C Heart-Failure-Recovery-and-Deterioration-Trajectory-Prediction ls-files
find Heart-Failure-Recovery-and-Deterioration-Trajectory-Prediction -path '*/.git' -prune -o -type f -print
shasum -a 256 /Users/Home/Downloads/Personalized_Patient_Recovery_Trajectory_FINAL_Implementation_Master_Plan.docx
shasum -a 256 '/Users/Home/Downloads/Personalized_Patient_Recovery_Trajectory_FINAL_Work_Division_3_Members (1).docx'
shasum -a 256 docs/CODEX_PROJECT_CONTEXT_V1.md
textutil -convert txt -stdout /Users/Home/Downloads/Personalized_Patient_Recovery_Trajectory_FINAL_Implementation_Master_Plan.docx
textutil -convert txt -stdout '/Users/Home/Downloads/Personalized_Patient_Recovery_Trajectory_FINAL_Work_Division_3_Members (1).docx'
rg -n 'real_data_input_manifest|CanonicalExample|CurrentSOFA|temporal_information_views|split_v1|timestamp_spec' .
PYTHONPYCACHEPREFIX=/tmp/healthcare-parent-pyc PYTHONPATH=src:tests:. python3 -m pytest tests -q -rs
cd Heart-Failure-Recovery-and-Deterioration-Trajectory-Prediction
PYTHONPYCACHEPREFIX=/tmp/sanskruti-pyc PYTHONPATH=. python3 -m pytest -q -rs
python3 -c 'import sys; print(sys.path)'
python3 --version
python3 -c 'import pandas, numpy, sklearn, yaml, fastapi, pydantic; print(...)'
python3 -c 'import importlib.util as u; print({x: bool(u.find_spec(x)) for x in [...])'
git grep -n -E 'train_test_split|dischtime|length_of_stay|forward.fill|MIMIC|SOFA|24h|48h|72h'
git grep -n -E '/Users/|Downloads/|Desktop/|C:\\Users\\'
file data/**/*.parquet backend/auth.db
sqlite3 backend/auth.db '.tables'
sqlite3 backend/auth.db 'SELECT name FROM sqlite_master WHERE type="table" ORDER BY name;'
```

Additional inspections used `sed -n`, `nl -ba`, `rg`, `find`, `stat`, `file`, and `shasum` on the exact files cited in this report. Python source syntax was checked with an in-memory `compile()` walk rather than `compileall` writing into the repository. No package installation, training, refit, real-label generation, merge, or final-test access occurred.

# 46. EXACT TEST COUNTS / FAILURES / SKIPS

- Integrated isolated suite: **906 passed, 0 failed, 2 skipped, 0 xfailed; exit 0; 10.01 seconds**.
- Candidate suite: **0 tests executed; 8 collection errors; exit 1; 1.73 seconds**. Statically, 57 test functions exist.
- Unsafe parent-wide discovery diagnostic: **12 collection errors** caused by nested-suite discovery/import collision; it is not an integrated baseline result.
- Candidate syntax audit: **67/67 Python files compiled from source text**; runtime imports are not implied.
- Static tools: Black **not run/unavailable**, Ruff **not run/unavailable**, mypy **not run/unavailable**.
- Parent skip causes: dependency-gated Captum and SHAP/XGBoost coverage.
- Candidate collection causes: Python 3.9 union-type annotation evaluation and XGBoost/OpenMP load failure.

# 47. FINAL USE-VS-REBUILD RECOMMENDATION

**Recommendation D — REBUILD SANSKRUTI TRACK.**

**Would I merge/use Sanskruti's implementation in the integrated project today? NO.** It implements a materially different research question and violates multiple frozen invariants before any naming or packaging issue is considered.

**Should we build Sanskruti's track ourselves instead? YES — REBUILD IS SAFER.** Salvaging the core would require replacing cohort, temporal grid, imputation, SOFA, labels, split, XGBoost inputs/governance, evidence, and serving handoffs. Retaining that shell would create a high risk of hidden old semantics. Rebuilding directly to the already-tested Vedant/Pulkit contracts has a smaller scientific surface and clearer acceptance path.

Confidence remains **HIGH**. Lack of real MIMIC access limits verification of source mappings and real counts, but it does not affect the conclusion: the hard-coded algorithms/configs are incompatible on their face.

# 48. NEXT ACTION

Do not merge, copy, or wire the candidate repository into either `src/` tree. Preserve it as a read-only historical/demo reference at commit `dc95164b808becd6e822bccd28c0135303753e40`.

Open a new Sanskruti rebuild workstream whose first acceptance milestone is not model training: it is a minimal, verified `real_data_input_manifest_v1` bundle containing the all-adult first-ICU cohort/structural index, source-preserving timeline, ordered canonical feature schema, pinned cutoff-safe SOFA with six flags, exact recovery/ICU labels, and leakage evidence. Review that bundle jointly against `src/data/real_adapter.py`, `CanonicalHistoryInputProvider`, and `CurrentSOFAProvider`. Only after G1 acceptance should XGBoost training/search and Pulkit serving artifact integration begin.
