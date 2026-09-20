# Personalized Patient Recovery Trajectory — Project Scope v2

| Field | Value |
|---|---|
| Scientific version | `PROJECT_SCOPE_V2` / `2.0-draft` |
| Status | **DRAFT COMPLETE — TEAM FREEZE REQUIRED** |
| Approval | **REVIEW REQUIRED**; no member signoff is asserted |
| Effective scope | Final academic evaluation on an authorized, versioned synthetic cardiac dataset |
| Amends | Final Implementation Master Plan v1.0, Final Work Division v1.0, and `CODEX_PROJECT_CONTEXT_V1.md` only where explicitly stated |
| Change reason | MIMIC authorization was requested but was not obtained; the team therefore adopted a fully synthetic cardiac dataset for the final academic implementation |
| Historical documents | Preserved; this amendment does not rewrite project history |
| Downstream impact | Data/cohort/provenance, split policy, source mappings, acceptance, training artifacts, and scientific evidence must be regenerated or migrated as identified in the impact matrix |

The project title is retained provisionally to avoid inventing a new title. **REVIEW REQUIRED — FINAL PROJECT TITLE.** The active scope statement is “retrospective sequential forecasting over synthetic ICU-like cardiac episodes.”

## 1. Authority and interpretation

This document is the active scope amendment once reviewed. Precedence for the rebuild is: explicit later team decision; this versioned amendment; unchanged frozen Vedant/Pulkit contracts; unchanged portions of the original Final Implementation Master Plan and Work Division; historical reviews; then reference implementations.

Only two foundational project decisions are authorized directly: MIMIC-IV is replaced by synthetic project data, and the general ICU population is replaced by a cardiac/heart-disease domain. No evidence authorizes replacing the repeated-cutoff clock, history, three tasks, model families, statistical governance, serving pipeline, or replay mode.

## 2. Final data source and data-scope identities

`FINAL PROJECT DATA SOURCE = SYNTHETIC`.

- `AUTHORIZED_FINAL_SYNTHETIC_DATA` means the future accepted scientific dataset bound to generator, schema, config, seed/provenance, dataset, cohort, and split identities/hashes.
- `ENGINEERING_SYNTHETIC_FIXTURE` means a test/smoke/demo input. A fixture cannot become final scientific data through naming or reuse.
- No real MIMIC patient data are used; no MIMIC-derived experimental result is claimed.
- PhysioNet credentials, MIMIC SQL/item IDs/table names/cohort counts, anchor-era splits, MIMIC-specific SOFA provenance, and MIMIC-specific ventilation provenance are superseded for the final run.
- Generator implementation, dataset creation, and acceptance are later-phase work and are not performed in Phase 1.

Required future final-dataset lineage: generator version and code hash; schema version; exact config hash; seed/provenance policy; dataset hash; population/cohort specification; immutable subject split and hash; environment and producer identity.

## 3. Population and episode

The population domain is **adult cardiac/heart-disease**. It is not unrestricted ICU and is not silently narrowed to heart failure only. Exact subtype taxonomy and distribution are deferred to the Phase-2 generator specification; no diagnostic taxonomy is invented here.

The prediction episode is one synthetic ICU-like current critical-care stay with explicit `subject_id`, `stay_id`, `intime`, and `outtime`. First/last measurements may not substitute for episode boundaries. The core retains one canonical eligible episode per subject. If later generation permits multiple episodes, select the chronologically first eligible episode unless a later reviewed scope version changes that rule.

## 4. Event time and prediction clock

Forecasting is retrospective and event-time based. Inputs at cutoff `t` may use only information with `event_time <= t`, subject to lookback rules. Documentation/ingestion latency is not modeled.

The existing `timestamp_spec_v1` is retained:

```text
t_k = intime + 24h + 6h*k, k = 0..11
retain exactly when t_k <= outtime - 6h
maximum 12 prediction rows per stay
```

Sanskruti's later builder must consume Vedant's generator rather than implement a competing clock.

## 5. Canonical history and missingness

History is `(t-48h, t]`, represented by eight nonoverlapping, left-open/right-closed six-hour bins:

```text
(t-48,t-42], (t-42,t-36], (t-36,t-30], (t-30,t-24],
(t-24,t-18], (t-18,t-12], (t-12,t-6], (t-6,t]
```

Nothing at or before `t-48h` may enter or seed state. Nominal pre-`intime` history is structural padding, never ward history, missing clinical data, or fabricated observations. Structural padding, within-episode missingness, genuine-observation mask, and TSLO are distinct. Imputation, encoding, scaling, target statistics, and class weights are fit on training subjects only.

The common row retains subject/stay/cutoff identity; 8×6h history; statics; observation masks; TSLO; padding mask; per-task eligibility; labels where available; and schema/provenance identities. Exact synthetic dynamic/static features are deferred, but each future feature must declare semantic name, unit, generator source, event-time semantics, valid range, aggregation, leakage status, and canonical order.

## 6. Severity and recovery

SOFA-based recovery is retained. MIMIC-Code provenance is superseded, but the future synthetic dataset must support a versioned six-component, 0–24, cutoff-safe SOFA-style calculation. One `sofa_at(T)` implementation is used at `t`, `t+24h`, and `t+48h`, with six component-observed flags.

If a future generator cannot support this intended definition, a custom score must not be called SOFA. Renaming or changing the measure requires a new scientific scope version and downstream contract impact review; it is not authorized here.

```text
DeltaSOFA24 = SOFA(t+24h) - SOFA(t)
DeltaSOFA48 = SOFA(t+48h) - SOFA(t)
```

The outputs are continuous, independent, and never chained. Improving/stable/deteriorating classes and a 72-hour trajectory class are out of scope. Display may clip `SOFA(t)+delta` to `[0,24]`; raw delta metrics remain unclipped.

## 7. Remaining episode time

The target is time until the current synthetic ICU-like episode ends:

```text
R(t) = (outtime - t) in hours
model target = log1p((outtime - t) hours)
```

It is not hospital discharge ETA, survival time, or recovery time and may not be approximated by remaining bins or last measurement time.

## 8. New organ-support initiation

The third task remains 24-hour new organ-support initiation. A positive is an eligible OFF→ON transition for a qualifying vasopressor or invasive mechanical ventilation in `(t, t+24h]`. Support already ON at `t` is continuation, not initiation. Partial follow-up without an earlier positive is censored; a positive before episode end remains positive.

Future synthetic event records must expose enough versioned category, start/stop, active interval, rate/unit where relevant, and invasive/non-invasive semantics to feed Pulkit's existing state machines. Phase 1 freezes the interface, not event distributions.

## 9. Split and protected test

The MIMIC anchor-era split is **SUPERSEDED**. Synthetic fake years are prohibited. The replacement policy remains an **UNLOCKED SCIENTIFIC PARAMETER — SYNTHETIC SUBJECT SPLIT POLICY**, to be resolved before Phase 10.

Non-negotiable principles remain: subject-level separation; deterministic assignment; immutable artifact/hash; train/validation/final-test partitions; validation-only development and selection; final test sealed until a G3-equivalent freeze; and no duplicate or clone-derived trajectory crossing partitions. Percentages, seed, stratification, and generator-batch holdout are deliberately unset.

## 10. Models, parity, search, and evaluation

Core candidates remain naive, XGBoost, and GRU. LSTM remains one fixed sensitivity run per task. TFT/MTL/Transformers and other families are outside the core.

XGBoost and GRU receive identical canonical information; XGBoost only flattens it. Generator latent variables, trajectory class, future summaries, full-episode statistics, and privileged engineered features are forbidden.

Exactly 30 XGBoost and 30 GRU validation configurations per task are retained. Recovery selects lowest stay-balanced +24h MAE; +48h is secondary. Remaining time selects lowest stay-balanced weighted median absolute error in hours. Support selects highest stay-balanced AUPRC on raw probabilities. Only the selected support classifier receives validation-fitted isotonic calibration and a validation-selected predefined-F1 threshold. Performance is not a correctness gate.

Repeated prediction rows retain stay-balanced estimators and `stay_id` cluster bootstrap, carrying all eligible rows per sampled stay copy and reporting example and unique-stay counts.

## 11. Serving, explanations, and product mode

`PredictionPipeline` remains the serving authority: truncate raw history at `t`, call the same canonical builder and frozen training preprocessor, route selected task models, calibrate only support, and validate versioned outputs/artifacts. XGBoost routes to TreeSHAP and GRU to Integrated Gradients.

The product remains **RETROSPECTIVE SEQUENTIAL REPLAY — NOT REAL-TIME CLINICAL PREDICTION**. A synthetic source does not authorize real-time monitoring, cached future answers, or alternate API/dashboard feature pipelines. `prediction_schema_v1` remains conceptually unchanged because tasks remain unchanged.

## 12. Scientific claims and nonclaims

Allowed framing: methodological prototype; software/research framework; evaluated on versioned synthetic cardiac trajectories; retrospective sequential forecasting demonstration; integration/reproducibility demonstration.

Prohibited claims: external clinical validation; bedside accuracy or safety; deployment readiness; patient-specific clinical utility; real-world cardiac prognosis accuracy; causal/treatment recommendation; MIMIC validation; real-time clinical prediction. Final reporting must state that MIMIC authorization was not obtained and no real-patient validation was performed.

## 13. Historical Sanskruti repository

`Heart-Failure-Recovery-and-Deterioration-Trajectory-Prediction` is a **HISTORICAL / REFERENCE IMPLEMENTATION**, not the authoritative final implementation. It is not merged, registered, or promoted by this phase. Later phases may salvage reviewed implementation ideas but may not inherit its first-48h grid, trajectory targets, proxy SOFA, all-data preprocessing, or results as scope.

## 14. Contract status and future migration

Retained v1 semantics: timestamp grid; three task equations; tensor information concepts; model families; evaluation; explanation routing; PredictionPipeline; prediction response; replay; test protection.

Future successors/adapters are expected for: project/cohort/generator/schema provenance; data acceptance; synthetic subject split; MIMIC-bound metadata in canonical/registry contracts; and synthetic support/SOFA source mappings. Phase 1 changes no runtime contract.

See `project_scope_v2_delta.csv` for requirement classification and `project_scope_v2_impact.md` for downstream action. Machine-readable authority is `configs/governance/project_scope_v2.json`.

## 15. Unresolved later-phase parameters

Deferred: exact cardiac subtype taxonomy/mix; cohort size; latent-state design; missingness mechanism; support frequency; episode-duration distribution; dynamic/static feature inventory; split percentages/seed/stratification/batch policy; generator seed policy; standard-SOFA synthetic input details; and final environment/package manager. Their absence does not authorize defaults.

## 16. Approval and review

Document presence is not approval. Required human review is recorded in `docs/sanskruti/PHASE1_SCOPE_V2_REVIEW.md`. Until genuine signoff is recorded, status remains **DRAFT COMPLETE — TEAM FREEZE REQUIRED**.
