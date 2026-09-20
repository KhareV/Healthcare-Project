# Synthetic Standard SOFA Provenance v1

## Status and scientific limitation

This document describes the Phase-6 engineering implementation of a standard six-component SOFA-style score over synthetic canonical observations. The threshold source is authoritative, but this project-specific specification remains `DRAFT_REVIEW_REQUIRED_NOT_AUTHORIZED_FOR_FINAL_PRODUCTION`: the `(T-24h,T]` endpoint convention and exact-time PaO2/FiO2 pairing need team review, and Phase 9 has not supplied the final synthetic invasive-ventilation and vasoactive-dose contract. The engine is usable with explicit engineering evidence and fails closed when those pipeline dependencies are unavailable.

Implementing standard score logic does not clinically validate the generated trajectories. Inputs and resulting scores are synthetic benchmark quantities. No real-patient prognostic validity is claimed.

## Authoritative scoring reference

The component table comes from Vincent JL, Moreno R, Takala J, et al., “The SOFA (Sepsis-related Organ Failure Assessment) score to describe organ dysfunction/failure,” *Intensive Care Medicine* 1996;22:707–710, DOI `10.1007/BF01709751`, Table 3. The reference was verified on 2026-09-20 using the [ESICM-hosted copy](https://www.esicm.org/wp-content/uploads/2018/03/esicmworking5-2.pdf). The machine-readable transcription is `configs/synthetic/sofa_spec_v1.json`; its SHA-256 is recorded in the Phase-6 lineage artifact and is always computed from the file.

## Score contract

There are exactly six ordered components: respiratory, coagulation, liver, cardiovascular, central nervous system, and renal. Each contributes an integer from 0 through 4; the total is the exact sum and must be 0 through 24. No clipping repairs invalid component output.

One function, `data.synthetic.sofa.sofa_at`, evaluates every time point. Baseline `t`, `t+24h`, and `t+48h` change only the cutoff argument. There is no horizon flag and no baseline/future/dashboard/label variant.

The draft project rolling window is `(T-24h,T]`: the left edge is excluded and the cutoff is included. All candidates are scored individually and the maximum score per component is retained. Same-score evidence is retained in deterministic reference order. This boundary is explicitly `REVIEW_REQUIRED_SOFA_ROLLING_WINDOW_BOUNDARY`, not falsely frozen.

## Missingness and observability

A clinically unobserved component contributes zero and has `observed=false`. A genuinely observed normal component has zero and `observed=true`. There is no 50-percent, majority-observed, exclusion, imputation, carry-forward, population estimate, or model-based estimate rule.

Pipeline unavailability is different. A missing/unknown support provider raises `SOFADependencyUnavailable` with `BLOCKED — REQUIRED SOFA DEPENDENCY UNAVAILABLE`; it is never converted to six missing clinical components. Known inactive support is an explicit state from a working provider.

## Canonical dependency map

| Component | Canonical facts | Unit | Phase-5 provenance | Event-time and selection semantics | Observed criterion | Support dependency |
|---|---|---|---|---|---|---|
| Respiratory | `pao2`, `fio2` | `mmHg`, fraction | `PROV-RAW-PAO2-V1`, `PROV-RAW-FIO2-V1` | Exact-time pairs available in `(T-24h,T]`; ratio scored; worst score selected | At least one valid pair with known invasive state at the pair time | Pulkit invasive-ventilation state, final source deferred to Phase 9 |
| Coagulation | `platelet_count` | `10^9/L` | `PROV-RAW-PLATELET-V1` | Every point in window; lowest values naturally yield highest score | At least one valid point | None |
| Liver | `bilirubin_total` | `mg/dL` | `PROV-RAW-BILIRUBIN-V1` | Every point in window; highest rule score retained | At least one valid point | None |
| Cardiovascular | `mean_arterial_pressure`; dose segments | `mmHg`; `ug/kg/min` | `PROV-RAW-MAP-V1`; Phase-9 dose provenance pending | MAP and qualifying >=60-minute dose exposure portions in window; worst retained | At least one MAP or qualifying exposure | Phase-9 rate/unit/agent contract required; Pulkit binary state alone is insufficient |
| Central nervous system | `glasgow_coma_scale` | integer score | `PROV-RAW-GCS-V1` | Every point in window; lowest GCS yields highest score | At least one valid integer GCS | None; no invented sedation adjustment |
| Renal | `creatinine`; `urine_output_volume` | `mg/dL`; `mL` | `PROV-RAW-CREATININE-V1`, `PROV-RAW-URINE-V1` | Worst creatinine score versus complete contiguous 24-hour urine total | Creatinine exists or urine intervals exactly cover full window | None |

## Respiratory details

PaO2 and FiO2 must be exact-time pairs in this draft contract; unmatched events do not form a ratio. Every pair also requires an explicit invasive-ventilation ACTIVE/INACTIVE state at that instant. Unknown is a dependency failure. Ratios below 400 and 300 score 1 and 2. Ratios below 200 and 100 score 3 and 4 only with invasive respiratory support. Invalid/nonpositive or out-of-domain canonical values and wrong units fail closed.

The exact-time pairing decision is conservative and avoids an invented FiO2 carry-forward interval, but the Phase-3 generator currently emits PaO2 and FiO2 asynchronously. Therefore it is explicitly review-required before final production; it is not presented as a frozen scientific decision.

## Cardiovascular details

MAP below 70 mmHg scores 1. Qualifying dobutamine, dopamine, epinephrine, and norepinephrine rate categories follow the versioned threshold table. A dose exposure must overlap the score window for at least 60 minutes of the window, carry the exact `ug/kg/min` unit, be cutoff-truncated, and identify its source. No drug equivalence or generic pressor bit is inferred.

`PulkitStateSOFASupportProvider` reuses Pulkit's existing `query_invasive_ventilation_state`. The binary Pulkit vasopressor endpoint cannot supply SOFA dose categories; the adapter consequently requires Phase-9 normalized dose exposures as a separate injected input. This preserves ownership rather than creating a competing endpoint state machine.

## Renal interval details

Urine is a quantity-over-interval event available only at `interval_end`. No interval extending past the cutoff is read. No interval total is treated as a point and no proportional allocation is guessed. Urine scoring is enabled only when nonoverlapping, contiguous intervals cover the exact entire 24-hour score window. Otherwise urine is unavailable and creatinine may still make the renal component observed. The higher of valid creatinine and complete urine scores is retained.

## Cutoff and episode safety

The scorer first restricts by exact subject and stay, then ignores rows after `T` and at/before `T-24h`. It validates scoreable rows against episode bounds. The cutoff itself may be any timezone-aware time inside the episode, which permits Phase 8 to evaluate label horizons that are not prediction-grid rows. The serving adapter can additionally receive legal Phase-4 cutoffs and reject replay requests outside that set.

## Result and provenance contract

Every component returns its name, integer score, observed flag, selected evidence references, source timestamps, and spec version. The total result carries subject/stay/cutoff identity; all six named components; spec version/hash; implementation version; processed-schema version/hash; concept-map version/hash; provenance hash; timeline source version/hash; support-contract version/hash; and a deterministic hash of that source-binding envelope.

`SyntheticCurrentSOFAProvider` validates the Phase-5 processed manifest, cross-checks SOFA concepts/units/domains/event kinds against the Phase-5 concept map, calls the same `sofa_at`, and adapts the result to Pulkit's existing `CurrentSOFAState`. It does not compute recovery deltas or future reconstructed values.

## Prohibited shortcuts

Source columns named SOFA/future SOFA/component points are not consumed. The historical candidate's 0–12 three-component proxy is not imported because it omits standard components, uses different inputs, is not rolling-window/cutoff safe, has no six observed flags, and cannot satisfy the 0–24 contract.

## Version/hash policy

Hashes are produced from file bytes, never typed as assertions. Any changed spec, processed schema, concept map, provenance dictionary, timeline, or support contract creates a different binding. The storage-backed provider rejects a requested expected spec hash that differs and the score engine rejects support version/hash disagreement.
