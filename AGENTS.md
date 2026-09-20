# Project Agent Rules

These rules apply to the entire repository. The active scope amendment is `docs/governance/project_scope_v2.md`, discovered through `docs/CODEX_PROJECT_CONTEXT_V2.md`. It amends the two v1.0 FINAL DEVELOPMENT BASELINE documents named in `docs/CODEX_PROJECT_CONTEXT_V1.md` without erasing their history. Project Scope v2 is a complete draft pending genuine team freeze; no approval is implied by file presence.

Sanskruti Phase 2 generator design is specified in `configs/synthetic/synthetic_generator_v1.yaml` and `docs/sanskruti/SYNTHETIC_GENERATOR_SPEC_V1.md`. It remains draft/review-required, contains no generator implementation, and does not authorize final data generation.

## Authority and change control

- Scientific source of truth: explicit later team decisions and Project Scope v2 for amended topics; then the Final Implementation Master Plan for unchanged topics. Ownership/execution source of truth: the Final Work Division for unchanged responsibilities. Then repository code/configuration, then general engineering knowledge.
- Do not silently change cohort, time semantics, prediction grid, lookback, labels, eligibility/censoring, split, metrics, selection, calibration, threshold, or serving semantics. Use reviewed, versioned change control and identify stale downstream artifacts.
- Never invent synthetic generator fields, clinical definitions, distributions, seeds, split parameters, or mappings. Mark unavailable decisions `UNRESOLVED — REQUIRES VERIFICATION` and stop dependent scientific work. Historical MIMIC requirements are not final-data requirements under Project Scope v2.
- Build against versioned contracts, not another member's internals. Do not patch downstream code around an incompatible contract.

## Frozen scientific invariants

- Retrospective event-time forecasting on authorized final synthetic data; adult cardiac/heart-disease subject; one canonical synthetic ICU-like episode per subject, using the chronologically first eligible episode if multiple are later generated.
- Generate `t_k = intime + 24h + 6h*k`, `k=0..11`, only when `t_k <= outtime - 6h`; at most 12 cutoffs per stay.
- Canonical history is `(t-48h, t]` in eight disjoint 6-hour bins. Nothing at or before `t-48h` may populate or seed forward fill. Pre-ICU positions are padding, not clinical missingness.
- Observation mask is 1 only for a genuine observation in the relevant bin. TSLO uses only genuine in-lookback observations and the frozen sentinel when none exists.
- Recovery targets are independent: `SOFA(t+24)-SOFA(t)` and `SOFA(t+48)-SOFA(t)`. Never chain horizons. Use one pinned `sofa_at(T)` implementation for every cutoff.
- ICU-time target is `log1p((outtime-t) hours)`: time until the current ICU stay ends, not hospital discharge, recovery, or survival.
- Organ support is eligible OFF-to-ON initiation of qualifying vasopressor or invasive ventilation in `(t,t+24h]`. Incomplete follow-up without an event is censored; a positive before exit remains positive.
- Use one canonical grid with task-specific eligibility masks.
- The historical MIMIC anchor-era split is superseded. The synthetic split policy is unresolved until its reviewed later phase; it must be deterministic, immutable, subject-disjoint, clone-safe, and preserve train/validation/sealed-final-test governance. Never assign fake MIMIC years.
- XGBoost and GRU must receive identical canonical information; XGBoost only flattens it. LSTM is one fixed sensitivity run per task using the selected GRU settings where meaningful, with no independent search.
- Select XGBoost versus GRU independently per task on validation: recovery lowest stay-balanced 24h MAE; ICU time lowest stay-balanced weighted median absolute error in hours; support highest stay-balanced AUPRC.
- Calibrate only the selected support classifier with validation-fitted isotonic regression; choose its F1 threshold on calibrated validation probabilities. Freeze both before test.
- Weight each eligible snapshot in stay `i` by `1/n_i`. Bootstrap `stay_id` clusters with replacement, carry all rows per stay copy, recompute the same estimator, and use percentile 95% CIs. Do not call these individual-patient intervals.
- Serving explanations follow the selected family: XGBoost uses TreeSHAP; GRU uses Integrated Gradients.

## Engineering, testing, and governance

- Use configuration-driven, script/package-based execution; no notebook-only implementation or hidden local state.
- Record deterministic seeds, run IDs, exact configs and hashes, code commit, environment, data/cohort/feature/label/split/preprocessor versions, and artifact lineage.
- Fit imputers, encoders, scalers, target statistics, and class weights on training subjects only.
- Test semantics and adversarial failure modes: exact time/bin boundaries, cap, horizon masks, pre-ICU/pre-lookback/future leakage, padding versus missingness, genuine-observation masks/TSLO, subject isolation, information parity, task censoring, weighted metrics, cluster bootstrap, validation-only calibration, artifact compatibility, and training-serving equivalence.
- Test data is unavailable for design, preprocessing, tuning, architecture/family selection, calibration, or thresholding. It opens only after the G3 freeze. A result-affecting defect after opening requires transparent invalidation, version reset, and affected-path rerun.
- Performance is never a correctness or completion gate. Negative findings are valid. MTL/TFT are optional and must not destabilize the core.
- Never commit restricted data, credentials, tokens, identifying extracts, or material presented as real patient data. Authorized final synthetic data must remain distinct from engineering fixtures and carry generator/schema/config/seed/dataset provenance.

## Ownership boundaries

- Sanskruti: synthetic generator/data truth, cardiac cohort and episode contract, timeline/provenance, synthetic standard-SOFA inputs and implementation, canonical features, recovery and episode-time labels, support-source collaboration, and primary XGBoost implementation.
- Vedant: timestamps, subject split, tensor/loader and training framework, GRU/LSTM, tuning, stay-balanced evaluation, grouped bootstrap, per-task selection, support calibration/threshold, registry, test governance, model/evaluation evidence.
- Pulkit: support event-state/label, PredictionPipeline, model/explanation routing, FastAPI, dashboard/replay, integration, packaging.
- Follow the active amendment in `docs/CODEX_PROJECT_CONTEXT_V2.md` and the unchanged cross-review/handoff contracts in `docs/CODEX_PROJECT_CONTEXT_V1.md`. Phase work must respect G0–G5 isolation and the Sanskruti rebuild phase boundary.
