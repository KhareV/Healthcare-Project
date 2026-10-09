# Performance V3 — Governance and Lineage

Status: V3 DEVELOPMENT GOVERNANCE (authored 2026-10-10 on branch `healthcare-performance-v3`).
Nothing in this document edits, supersedes, or re-signs a historical artifact. Historical
documents are annotated here, never rewritten, because several are hash-bound.

## 1. Authority chain

| Layer | Source | What it governs |
|---|---|---|
| Scientific source of truth | `Personalized_Patient_Recovery_Trajectory_FINAL_Implementation_Master_Plan.docx` (v1.0, 17 Sep 2026; outside the repo, in `~/Downloads`) | Prediction unit, 6-hour grid, 48-hour 8-bin history, independent SOFA deltas, ICU remaining-time target, OFF→ON support initiation, stay-balanced metrics, grouped bootstrap, validation-only selection, test isolation |
| Ownership source of truth | `…FINAL_Work_Division_3_Members (1).docx` (v1.0) | Sanskruti = data/labels/XGBoost, Vedant = split/evaluation/selection/governance, Pulkit = support state/serving/product |
| Dataset substitution | `docs/governance/synthetic_design_decision_authority_v1.md`, `docs/governance/synthetic_split_preprocessing_decision_v1.md` (`FROZEN_SYNTHETIC_AUTHORIZED`, `USER_DELEGATED_AI_PROJECT_DECISION`) | Replaces MIMIC-IV with the authorized synthetic adult-cardiac ICU-like cohort and defines its subject split. Named-human reviewer approvals (SUP-020) were never obtained; only delegated authority froze the design. |
| Executed lineage | Benchmark V1 → Performance V2 (below) | Frozen, consumed, immutable |

Master Plan scientific definitions are unchanged by V3. V3 changes only candidate
features, candidate model families, search budget, calibration choice, and cohort size,
all inside the model-development layer the Master Plan leaves to validation evidence.

## 2. Completed Benchmark V1

* Generator: `configs/synthetic/final_benchmark_v1.json` (seed `20260921`, 2,000 subjects), latent process `latent_process_v1.yaml`, support process `support_process_v1.yaml`.
* G1 acceptance: `artifacts/acceptance/g1_synthetic_data_freeze_v1.json`. Subject split 1,400 / 300 / 300 (train / validation / test).
* Selection: recovery XGBoost, ICU time GRU, organ support XGBoost. G3 freeze `artifacts/governance/g3_freeze.json`; G4 freeze `artifacts/governance/g4_test_evaluation_freeze_v1.json`.
* Final test consumed once: `artifacts/governance/test_access_state.json` = `FINAL_RUN_COMPLETED`. An earlier zero-row crashed attempt is retained in `artifacts/governance/history/` and `stage5_final_test_access_incident_v1.json`. The V1 test subjects are exposed.

## 3. Completed frozen Performance V2

* Selected models: `artifacts/performance_v2/phase3/selected_models_v2.json`
  SHA-256 `6fed05f3c71ce648658b2f864e69fa9f80f3ec8538f53a4e370fa8a4eaf176a8` (verified on disk 2026-10-10).
* Final evaluation freeze: `artifacts/performance_v2/governance/v2_final_evaluation_freeze_v1.json`
  SHA-256 `a433c1ee203f6cf9f01b131a2a9a10c73da57097564705ce64d7351c0057b649` (verified on disk 2026-10-10).
* Fresh test cohort: `configs/performance_v2/fresh_test_cohort_v1.json` — 1,000 subjects, namespace `SYN-V2`, seed 402995653 derived from SHA-256 of `selected_models_v2.json`. Consumed once (`v2_fresh_test_access_state.json`, `v2_second_run_guard_v1.json`). The V2 test subjects are exposed and are never V3 development data.
* Frozen choices: Recovery24 XGBoost `B_MIN`; Recovery48 XGBoost `B_MIN`; ICU time XGBoost `B_PLUS_F`; Support XGBoost `B_FULL`. Older prose describing Recovery48 as `B_FULL` is wrong; the frozen manifest governs.
* Reference final-test values (verified against `docs/performance_v2/FINAL_V2_TEST_EVALUATION.md`): Recovery24 MAE 1.0744; Recovery48 MAE 1.3379; ICU weighted median AE 5.2060 h; support raw AUROC 0.7869 / AUPRC 0.7637 / Brier 0.2016; calibrated AUROC 0.7772 / AUPRC 0.7042 / Brier 0.2024; F1 at frozen threshold 0.6928.

## 4. Parked draft generator-rebuild documents

`docs/governance/project_scope_v2.md` ("DRAFT COMPLETE — TEAM FREEZE REQUIRED"), `docs/CODEX_PROJECT_CONTEXT_V2.md`,
`docs/sanskruti/PHASE1_SCOPE_V2_*`, `PHASE2_GENERATOR_SPEC_REVIEW.md`,
`configs/synthetic/synthetic_generator_v1.yaml` (null parameters, `SPECIFICATION_ONLY`),
`configs/synthetic/cohort_spec_v2.yaml` (`DRAFT_IMPLEMENTATION_REVIEW_REQUIRED`),
`configs/synthetic/support_process_v1.proposed.yaml` (`PROPOSED_NOT_AUTHORIZED`; superseded in practice by the frozen sibling `support_process_v1.yaml`),
`configs/synthetic/synthetic_dataset_manifest_template_v1.json` (null template), `configs/split_spec_v1.yaml` (MIMIC anchor-era split, inapplicable to synthetic data).

These were never approved and never executed as a rebuild. A generator implementation
(`src/data/synthetic/*`, `scripts/generate_synthetic_dataset.py`) and the frozen config
lineage in section 2 are what produced every accepted dataset. V3 neither resumes nor
depends on the rebuild; the text in `AGENTS.md` that calls Project Scope v2 the active
amendment, and `docs/evidence/models/evidence_manifest_v1.json` / `VALIDATION_SUITE_GOVERNANCE.md` /
`configs/tensor_contract_v1.json`, are stale relative to the executed lineage. They are
left byte-identical on purpose (several are hash-bound, e.g. `project_scope_v2_hashes.json`).

## 5. V3 scope and rationale

V3 asks whether (a) more development subjects, (b) richer cutoff-safe temporal features,
(c) stronger boosted-tree / sequence candidates, and (d) a better-chosen support
calibrator improve held-out prediction, judged on a brand-new sealed cohort against the
unmodified V2 models. Negative or mixed outcomes are valid and reported.

## 6. What stays identical

Prediction clock `t_k = intime+24h+6h·k, k=0..11, t_k ≤ outtime−6h`; `(t−48h,t]` in eight
6-hour bins with padding/observation-mask/TSLO semantics; `ΔSOFA24`, `ΔSOFA48` (independent,
raw, unclipped for evaluation); `log1p(outtime−t)` with non-negative inverse; OFF→ON
initiation of qualifying vasopressor or invasive ventilation in `(t,t+24h]` with censoring;
single canonical grid with task masks; stay-balanced `1/n_i` weights; stay-cluster
percentile bootstrap; validation/development-only selection, calibration and threshold.

## 7. What is new in V3

Candidate feature families (versioned `feature_contract_v3`), LightGBM/CatBoost/linear
comparators, a bounded sequence model, support calibrator comparison (raw / Platt /
isotonic / beta) selected by a precommitted rule, learning-curve study, V3 sealed
cohort, and a paired V2-vs-V3 stay-cluster bootstrap. Exact definitions live in
`configs/performance_v3/scientific_protocol_v1.yaml`.

## 8. Immutable historical artifacts

Everything under `artifacts/` (except `artifacts/performance_v3/`), `configs/`, `docs/`
(except `docs/performance_v3/`), `models/`, and the code under `src/`, `scripts/`,
`api/`, `tests/` as captured in
`artifacts/performance_v3/governance/pre_phase_hash_inventory_v1.sha256`
(2,264 files, inventory SHA-256 `f2c67aefe87980d086b4afb484c2c50674345a80653e241fa5696bf2a2dd414b`,
captured before any V3 work). New V3 code is additive; any edit to an inventoried file
must be a reviewed, listed exception in the closure audit.

## 9. Archive inventory (untracked, repo root)

* `artifacts.zip` — 281.8 MB, 4,106 entries, 1.79 GB uncompressed: a macOS-Finder backup snapshot of `artifacts/` (search, performance_v2, data, smoke, models, governance, final_test, …). Not referenced by any manifest, not part of reproducibility, not used by V3. Left untouched and uncommitted.
* `models.zip` — 327.5 MB, 8 entries: backup of `models/kokoro/` TTS ONNX weights (gitignored in the working tree). Unrelated to the science. Left untouched and uncommitted.

## 10. Legitimate comparisons

* V3 vs V2 on the new sealed V3 cohort, identical eligible rows, paired stay-cluster bootstrap — legitimate.
* V3 development metrics vs V2 development/CV metrics — descriptive only; never a promotion basis.
* Learning-curve metrics on the fixed development-evaluation partition — development evidence, not final validation; repeated looks inflate optimism and are labelled as such.
* Any comparison using V1 or V2 test subjects as V3 training/selection data — illegitimate and blocked by test.
