# Synthetic Generator Leakage Threat Model v1

Status: **DRAFT — REVIEW REQUIRED**. Scope: `synthetic_generator_v1` and `synthetic_raw_schema_v1`. This document defines prevention obligations; it does not claim that a final generator or dataset exists.

## Trust boundaries

The generator may use hidden state internally. The exported raw dataset may contain entire episodes. A downstream request at cutoff `t` may see only genuine facts available at or before `t`, transformed through one shared canonical builder. Structural label fields are available only to governed label/eligibility code. Final-test membership is assigned after generation and remains outside the generator.

| ID | Leakage route | Failure mode | Prevention mechanism | Executable-test owner / phase | Status |
|---|---|---|---|---|---|
| L01 | Latent trajectory exposed | A public improve/stable/deteriorate field directly identifies outcome-generating template | No latent class exists; hidden vector is `GENERATOR_INTERNAL_AUDIT_ONLY`; exported-schema allowlist | Phase 2 spec validator; Phase 3 artifact validator | SPECIFIED |
| L02 | Hidden state exported under an alias | Continuous latent severity becomes a model column | Role allowlist; final export excludes internal table; semantic inventory review rather than substring-only defense | Phase 2/3 | SPECIFIED |
| L03 | Future physiology exported at cutoff | Full episode rows after `t` enter features | `event_time <= t` truncation; adversarial future-mutation equality tests | Pulkit history + Phase 7 | SPECIFIED |
| L04 | Future SOFA field | Generator writes SOFA at +24/+48 or a future severity summary | Generator emits raw domains only; `sofa_at(T)` is Phase 6; prohibited exports | Phase 2/3/6 | SPECIFIED |
| L05 | Recovery target copied | DeltaSOFA or recovery class is stored as input | No label generation; explicit prohibited names/categories; schema additional fields fail | Phase 2/3/8 | SPECIFIED |
| L06 | `outtime` feature | Exact remaining episode time is trivial | `outtime` role is `STRUCTURAL_LABEL_ONLY`; canonical schema must reject it | Phase 2 validator; Phase 7/8 | SPECIFIED |
| L07 | Duration proxy static | `total_episode_hours` or discharge state leaks remaining time | Prohibited semantic category; restricted statics allowlist | Phase 2/3/7 | SPECIFIED |
| L08 | Future support flag | `support_will_start` reveals label | No future flag; only genuine intervals; names and semantic categories prohibited | Phase 2/3/9 | SPECIFIED |
| L09 | Future interval end known early | A support interval row exposes planned cessation after `t` | History adapter reveals start/state and only cessation observed by `t`; future end withheld | Pulkit Phase 7 migration/Phase 9 | DESIGN OBLIGATION |
| L10 | Support label encoded in statics | Phenotype/risk group deterministically identifies future support | No risk group; stochastic hazards with overlapping conditionals; static review | Phase 2/3/9 | SPECIFIED |
| L11 | Target encoded in identifier | IDs such as `DETERIORATING_0001` leak class | Opaque ordinal formats; collision/format validator; no subtype/target tokens | Phase 2/3 | SPECIFIED |
| L12 | Split encoded in generator output | Generator sees or emits train/validation/test | No split table/field/config; generation occurs before Phase-10 split | Phase 2 validator; Phase 3/10 | SPECIFIED |
| L13 | Split-specific generation | Different distributions are generated per partition | Generator has no partition input; single population run precedes split | Phase 3 lineage; Phase 10 | SPECIFIED |
| L14 | Subject clones cross split | RNG substream bug creates duplicated trajectories later assigned apart | Per-subject substreams plus clone audit before split; clone groups cannot cross split | Phase 3 and Phase 10 | SPECIFIED |
| L15 | Three deterministic templates | Model recovers generator class rather than forecasting uncertain future | Continuous heterogeneous stochastic dynamics; no class-conditioned fixed slopes | Phase 2 review; Phase 3 distribution audit | SPECIFIED |
| L16 | One latent scalar controls all variables | Future becomes near-deterministic and multivariate state unrealistic | Five domains; shared and variable-specific loadings/noise; coupling audit | Phase 3 | SPECIFIED |
| L17 | Episode cessation equals last measurement | Measurement density reveals exact end/remaining time | Explicit structural `outtime`; observation process separate; no inference from cessation | Phase 2/3/8 | SPECIFIED |
| L18 | Eligibility/censor reason feature | Future follow-up availability leaks endpoint | Eligibility and censoring derived downstream and excluded from feature view | Phase 7/8/9 | SPECIFIED |
| L19 | Future-informed missingness | Events deleted according to eventual outcome class | No outcome class; observation intensity depends only on current hidden state/local randomness | Phase 3 causal-order audit | SPECIFIED |
| L20 | Imputation before split | Global medians/statistics leak validation/test | Generator performs no imputation; Phase 10 fits learned preprocessing on training subjects only | Phase 2 test; Phase 10 | SPECIFIED |
| L21 | Pre-lookback forward fill | Data at/before `t-48h` seeds canonical history | Generator emits raw events; Phase 7 exact-window builder rejects earlier seeds | Phase 7 | SPECIFIED |
| L22 | Fake calendar split | Synthetic year recreates MIMIC-era membership | UTC synthetic epoch has no split semantics; no anchor-year field | Phase 2 validator; Phase 10 | SPECIFIED |
| L23 | Provenance used as feature | Generator component/seed/hash reveals generation mixture | Provenance-only roles excluded from canonical schema | Phase 2/5/7 | SPECIFIED |
| L24 | Observation identifier ordering leaks state | Event IDs encode concept severity or future count | IDs derive from stay/concept/local ordinal only; not model-eligible | Phase 3 | SPECIFIED |
| L25 | Deliberate duplicates encode group | Duplicate presence becomes a target proxy | No ambiguous duplicates in final output; hostile duplicates confined to fixtures | Phase 3 | SPECIFIED |
| L26 | Support threshold copies visible feature | One MAP value deterministically yields support initiation | Stochastic multi-factor hazard with innovations and random effects | Phase 3/9 | SPECIFIED |
| L27 | Generator tuned to performance | Trajectories edited after validation/test results | Frozen version/hash; no performance gate; result-affecting changes regenerate and invalidate | Governance/G3 | SPECIFIED |
| L28 | Internal audit accidentally shipped | Latent trace joins raw artifacts through wildcard export | Internal table `exported=false`; manifest assertion forbids latent artifact; explicit artifact allowlist | Phase 3 manifest validator | SPECIFIED |
| L29 | Full-episode aggregates | Count/max/slope over future events enters XGBoost | Generator exports no aggregates; Phase 7 provides identical cutoff-local information to XGB/GRU | Phase 7/11 | SPECIFIED |
| L30 | Serving/training divergence | Serving truncates differently than training | One shared Phase-7 builder and cutoff-history contract; equivalence tests | Phase 7/Pulkit integration | SPECIFIED |

## Hostile questions and required answers

1. Can a model read a static generator target? **No.** No outcome class or risk group exists in the exported schema.
2. Is `outtime` a model input? **No.** It is structural/label-only.
3. Does generation know split membership? **No.** Phase 10 splits after generation.
4. Does generation emit DeltaSOFA? **No.** Phase 6/8 derive scores and deltas.
5. Does generation emit organ-support labels? **No.** It emits intervals only.
6. Can Pulkit derive OFF/ON? **Yes by contract**, after a future synthetic adapter and vocabulary freeze.
7. Can Phase 6 calculate all six SOFA domains? **Yes by raw-domain coverage**, subject to future scoring/observability rules.
8. Can Vedant generate cutoffs? **Yes**, from explicit `stay_id/intime/outtime`.
9. Can stored history be truncated at `t`? **Yes**, by event availability and state-interval semantics.
10. Can future mutations after `t` leave history through `t` unchanged? **Yes by required construction and later adversarial tests.**

## Residual risks requiring later executable evidence

Phase 2 validates documents and metadata, not generated records. Phase 3 must prove causal RNG ordering, no latent export, no clones, and manifest integrity. Phase 7 must prove exact cutoff/window isolation and training-serving equivalence. Phase 9 must prove interval adaptation and censoring. Phase 10 must prove subject/clone isolation and train-only fitting. Until those tests exist, these are design controls rather than empirical guarantees.
