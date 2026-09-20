# Synthetic Cardiac Generator Specification v1

## Identity and status

| Field | Value |
|---|---|
| Scientific generator specification | `synthetic_generator_v1` |
| Serialized config | `synthetic_generator_config_v1` |
| Raw schema | `synthetic_raw_schema_v1` |
| Project scope | `PROJECT_SCOPE_V2` |
| Status | **DRAFT SPECIFICATION — TEAM REVIEW REQUIRED** |
| Implementation | **NOT IMPLEMENTED — PHASE 3** |
| Final dataset | **NOT GENERATED** |
| Approval | None recorded; no approval or freeze is inferred |

This specification implements the Phase-2 design contract against the draft-complete Project Scope v2. It does not authorize Phase 3 until the unresolved generator parameters and human review items are resolved.

## Purpose and nonclinical disclosure

MIMIC authorization was requested but was not obtained. The project therefore uses project-designed synthetic cardiac trajectories for methodological and software evaluation of its retrospective forecasting pipeline.

The generator is not a MIMIC replica, epidemiologic model, digital twin, causal simulator, validated population model, or representation of an individual patient. Mixture weights and parameter values are synthetic benchmark design choices. Results on generated data are not clinical validation and cannot support bedside, prognosis, treatment, deployment, prevalence, or real-world performance claims.

## Goals

The future generator must make it possible for downstream code to construct:

- explicit synthetic ICU-like episodes and Vedant's legal repeated cutoffs;
- event-level histories for the exact `(t-48h,t]` window;
- raw inputs for a versioned six-component 0–24 SOFA calculation;
- independent DeltaSOFA24 and DeltaSOFA48 labels;
- time remaining until the current episode ends;
- vasopressor and invasive-ventilation OFF→ON state transitions;
- task eligibility and incomplete-follow-up censoring;
- one shared canonical builder producing information-equivalent inputs for XGBoost and GRU;
- cutoff-safe retrospective serving and replay.

The generator emits raw facts. It does not calculate any of those derived representations, eligibility flags, or targets.

## Non-goals and phase boundary

Phase 2 does not implement simulation, generate patients, choose a final cohort size or seed, create prediction times, bin events, forward-fill, impute, normalize, compute SOFA, create labels, assign splits, train models, calibrate probabilities, create predictions, or modify downstream scientific modules. It creates only specifications, validation code, and governance tests.

## Seven-layer architecture

1. **Subject and episode structure:** opaque identity, adult cardiac statics, explicit start/end, and exactly one core episode per subject.
2. **Hidden dynamic physiological state:** internal continuous stochastic factors that are never model-visible.
3. **True physiological variables:** latent-driven, cross-correlated continuous or categorical clinical quantities.
4. **Observation process:** irregular variable-specific measurement opportunities, missingness by absence, and bounded measurement noise.
5. **Intervention/support process:** stochastic OFF/ON transitions and closed-open support intervals.
6. **Exported raw data:** subjects, episodes, long-form events, support intervals, and an execution manifest.
7. **Downstream derivation:** timestamps, bins, masks, TSLO, SOFA, labels, splits, and preprocessing. Layer 7 is not generator-owned.

The architecture is strictly:

```text
generator specification/implementation
  -> raw synthetic cardiac timeline
  -> episode/cohort acceptance
  -> Vedant timestamp generator
  -> one shared cutoff-aware feature builder
  -> labels, models, PredictionPipeline, replay
```

## Population contract

The population is adult and cardiac/heart-disease focused. It is not unrestricted ICU and must not silently become heart-failure-only. `cardiac_condition_group` provides a structural location for a reviewed multi-phenotype vocabulary, but the exact taxonomy and mixture remain `REVIEW REQUIRED`. Any future mixture is benchmark composition, not estimated prevalence.

The only currently required statics are adult age, a reviewed sex/gender category, and cardiac condition group. Additional comorbidities are excluded unless a later review states their purpose, generation mechanism, model eligibility, and leakage analysis.

Final subject count remains unlocked. Phase 3 may not copy the old 100-subject convenience default. The sizing decision must consider independent subjects, resulting episodes and repeated cutoffs, eligible stays per task, validation-search stability, grouped evaluation/bootstrap, and actual compute/storage constraints. Prediction rows must never be misreported as independent subjects.

## Identity and episode contract

Identifiers are deterministic opaque strings derived from subject ordinal and local seed policy. They contain no phenotype, target, support outcome, duration, or split. Real identifiers, names, MRNs, and copied patient dates are forbidden. Collisions are hard failures.

Core v1 generates exactly one canonical episode per subject. Every episode contains explicit `subject_id`, `stay_id`, `intime`, and `outtime`, with `outtime > intime`. Measurement boundaries never substitute for structural episode boundaries.

Duration follows a truncated log-normal structural model:

```text
log(duration_hours)
  = duration_intercept
  + duration_latent_coefficients dot z_0
  + duration_support_propensity_coefficient * support_propensity
  + duration_noise
```

`duration_noise` is stochastic. Parameters and truncation bounds remain review-required. The model must yield variable durations and cannot be a deterministic algebraic function of one exported feature. `outtime = intime + duration`; it is structural/label-only and never a model feature.

Duration variability must permit structural coverage of no-cutoff, one-cutoff, multiple-cutoff, +24-only, censored-support, and long-episode cases. These are coverage requirements, not prevalence quotas.

## Time policy

All timestamps are timezone-aware UTC ISO-8601 values with sufficient precision for exact boundary tests. Naive/aware mixing is prohibited. Synthetic dates originate from a deterministic, nonclinical epoch (`2100-01-01T00:00:00Z`) and per-subject substream offsets. Calendar year carries no scientific meaning, is not model-eligible, and cannot determine split.

All exported events satisfy `intime <= event_time <= outtime`. There are no pre-episode events in v1. Ordering is deterministic by time and stable event identity. No fake MIMIC anchor years are emitted.

The full stored episode contains future events relative to early cutoffs. That is intentional. Pulkit's history layer and the shared builder must expose only facts genuinely available at or before each cutoff.

## Hidden continuous dynamics

The internal state is a five-dimensional continuous vector:

- systemic severity;
- cardiac stress;
- respiratory burden;
- renal burden;
- neurologic burden.

At an internal one-hour simulation step:

```text
z_(n+1) = z_n
          + K*(mu_i-z_n)*dt
          + d_i*dt
          + C*tanh(z_n)*dt
          + shock_(i,n)
          + L*epsilon_(i,n)*sqrt(dt)
          + documented_intervention_effect_(i,n)
```

`mu_i` and `d_i` provide continuous subject heterogeneity; `K` provides mean reversion; `C` creates bounded cross-domain coupling; `L*epsilon` combines shared and organ-specific stochastic innovations; sparse signed shocks create nonlinear turning points; and any intervention effect is bounded and delayed. Exact matrices/distributions remain review-required, but the algorithm cannot be replaced by implementation preference.

There is no improving/stable/deteriorating latent class. Improvement, worsening, fluctuation, and approximate stability emerge from continuous processes. Hidden states are not exported. Per-patient latent traces should not be persisted; if temporarily required for development validation, they live only in `GENERATOR_INTERNAL_AUDIT_ONLY` and are excluded from final artifacts, preprocessing, models, serving, and evidence.

## True physiology and cross-variable coherence

For numeric concept `v`:

```text
x_v(t) = inverse_link_v(
    baseline_v
    + patient_effect_v
    + loading_v dot z(t)
    + periodic_term_v(t)
    + process_error_v(t)
)
```

Variable-specific autocorrelated errors prevent all variables from collapsing to one severity scalar. Shared latent loadings create meaningful covariance. Categorical states use latent-dependent softmax probabilities. Static values do not change within an episode.

The process uses explicit synthetic safety domains and unit contracts, not asserted clinical reference ranges. Sampling links/truncation must enforce domains by construction; silent clipping cannot conceal defective distributions. Coherence checks include pressure relationships, bounded oxygen fractions/saturations, integer GCS, interval-consistent urine output, support-state consistency, and finite observations. This is modest consistency—not a full physiologic or causal simulator.

## Raw physiological inventory

Mandatory SOFA prerequisites are PaO2, FiO2, platelet count, total bilirubin, mean arterial pressure, vasopressor context/rate, GCS, creatinine, interval urine-output volume, and invasive-ventilation state/context. Candidate cardiac/critical-care concepts are heart rate, systolic and diastolic pressure, respiratory rate, oxygen saturation, temperature, and lactate.

Every concept in the machine config has one unit, type, synthetic generation domain, event semantics, latent dependencies, and missingness policy. Phase 5—not this specification—chooses final model features, aggregation, and order. Phase 6—not this specification—defines SOFA thresholds, lookbacks, component observability, and `sofa_at(T)`.

## Observation and missingness process

True physiology and observation are separate. For concept `v`, observations arise from a severity-modulated renewal process:

```text
lambda_v(t)
  = base_rate_v
    * exp(observation_loading_v dot z(t) + patient_observation_effect_v)
```

Configured lower/upper rate bounds prevent degenerate schedules. Local-RNG exponential waits produce irregular event times. Rates differ by variable and may change with hidden severity, creating documented informative missingness without an exported outcome label.

No measurement means no row. The final event table does not contain placeholder rows with null values. The generator never forward-fills, imputes, normalizes, standardizes, calculates masks or TSLO, or aligns to one- or six-hour grids. Final data contain no deliberate ambiguous duplicate events; duplicates belong only to hostile engineering fixtures.

## Measurement noise

Observed measurements are distinct from true states:

```text
observed_v = bounded_inverse_link_v(
    link_v(true_v) + subject_bias_v + measurement_error_v
)
```

Bias and errors use dedicated deterministic substreams. Scales/distributions remain review-required. Observations must be finite and in the declared synthetic generation domain. Invalid output is a hard failure rather than silent repair.

## Support-process contract

The generator emits closed-open `[start,end)` intervals—not support labels. OFF→ON and ON→OFF transitions follow stochastic Markov hazards conditioned on current/recent hidden state, continuous patient effects, and random innovations. They are neither independent coin flips nor deterministic copies of one visible threshold. Intervals may span multiple legal cutoffs and are truncated at episode end.

Vasopressor intervals expose a reviewed `agent_key`, start/end, and segmented nonnegative rate/unit where relevant. Final synthetic vocabulary and activity units remain review-required. The contract targets Pulkit's `NormalizedActiveInterval` and explicitly distinguishes qualifying from nonqualifying entries.

Respiratory intervals use the existing internal categories `INVASIVE`, `NON_INVASIVE`, `HFNC`, and `ORDINARY_OXYGEN`. Only `INVASIVE` qualifies for the organ-support endpoint. The contract targets Pulkit's `NormalizedVentilationInterval`; unknown categories fail closed.

If support influences physiology, it may apply a bounded delayed change to the relevant hidden drift. Support cannot instantaneously reset severity because a positive event occurred, and no causal treatment claim is allowed.

## Raw artifact layout

The format-independent schema defines:

- `subjects`: opaque identity and restricted model-eligible statics;
- `episodes`: identity, explicit bounds, episode ordinal, and RNG provenance;
- `raw_events`: long-form observed point/interval events with explicit event time, value, unit, and process identity;
- `support_intervals`: state intervals/rate segments consumable by Pulkit adapters;
- `generator_internal_audit`: nonexported development-only latent validation structure;
- `synthetic_dataset_manifest_v1`: execution identity, counts, paths, formats, and hashes.

The primary storage format remains unlocked because the final environment/package manager is unresolved. Phase 3 must select one—not multiple redundant formats—after comparing typed schema, deterministic reading/canonical hashing, expected size, installed dependencies, and downstream compatibility. No database server is required.

## Field-role and model-eligibility policy

Every planned field is assigned exactly one role in the schema and [field-role matrix](SYNTHETIC_FIELD_ROLE_MATRIX_V1.csv):

- `MODEL_ELIGIBLE_RAW`: observed raw fact, still subject to cutoff truncation and Phase-5/7 selection;
- `STATIC_MODEL_ELIGIBLE`: reviewed baseline static, never a future summary;
- `STRUCTURAL_LABEL_ONLY`: linkage/time truth required for structure or labels, never a feature;
- `PROVENANCE_ONLY`: audit and compatibility identity, never a feature;
- `GENERATOR_INTERNAL_AUDIT_ONLY`: hidden generator state, never exported.

`outtime`, total/future duration, target eligibility, censor reason, future SOFA, DeltaSOFA, remaining stay, future support, support labels, latent trajectory/state, split identity, final outcome, target, and class labels are prohibited as model features. Identifiers also never enter models.

Support interval ends require special truncation: a future cessation time must not be exposed before cessation occurs. The future history adapter must reconstruct the known state at `t` without revealing an interval's future end.

## Target and representation separation

- Generator emits PaO2/FiO2/platelets/bilirubin/MAP/GCS/creatinine/urine/support context; Phase 6 computes `sofa_at(T)`.
- Phase 8 computes independent DeltaSOFA24/48 from future cutoff-safe scores.
- Generator emits `outtime`; Phase 8 computes hours remaining and `log1p`.
- Generator emits support intervals; Pulkit determines state, at-risk status, initiation, censoring, and label.
- Vedant generates legal prediction timestamps.
- Phase 7 creates the sole canonical feature pipeline for training, serving, and dashboard replay.
- Phase 10 applies a deterministic subject split after generation.

No training-, serving-, or dashboard-specific generator pipeline is permitted.

## RNG and deterministic reproduction

The future executable generator uses local `numpy.random.Generator(PCG64)` instances only. A primary seed field exists but its final value remains unlocked. `SeedSequence([primary_seed, subject_ordinal, component_code])` derives independent named substreams for identity/statics, episode/calendar, latent dynamics, physiology, observation, measurement, and support. Component codes are frozen in the config and generation cannot depend on call/spawn order or Python's process-randomized `hash()`.

Subject and event ordering is canonical. Exact code identity, config/schema hashes, Python and NumPy versions, seed identity, and serialization policy are bound in the manifest. Identical bound inputs must reproduce byte-equivalent canonical artifacts where the approved environment promises determinism.

## Manifest and provenance

Phase 3 must populate—not fabricate—the `synthetic_dataset_manifest_v1` template. It binds Project Scope v2, generator version/code identity, generator config version/hash, raw schema version/hash, seed policy/identity, environment, timestamp/command, requested/generated subjects, per-table counts, and every artifact's logical name, path, schema table, format, record count, and SHA-256.

The current manifest file is a template with null execution values and status `SPECIFICATION_ONLY_NO_DATA_GENERATED`. It is not a dataset manifest and cannot be registered as scientific data.

## Generator validation contract for Phase 3

Phase 3 must test deterministic reproduction; unique identifiers; one episode per subject; valid explicit bounds; no out-of-episode events; valid units/types/domains; finite values; exactly one numeric/text event payload; interval collection semantics; valid closed-open support intervals; absence of model-visible latent/target/split fields; no RNG-created subject clones; and manifest count/hash integrity.

Endpoint frequencies, recovery directions, SOFA deltas, cutoffs, and eligibility are descriptive reports, not quality targets. Zero usable examples for a required endpoint is a feasibility failure. Any nonzero prevalence is reported without post-hoc generator editing to improve model results.

## Change control

After generator v1 is frozen, neither disappointing nor impressive model performance authorizes trajectory, noise, missingness, support, duration, or population changes. Any trajectory-changing parameter update creates a new config hash and regenerates every dependent artifact. Scientific process changes require a generator version bump; schema/field-semantic changes require raw-schema review/versioning. Test-set-informed generator editing is prohibited and invalidates affected results.

## Historical candidate reuse decision

Reusable concepts are local seeded RNG, irregular measurements, variable-specific missingness, explicit synthetic disclosure, bounded values, and one episode per subject. Reuse requires redesigned contracts and local substreams.

Rejected concepts are the public `trajectory` field, three class-conditioned fixed baseline/slope templates, target reuse of that class, inferred admission/discharge from measurements, first-48-hour-only prediction, 24/48/72 class labels, global preprocessing, deliberate duplicates in final data, unsupported “realistic” claims, and a convenience final size of 100.

## Vedant integration

`subject_id` maps to Vedant subject identity. `stay_id/intime/outtime` directly instantiate `RetainedICUStay`. Vedant—not the generator—creates legal repeated cutoffs. Full event history feeds the future shared canonical builder; subject statics feed its static view. Episode bounds determine horizon availability. Generator/schema/config/dataset hashes feed synthetic acceptance, registry, and reproducibility. No Vedant module is changed in Phase 2.

## Pulkit integration

Long-form events map to a future authorized synthetic `HistoryProvider`; `event_time` enforces `<=t` truncation. Support intervals adapt to current vasopressor/ventilation state contracts. SOFA prerequisite events feed a future shared `CurrentSOFAProvider`. Generator/schema identities become compatibility metadata. Future interval ends require availability-aware truncation. No Pulkit state, pipeline, API, dashboard, or explanation module is changed in Phase 2.

## Unresolved decisions before final Phase-3 execution

- final subject count and evidence-backed sizing decision;
- primary seed;
- cardiac subtype vocabulary and benchmark mixture;
- age maximum/distribution and sex/gender vocabulary/distribution;
- duration bounds/coefficients;
- latent baseline, drift, coupling, shock, and noise parameters;
- true-physiology loadings and variable-specific process parameters;
- observation/missingness rate parameters;
- measurement-noise parameters;
- vasopressor vocabulary, units, and support hazard/effect parameters;
- one primary storage format/output path;
- final environment and package manager.

These unresolved scalar/engineering choices do not permit an implementer to change the documented architecture or invent outcome classes, labels, split behavior, episode semantics, table relationships, or model-visible latent state.

## Phase-3 implementation contract

Phase 3 may implement a deterministic simulator only after human review resolves its required parameters. It must consume the serialized config, validate the schema before writing, populate the manifest from actual execution, create no model-ready representation, and fail closed on every prohibited export or structural inconsistency. Any proposed implementation divergence returns to reviewed specification change control.
