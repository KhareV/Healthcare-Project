# Sanskruti Rebuild Phase 7 — Canonical Feature Builder Review

## 1. Executive outcome

Phase 7 is implemented for engineering fixture and smoke data. One versioned, label-free builder now converts each legal Phase-4 cutoff and Phase-5 canonical timeline into exactly eight oldest-to-newest six-hour bins over `(t-48h,t]`. It emits 15 ordered raw temporal channels, feature-aligned genuine-observation masks, feature-aligned time-since-last-observation (TSLO), a structural-padding mask, and three separately ordered raw statics. The same builder object is injected into Pulkit's history-truncation seam, and Vedant's XGBoost and GRU information views are derived from the same validated object.

No recovery labels, remaining-stay labels, support labels, split assignment, learned imputer, scaler, categorical encoder, model, calibration, threshold, or final-test artifact was created. Final production feature execution remains blocked because no authorized final synthetic dataset exists and Phase 9 has not supplied a reviewed support contract.

| Deliverable | Result |
|---|---|
| Frozen Phase-7 schema/order | PASS, `synthetic_feature_schema_v1`, F=15 |
| Exact 8x6h builder | PASS |
| Genuine observation mask | PASS |
| TSLO and 54-hour sentinel | PASS |
| Structural padding | PASS |
| Raw static contract | PASS, raw S=3 |
| Pulkit cutoff-history integration | PASS |
| Vedant XGBoost/GRU information parity | PASS |
| Engineering fixture bundle | PASS, 27 examples |
| Engineering smoke bundle | PASS, 136 examples |
| Final production bundle | BLOCKED, correctly not generated |

## 2. Scope and authority

This implementation follows Project Scope v2 and the repository rules. The population remains synthetic, adult, cardiac/heart-disease focused, and the task remains retrospective sequential forecasting. Phase 7 only constructs historical preprocessor inputs. The Phase-5 processed schema and concept map are the source contracts; the Phase-4 structural index is the sole source of legal cutoff identities; the existing timestamp and tensor contracts define receiver semantics.

The new schema binds these upstream contracts by SHA-256:

- processed schema: `7e0ba3d3789b5b469198773a5b1a7af2062ad973466a4af0135fdf97383688d9`;
- concept map: `24de193ea0041c128d921d1585dff17cd461af5fc7c415853449aa400e7bf74d`;
- timestamp specification: `de3a424fcd60affda178afbde49a0c83fd51610f7682cf73306d124454b18a9b`;
- tensor contract: `d998456430d632b81778d52f33a9b98214e3599263d443b191f5f8f75e307e9e`.

The feature schema itself hashes to `afd714cdac8f9c68f3d01a27969b2fd8dd8cb56b7684bd96c55131cfc75b5077`.

## 3. Canonical temporal contract

For cutoff `t`, the builder uses the hard window `(t-48h,t]`, divided into eight disjoint bins `(left,right]`, ordered oldest to newest. It does not round timestamps. An event exactly at `t-48h` is excluded; one immediately after is included; an event exactly at any bin right edge belongs to that bin only; an event exactly at `t` is included; anything after `t` is excluded.

The builder checks all three elements of structural identity:

1. `grid_index` is legal;
2. `icu_elapsed_hours = 24 + 6*grid_index`;
3. `prediction_time = intime + icu_elapsed_hours`.

Cutoff must also lie inside the episode. Rows are isolated by exact subject and stay before aggregation. Input row order is irrelevant.

## 4. Frozen temporal feature order

The authoritative order is:

| Index | Channel | Source | Unit | Aggregation |
|---:|---|---|---|---|
| 0 | `pao2__latest` | `pao2` | mmHg | latest |
| 1 | `fio2__latest` | `fio2` | fraction | latest |
| 2 | `platelet_count__latest` | `platelet_count` | 10^9/L | latest |
| 3 | `bilirubin_total__latest` | `bilirubin_total` | mg/dL | latest |
| 4 | `mean_arterial_pressure__latest` | `mean_arterial_pressure` | mmHg | latest |
| 5 | `glasgow_coma_scale__latest` | `glasgow_coma_scale` | score | latest |
| 6 | `creatinine__latest` | `creatinine` | mg/dL | latest |
| 7 | `urine_output_volume__sum` | `urine_output_volume` | mL | sum of available interval quantities |
| 8 | `heart_rate__latest` | `heart_rate` | beats/min | latest |
| 9 | `systolic_blood_pressure__latest` | `systolic_blood_pressure` | mmHg | latest |
| 10 | `diastolic_blood_pressure__latest` | `diastolic_blood_pressure` | mmHg | latest |
| 11 | `respiratory_rate__latest` | `respiratory_rate` | breaths/min | latest |
| 12 | `oxygen_saturation__latest` | `oxygen_saturation` | percent | latest |
| 13 | `temperature__latest` | `temperature` | degree Celsius | latest |
| 14 | `lactate__latest` | `lactate` | mmol/L | latest |

Every Phase-5 model-candidate concept is represented exactly once. Point observations use latest because that is deterministic, minimally assumptive, and avoids automatically multiplying channels. Same-time latest ties use lexicographically greatest event identity after sorting by `(event_time,event_id)`, consistent with deterministic Phase-5 identity semantics. Urine is a quantity interval, so available interval volumes whose availability/end time is within the bin are summed; it is not converted into a point measurement.

Mean, minimum, maximum, slope, and delta are deliberately **not model channels in v1**. Consequently their generic test cases are not applicable to this schema: implementing them would create unapproved model information. In particular, no slope is inferred from forwarded copies and no delta is computed against an imputed or pre-lookback value. Adding any such channel requires a schema version bump and review.

## 5. Missingness and carry-forward

There is no carry-forward in v1. An active, nonpadding channel without genuine in-bin evidence remains JSON `null`/Python `None`, and its observation mask is false. Phase 10 must fit any imputation using training subjects only. This decision is compatible with the current raw preprocessor-input boundary and avoids allowing old observations to masquerade as current evidence.

For a latest or sum channel, the mask is true only when at least one genuine eligible source event supports that bin value. Imputation, padding, TSLO sentinels, and future transformations never set the mask. Padded values are null and padded observation masks are false.

## 6. TSLO contract

TSLO is a `[8,F]` matrix measured in hours at each bin end. It searches only genuine observations inside the hard `(t-48h,t]` lookback and at or before the evaluated bin end. A genuine event resets TSLO; missing later bins allow elapsed time to increase. Values are never used to reset TSLO because no forward fill exists.

When no genuine in-lookback observation exists, TSLO is `54.0`. This finite sentinel is strictly beyond the maximum valid in-window TSLO, which is less than or equal to 48 hours under the left-open window. Structural padding also uses `54.0` as a representation rule, while remaining distinguishable through `padding_mask=true`. Validation rejects a different TSLO value in padded cells.

## 7. Structural padding

A bin is padded exactly when its right endpoint is at or before ICU `intime`. Padding must be a contiguous prefix. At the earliest legal cutoff (`intime+24h`), the first four bins are padded and the last four are active. An event exactly at `intime` is excluded because its nominal bin is structural pre-ICU padding, not active clinical history. Pre-ICU observations can neither populate values nor seed TSLO.

Padding and active missingness are separate: both may have null values and false observation masks, but only padding has `padding_mask=true`.

## 8. Static contract

Statics remain separate rather than repeated over time. Their frozen raw order is:

1. `age_years` (integer);
2. `sex_category` (string);
3. `cardiac_condition_group` (string).

Raw S is 3. Values must be present and scalar. Phase 7 does not assign numeric category codes; Phase 10 owns train-only encoding. The builder and schema explicitly reject outcome/future fields including `outtime`, duration, remaining stay, future SOFA, delta SOFA, support label, censor reason, split/test identity, latent state, trajectory, outcome class, and full-stay aggregates.

## 9. Label-free receiver contract

The previous `CanonicalExample` required historical MIMIC split and label fields and was therefore unsuitable for synthetic Phase 7. The smallest explicit receiver change adds immutable `CanonicalFeatureInput`, which carries feature identity, schema hash, embedded feature order, raw tensors/masks/TSLO/statics, and quality metadata—but no split, eligibility, or targets. Existing `CanonicalExample` behavior remains backward compatible.

Validation checks schema version/hash/order, tensor versions, 8xF shapes, data types, mask/value agreement, padding prefix, padding sentinel, static order, and legal structural grid fields. Embedding `temporal_feature_names` in every object makes reordered, missing, or extra channel attacks fail even when matrix dimensions happen to match.

## 10. Pulkit integration and training-serving parity

`SyntheticCanonicalFeatureBuilder` is callable and is injected into `CanonicalHistoryInputProvider`. Pulkit's existing provider retrieves the exact stay, verifies the requested structural cutoff, truncates history at the cutoff, and passes that history to this same builder. There is no second serving feature implementation and no prediction-table lookup.

Tests compare direct full-history construction with explicit Pulkit-truncated construction and obtain exact equality. Equivalent `Z` and `+00:00` timestamps are compared as instants, not fragile strings. Pulkit data-quality counts are derived only from masks and padding.

## 11. Vedant integration and information parity

`temporal_information_views()` accepts the new label-free input. GRU receives the canonical temporal values/masks/TSLO/padding/statics. XGBoost receives only the deterministic flattening of the same information. Tests unflatten the XGBoost view and prove exact equality of temporal values, observation masks, TSLO, padding, and statics. There are no XGBoost-only full-stay summaries or engineered outcome proxies.

Phase 10 can consume the raw statics and null dynamic cells to fit preprocessing on training subjects only. Phase 11 can consume the resulting canonical information without changing F/order or independently engineering XGBoost features.

## 12. Phase 8 and Phase 9 handoffs

Phase 8 must join labels and eligibility by exact `(stay_id,prediction_time)` identity after feature construction. It must not alter historical values, masks, TSLO, padding, feature order, or statics.

Phase 9 owns synthetic organ-support source/event truth. No Phase-5 canonical support concept exists, so Phase 7 did not fabricate ventilation or vasopressor channels. PaO2, FiO2, and MAP remain legitimate independent historical measurements. If reviewed Phase-9 decisions make support-state channels mandatory, this is a scientific schema change: create a successor feature schema, regenerate all feature artifacts, and re-review downstream compatibility. Final-mode generation currently fails closed rather than silently treating absent support as off.

Current total SOFA is not included. SOFA is a recovery-label source, not automatically a predictor, and no reviewed Phase-5 policy marked it as model eligible.

## 13. Artifact generation and lineage

`scripts/build_synthetic_features.py` validates the processed manifest, loads its canonical timeline/statics, loads the Phase-4 structural index, checks exact identity uniqueness, runs the canonical builder once per structural row, and atomically writes a JSONL input bundle, descriptive QA, and hash-bound manifest. The manifest binds upstream processed data, structural index, feature schema, timestamp/tensor contracts, builder code, counts, and semantic hashes.

Engineering fixture artifacts:

- 27 examples, F=15, raw S=3;
- padding-bin distribution `{0:11, 1:4, 2:4, 3:4, 4:4}`;
- 1,504 raw-missing cells;
- input SHA-256 `281bcd06fc2130a5a9d1c81e7de58609b92cd8d9b21209c90068d3b73ba56611`;
- manifest SHA-256 `cbc9d243e5b71023c6d22907737ca584f3ede01b5f801a207d7e3b1ca2f41666`.

Engineering smoke artifacts:

- 136 examples, F=15, raw S=3;
- padding-bin distribution `{0:55, 1:17, 2:19, 3:22, 4:23}`;
- 6,940 raw-missing cells;
- input SHA-256 `d5e3b16d2e8ac03860492f1154d052839458090ae3f418952a4236b1e8b8f679`;
- manifest SHA-256 `8975494a97513f8a5cad030db103eeabbcdccb6a93d69d7c80c166e2b8783cd4`.

These are engineering synthetic artifacts, not real patient data and not authorized final data. Missingness does not drop rows. QA contains only descriptive feature/mask/padding statistics and does not inspect outcomes or test performance.

## 14. Leakage and hostile-test evidence

Focused tests prove that future events, 60-hour-old events, exact left-boundary events, pre-ICU events, and rows from another subject/stay have zero effect. Unknown relevant concepts and wrong unit/provenance bindings fail closed. Same-time events are deterministic, input order is irrelevant, and duplicate event identities fail. Prohibited future/outcome static fields fail at builder initialization.

No use is made of `outtime` as a feature; it is used only to validate that a legal cutoff lies inside the episode. No total episode duration, remaining-stay value, full-stay aggregate, generator latent state, eventual outcome, label, split, or test identity enters the tensor.

## 15. Verification

The pre-change baseline was `1150 passed, 2 skipped`. The focused Phase-7 suite passes `19 passed`. The final full integrated regression passes `1169 passed, 2 skipped`. The two skips are the pre-existing optional Captum and SHAP/XGBoost integration tests; their dependencies are absent. `compileall` also completes successfully across `src`, `scripts`, and `tests`.

The focused files cover schema and receiver attacks, all time edges/internal boundaries, latest and urine aggregation, TSLO, padding, cross-identity isolation, illegal cutoff identity, future/old perturbation invariance, Pulkit direct/truncated equality, family information parity, and artifact validation. Unselected mean/min/max/slope/delta policies are explicitly absent rather than pretending to be tested production channels.

## 16. Change control and review status

The engineering Phase-7 contract is frozen at `synthetic_feature_schema_v1`. Any change to channels, order, units, aggregation, carry, TSLO, sentinel, padding, or statics requires a new schema version, regenerated manifests, and downstream review. Builder SHA-256 is `6c7a26f5d6dec83cdf67cf43927a4abe26b90e8143678dfdb445089687e3eeef`; derived provenance SHA-256 is `ad01e167e74ecd26934f06e1a11004a2648ec4bb52703f065da8929f26785e8d`.

Phase 7 is complete for authorized engineering fixture/smoke evidence. Final production execution is not approved by this document and remains blocked on genuine final-data authorization and any reviewed Phase-9 support-feature decision.
