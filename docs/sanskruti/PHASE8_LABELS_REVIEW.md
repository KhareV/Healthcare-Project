# Sanskruti Rebuild Phase 8 — Recovery and Remaining-Episode Labels Review

## Executive outcome

Phase 8 implements and freezes two deterministic target engines without crossing into Phase 9 or Phase 10. The recovery engine calls the single Phase-6 `sofa_at(T)` result contract at `t`, `t+24h`, and `t+48h`, uses one baseline, and emits independent continuous future-minus-current deltas. The remaining-time engine computes exact elapsed hours to the current retained synthetic ICU-like episode `outtime`, followed by exact `log1p`.

The status is deliberately split:

| Component | Status |
|---|---|
| Recovery label engine | **IMPLEMENTED AND TESTED** |
| Final recovery label artifact | **BLOCKED — Phase-6 final SOFA requires Phase-9 production support evidence and review** |
| ICU-time label engine | **IMPLEMENTED** |
| Engineering ICU-time artifacts | **GENERATED** for fixture and smoke |
| Final ICU-time artifact | **BLOCKED — no authorized final retained cohort exists** |
| Combined Phase-8 target artifact | **PARTIAL**; recovery fields remain null with an explicit dependency status |

No missing dependency is misrepresented as patient ineligibility. No recovery zero, false eligibility, or support label was fabricated.

## Scope and authorities

The active scope remains synthetic, adult, cardiac/heart-disease focused, retrospective sequential forecasting. Project Scope v2 hashes to `09c2b15a05a87f78cc03b8be3f4e98475e34fdc01787fe6c58191e338e86a880`.

Phase 8 consumes, without scientifically modifying:

- Phase-4 retained episodes and outcome-free structural rows;
- Vedant's `timestamp_spec_v1` availability flags and inclusive horizon equality;
- Phase-5 canonical timeline and statics;
- Phase-6 `synthetic_standard_sofa_v1`, hash `900504be9b2d1186c15b5893fcd1ea5e4aa3d3c54c60b44d7b4b0079c79e4c13`;
- Phase-7 feature schema, hash `afd714cdac8f9c68f3d01a27969b2fd8dd8cb56b7684bd96c55131cfc75b5077`;
- Vedant's raw-delta horizon order `delta_sofa_24h`, `delta_sofa_48h`, training-only `RecoveryTargetScaler`, and direct log1p ICU-time target;
- Pulkit's independent baseline-plus-delta reconstruction and current-episode time response semantics.

Phase-6 status remains `DRAFT_REVIEW_REQUIRED_NOT_AUTHORIZED_FOR_FINAL_PRODUCTION`, and its support contract remains deferred to Phase 9. This blocks final recovery execution, not the pure formula engine.

## Versioned contracts

| Contract | Version | SHA-256 |
|---|---|---|
| Recovery | `synthetic_recovery_labels_v1` | `102fc750d8f63c8caba94b7373c12c156e6c39b002d7ac197a3f2592624085eb` |
| ICU time | `synthetic_remaining_episode_time_v1` | `c64c69d95c14f3df7e6c8d45f28b1098b585e0b81c7e19119370acb8a0f8f420` |
| Target schema | `synthetic_phase8_target_schema_v1` | `498e4ff7674bad3bbbb0d7953a1db0dba10ef19b733c3b549d3b5589ffa4a019` |
| Target provenance | `synthetic_phase8_target_provenance_v1` | `2b43fc2aafdaec014bda7a04befd3cbf94cea3351129ee257568cbf440853cf3` |

These are synthetic-contract successors because no current production Phase-8 target specification existed. They do not change Vedant's already-frozen target meanings.

## Canonical row universe and eligibility

The Phase-4 structural index is the complete row universe. Target construction never regenerates cutoffs or drops a row. Exact identity is `(subject_id, stay_id, prediction_time, grid_index)`; the future join key is exact `(stay_id,prediction_time)` with subject/grid validation. There is no row-order, nearest-time, or fuzzy match.

`recovery24_followup_available`, `recovery48_followup_available`, and `icu_time_temporally_eligible` are consumed from the structural index. The recovery engine independently recomputes temporal availability only as a consistency assertion and fails on disagreement. Equality at `t+24h==outtime` or `t+48h==outtime` is available. The Phase-6 scorer permits `sofa_at(outtime)` and arbitrary valid label cutoffs; a future score need not be another prediction-grid row.

Because final SOFA is blocked, partial artifacts retain `recovery24_structurally_eligible` and `recovery48_structurally_eligible`, while final recovery eligibility and values are null. This distinguishes structural follow-up from unavailable pipeline execution. ICU-time eligibility is true for every emitted legal row and has a present finite target.

## Recovery formula and audit contract

For one cutoff:

```text
S0  = sofa_at(t)
S24 = sofa_at(t+24h)
S48 = sofa_at(t+48h)
delta24 = S24 - S0
delta48 = S48 - S0
```

Both horizons use the same injected callable and the same `S0`. The engine validates returned stay, cutoff, six-component SOFA result, and common SOFA specification version/hash. It preserves the baseline and horizon component-observed flags in the recovery audit object; it does not apply completeness or 50-percent filters. Raw deltas remain in SOFA points, retain their sign, and are neither clipped nor standardized. The valid structural range `[-24,24]` is asserted, not used for clipping.

The mandatory golden case `S0=10`, `S24=7`, `S48=12` returns `(-3,+2)`, not `(-3,+5)`. This matches Pulkit's independent `S0+delta24` and `S0+delta48` reconstruction. No categorical improving/stable/deteriorating target exists.

For an unavailable horizon, eligibility is false and its target is null. For an available horizon, a missing target is a defect. Nullness is validated against eligibility; it is not used to infer eligibility.

## Remaining current-episode formula

For every canonical row:

```text
remaining_hours = (current_episode_outtime - prediction_time).total_seconds()/3600
icu_time_log1p = log1p(remaining_hours)
```

Arithmetic uses timezone-aware elapsed instants, permits fractional hours, and performs no rounding or bin-count approximation. Remaining time must be finite and strictly positive. The label is not clamped. There is no extra ICU-time target scaler. Vedant's prediction-side `expm1(max(0,prediction))` remains serving behavior and is not duplicated here.

Scientifically this is time until the current retained synthetic ICU-like/critical-care episode ends. It is not hospital discharge, hospital LOS, survival, recovery, death, or time to support. The pure function accepts only prediction time, episode outtime, and the frozen temporal eligibility flag.

## Feature/label separation

Future SOFA evidence and `outtime` exist only on the target/audit path. Tests prove:

- adding an event after `t` leaves the Phase-7 feature object exactly unchanged;
- changing `outtime` while keeping the cutoff legal leaves all Phase-7 model-facing fields exactly unchanged;
- the same outtime change alters remaining hours and log1p exactly;
- `outtime`, remaining hours, deltas, and future SOFA are absent from Phase-7 objects;
- Phase-7 feature artifacts are read-only and are joined only for identity compatibility.

Labels were not inspected to redesign features or retune the generator. The feature schema remains unchanged and hash-identical.

## Vedant and Pulkit compatibility

Vedant receives raw original-unit recovery deltas in fixed `[24h,48h]` order with independent masks. Phase 8 does not fit his `RecoveryTargetScaler`; Phase 10 will fit separate horizon statistics on training-eligible subjects. ICU time is already `log1p(remaining current episode hours)` and receives no target scaler.

Pulkit's reconstruction remains `current_sofa + delta24` and `current_sofa + delta48`, independently. Current SOFA stays sourced from Phase 6. No serving code consumes future labels, and no API/dashboard code was modified.

## Engineering artifacts and QA

Fixture partial artifact:

- 27 exact structural/feature identities;
- 27 ICU-time labels formula-validated;
- 15 structurally 24h-available, 3 structurally 48h-available;
- 3 both, 12 24h-only, 12 neither;
- recovery targets generated: 0, explicitly blocked;
- support targets generated: 0;
- target file hash `7e75c2c63748e5029d295a69995a906166741086e66b6270a8aececb9e9e7f87`;
- manifest hash `187a60b3b3aa12ea0c5027ee3ad28955fff32eef24324d9cedf7de185e00b88a`.

Smoke partial artifact:

- 136 exact structural/feature identities;
- 136 ICU-time labels formula-validated;
- 73 structurally 24h-available, 22 structurally 48h-available;
- 22 both, 51 24h-only, 63 neither;
- recovery targets generated: 0, explicitly blocked;
- support targets generated: 0;
- target file hash `5a9c38c4c9defe8ffbcb1892605026c3142113e1c2dc45dd4292c830cc7fc589`;
- manifest hash `ea59a37ebea1f4524acf68d812b38ddbf5efccc2c6972c6988163bfe56987b36`.

These are engineering artifacts, not final scientific data. QA reports correctness/eligibility counts only. It creates no split-specific distribution, performance gate, target statistic, or support prevalence.

## Manifest and provenance

Each manifest binds the Project-governed cohort/structural artifact, processed manifest and canonical timeline, Phase-7 feature manifest/schema, SOFA specification/status, both label specs, target schema, target provenance, builder code, output hashes, semantic hash, row count, and blocker statuses. It contains no model, split, preprocessing, scaler, or calibration hash. Validation rejects any bound-source or output drift and enforces unique exact identities, recovery nulls under the blocker, positive ICU times, and exact `expm1(log1p(x))` round trips.

## Verification

The clean pre-change baseline was `1169 passed, 2 skipped` in 18.23 seconds. Focused Phase-8 tests cover same-baseline/no-chaining recovery, signs and zero, equality/ineligibility boundaries, flag attacks, SOFA identity/hash binding, raw unclipped domain, observed flags, analytic ICU-time values and inverse, fractional time, positive/timezone contracts, feature-label separation, exact schema order, and both generated manifests.

The focused Phase-8 suite passes `17 passed`. The final integrated suite passes `1186 passed, 2 skipped` in 17.72 seconds, and `compileall` succeeds across `src`, `scripts`, and `tests`. The two pre-existing skips are optional Captum and SHAP/XGBoost integrations whose packages are absent. The governed system-evidence audit also passes with 17 indexed outputs and zero orphans.

## Boundaries and handoff

Phase 9 receives the same canonical identities, partial Phase-8 targets, and explicit recovery blocker. It may add production support events/state and the organ-support target, and may unblock final SOFA/recovery execution after review. Phase 8 generated no support state, eligibility, label, censor reason, or prevalence.

Phase 10 will receive historical inputs plus completed targets only after Phase 9. It alone assigns subject splits, fits feature transforms and recovery target scaling, and computes class weights. Phase 8 performed none of those actions.

Any target-formula, horizon, eligibility, SOFA, episode-boundary, or feature change requires versioned change control and full downstream regeneration. Phase 8 makes no final-data or clinical-validity claim.
