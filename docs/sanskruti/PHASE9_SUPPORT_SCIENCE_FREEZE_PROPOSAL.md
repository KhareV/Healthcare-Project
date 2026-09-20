# Phase 9 Support-Science Freeze Proposal

## Status and warning

**PROPOSED_NOT_AUTHORIZED.** This package prepares a scientific decision freeze; it does not freeze or authorize anything. No support interval, organ-support label, prevalence result, final cohort, split, preprocessor, or model was generated.

The support hazards, durations, rates, and intervention effects below are project-designed synthetic benchmark parameters. They are not estimates of real treatment propensity, real cardiac treatment rates, or causal treatment effects. Any later model performance demonstrates pipeline methodology on synthetic data, not bedside validity.

## Audit outcome

The pre-change repository was clean at `e5bdb0ca609cbbb01735d8466d59cabfa264537e`. The baseline command was:

```text
PYTHONPYCACHEPREFIX=$(mktemp -d) PYTHONPATH=src:tests:. python3 -m pytest tests -q -rs -p no:cacheprovider
```

It passed `1186` tests, skipped the two existing optional Captum and SHAP/XGBoost integrations, failed none, and completed in 17.98 seconds.

The audit confirmed the Phase-2 architectural mechanism but found null support hazards, random effects, rates, intervention effects, and vocabulary; a draft non-authorizing Pulkit dictionary; zero generated support rows; unresolved endpoint policies; and no authorized final cohort. The complete 20-item inventory is `configs/synthetic/support_unresolved_inventory_v1.json`, SHA-256 `eebc994d7501c33422b2621747159f2e9542731c4ea72bb8fdaf4085e59c2731`.

## Proposal artifacts

| Artifact | Purpose | SHA-256 |
|---|---|---|
| `configs/synthetic/support_process_v1.proposed.yaml` | Complete simulation-design proposal | `d3d98b2b2c52863b43e773c449e3137e2ab6cb167b874558188a186a82cef8ce` |
| `configs/event_dict_v2.proposed.yaml` | Synthetic, non-MIMIC source/endpoint mapping proposal | `9dd1cf78a7041801bf804c495147f2272ef2ad8edc2047c2709d906fb07c9abc` |
| `configs/synthetic/support_unresolved_inventory_v1.json` | Machine-readable unresolved-decision inventory | `eebc994d7501c33422b2621747159f2e9542731c4ea72bb8fdaf4085e59c2731` |

Historical authorities remain byte-identical: `synthetic_generator_v1.yaml` hashes to `5f7d3f73572a34282c214860f1c0c656b18727a16fa0c4687caf071f690a7484`; `event_dict_v1.yaml` hashes to `717f6183ad8014602b3cc32eeaa3a810eab8a26a19937bda8e214200bb2f9620`.

## Complete decision table

| ID | Decision | Current status | Proposed resolution | Source type | Rationale | Downstream impact | Human approval? | Final status |
|---|---|---|---|---|---|---|---|---|
| SUP-001 | Initiation hazards | Null | Hourly logistic hazards using existing systemic plus cardiac/respiratory latent coordinates and six-hour recent mean | PROJECT_SYNTHETIC_DESIGN | Compact, stateful, interpretable, not target-first | Events, labels, SOFA, recovery | Yes | PROPOSED_FOR_EXPLICIT_TEAM_APPROVAL |
| SUP-002 | Cessation | Null | Hourly logistic OFF hazard with current burden and time-on-support | PROJECT_SYNTHETIC_DESIGN | Permits continuation, cessation, and repeats | State and endpoint | Yes | PROPOSED_FOR_EXPLICIT_TEAM_APPROVAL |
| SUP-003 | Subject heterogeneity | Null | Independent component random intercepts `Normal(0,0.35)` from support child stream | PROJECT_SYNTHETIC_DESIGN | Minimal stable heterogeneity | Event timing | Yes | PROPOSED_FOR_EXPLICIT_TEAM_APPROVAL |
| SUP-004 | Rate distributions | Null | Agent-specific bounded lognormal distributions in `ug/kg/min` | PROJECT_SYNTHETIC_DESIGN | Stochastic rates span SOFA categories without quota tuning | SOFA/recovery | Yes | PROPOSED_FOR_EXPLICIT_TEAM_APPROVAL |
| SUP-005 | Intervention effects | Null | Two-hour-lagged modest bounded latent drift shifts with response multiplier | PROJECT_SYNTHETIC_DESIGN | Implements Phase-2 feedback without deterministic recovery | All future physiology | Yes | PROPOSED_FOR_EXPLICIT_TEAM_APPROVAL |
| SUP-006 | Vasoactive vocabulary | Null | Norepinephrine, epinephrine, dopamine; dobutamine SOFA-only | CLINICAL_DEFINITION plus PROJECT_SYNTHETIC_DESIGN vocabulary | Separates endpoint pressors from SOFA inotrope | Endpoint/SOFA | Yes | PROPOSED_FOR_EXPLICIT_TEAM_APPROVAL |
| SUP-007 | Units/activity/duplicates | Unresolved | Canonical `ug/kg/min`; positive finite rate; missing/zero active rate invalid; duplicates fail | CLINICAL_DEFINITION and ENGINEERING_ONLY | Matches Phase-6 unit; fails closed | State/SOFA | Yes | PROPOSED_FOR_EXPLICIT_TEAM_APPROVAL |
| SUP-008 | Respiratory mapping | MIMIC-bound unresolved | Synthetic `INVASIVE`, `NON_INVASIVE`, `HFNC`, `ORDINARY_OXYGEN`; only invasive qualifies | FROZEN project taxonomy with proposed symbolic profile | Preserves invasive distinction without fake MIMIC provenance | State/endpoint/SOFA | Yes | PROPOSED_FOR_EXPLICIT_TEAM_APPROVAL |
| SUP-009 | Onset equals outtime | Unresolved | Such a production row is ambiguous and invalid | ENGINEERING_ONLY | Avoids inventing event ordering | Endpoint | Yes | PROPOSED_FOR_EXPLICIT_TEAM_APPROVAL |
| SUP-010 | Unknown state | Unresolved | Fail closed; unknown is never OFF | FROZEN fail-closed principle, proposed production rule | Prevents false negative/at-risk classification | State/endpoint/SOFA | Yes | PROPOSED_FOR_EXPLICIT_TEAM_APPROVAL |
| SUP-011 | Audit reasons | Internal nonfinal | Versioned synthetic audit vocabulary retaining existing four dispositions | ENGINEERING_ONLY | Stable audit, not a model class | Label artifact | Yes | PROPOSED_FOR_EXPLICIT_TEAM_APPROVAL |
| SUP-012 | SOFA alignment | Unresolved | Shared vocabulary/interval versions and dictionary hash; no threshold changes | CLINICAL_DEFINITION | One interpretation for endpoint and SOFA | SOFA/recovery | Yes | PROPOSED_FOR_EXPLICIT_TEAM_APPROVAL |
| SUP-013 | Root seed | Null | `20260921` | ENGINEERING_ONLY | Memorable project-date seed, distinct from fixture seeds and 42 | Every generated artifact | Yes | PROPOSED_FOR_EXPLICIT_TEAM_APPROVAL |
| SUP-014 | Cohort size | Null | 2,000 independent subjects | PROJECT_SYNTHETIC_DESIGN | Proposed computational benchmark size; not a power or prevalence claim | All artifacts/evaluation | Yes | PROPOSED_FOR_EXPLICIT_TEAM_APPROVAL |
| SUP-015 | Regeneration | Supports always empty | Full deterministic Phase-3 replay with support integrated into latent simulation | ENGINEERING_ONLY | Prevents mixed provenance once support effects alter physiology | All downstream artifacts | Yes | PROPOSED_FOR_EXPLICIT_TEAM_APPROVAL |
| SUP-016 | Interval convention | Closed-open already specified | `[start,end)`; active at start and immediately before end, off at end | FROZEN_FROM_EXISTING_PROJECT_AUTHORITY | Matches Phase 2 and Pulkit synthetic engine | State/endpoint/SOFA | Confirm only | FROZEN_FROM_EXISTING_PROJECT_AUTHORITY |
| SUP-017 | Respiratory categories | Already specified | Retain four Phase-2 categories and hard-fail unknown | FROZEN_FROM_EXISTING_PROJECT_AUTHORITY | Existing explicit contract | State/endpoint/SOFA | Confirm only | FROZEN_FROM_EXISTING_PROJECT_AUTHORITY |
| SUP-018 | Composite endpoint | Already frozen | Preserve OFF-to-ON in `(t,t+24h]`, continuation and censoring semantics | FROZEN_FROM_EXISTING_PROJECT_AUTHORITY | Project Scope v2/Pulkit | Labels | No redesign | FROZEN_FROM_EXISTING_PROJECT_AUTHORITY |
| SUP-019 | Final latent scale | Final config absent | Numerical proposal is conditional on approval of the final latent-process scale | PROJECT_SYNTHETIC_DESIGN | Hazard coefficients operate on dimensionless latent coordinates | Generator/all downstream | Yes | BLOCKED_REQUIRES_EXTERNAL/AUTHORITATIVE_EVIDENCE |
| SUP-020 | Human reviews | Pending | Sanskruti source, Pulkit endpoint, Vedant receiver reviews | ENGINEERING_ONLY governance | Agent output cannot constitute team approval | Authorization | Yes | BLOCKED_REQUIRES_EXTERNAL/AUTHORITATIVE_EVIDENCE |

## Preserved endpoint

The endpoint remains new qualifying vasopressor or invasive mechanical-ventilation OFF-to-ON initiation in `(t,t+24h]`. An event at `t` defines baseline; an event exactly at `t+24h` is included. Continuation and a second pressor while pressor state is already ON are not initiation. One OFF component remains at risk when the other is ON; both ON is not at risk. Early exit without initiation is censored, while a known positive before exit remains positive.

No endpoint behavior was inferred from prevalence because no support events or prevalence were produced.

## Respiratory taxonomy

`INVASIVE` means invasive mechanical ventilatory support through an invasive airway. `NON_INVASIVE` includes noninvasive positive-pressure modes such as noninvasive CPAP/BiPAP; `HFNC` and `ORDINARY_OXYGEN` remain separate. Only `INVASIVE` turns Pulkit's invasive state ON and supplies the invasive context required by respiratory SOFA. Unknown categories fail rather than becoming OFF.

The symbolic names reuse the existing Phase-2/Pulkit internal vocabulary. They are not MIMIC concepts and carry no MIMIC table/item provenance.

Generation maintains one active flag per respiratory category, not a single mutually exclusive generic respiratory flag. A start candidate draws one category; an already-active selected category is continuation, while another category may start concurrently. Consequently noninvasive support at `t` does not block a later invasive start, preserving Pulkit's required noninvasive-to-invasive OFF-to-ON case.

## Vasoactive roles and units

| Concept | Canonical name | Endpoint qualifying | SOFA relevant | Rate required | Unit | Rationale/status |
|---|---|---:|---:|---:|---|---|
| Norepinephrine | `norepinephrine` | Yes | Yes | Yes | `ug/kg/min` | Proposed explicit synthetic vocabulary |
| Epinephrine | `epinephrine` | Yes | Yes | Yes | `ug/kg/min` | Proposed explicit synthetic vocabulary |
| Dopamine | `dopamine` | Yes | Yes | Yes | `ug/kg/min` | Proposed explicit synthetic vocabulary |
| Dobutamine | `dobutamine` | **No** | Yes | Yes | `ug/kg/min` | SOFA inotrope, not silently reclassified as endpoint vasopressor |

The canonical unit is inherited from the Phase-6 SOFA specification. The generator emits it directly; magnitude-based inference and cross-agent equivalence are prohibited. A rate in an active interval must be finite and strictly positive. Missing rate, zero-rate active rows, unknown unit, and unknown concept fail closed.

Generation likewise maintains per-agent active flags. A selected already-active agent continues without a new interval, while a different agent may overlap. Pulkit then collapses all active qualifying agents to one binary endpoint state, so a second qualifying pressor during an already-ON state cannot become a new endpoint initiation. Dobutamine may overlap without itself turning the endpoint pressor state ON.

## Proposed simulation model

The actual latent process has a one-hour step and five dimensionless coordinates: systemic severity, cardiac stress, respiratory burden, renal burden, and neurologic burden. It uses subject-local `PCG64` streams, with support reserved as component code 7. The proposal therefore uses one-hour discrete logistic hazards.

Vasopressor initiation logit:

```text
-4.60 + 0.55*systemic + 0.75*cardiac
      + 0.20*mean(systemic over prior 6h)
      + subject_effect + hourly_innovation
```

Invasive/respiratory-support initiation logit:

```text
-4.80 + 0.55*systemic + 0.90*respiratory
      + 0.20*mean(respiratory over prior 6h)
      + subject_effect + hourly_innovation
```

Subject effects are independent `Normal(0,0.35)` per process; hourly innovations are `Normal(0,0.15)`. These are synthetic design values. Reasonable sensitivity ranges are intercept `[-6.0,-3.5]`, contemporaneous loadings `[0.2,1.2]`, and recent-state loadings `[0,0.4]`. No prevalence calibration was run.

Cessation also uses an hourly logistic hazard. Vasopressor coefficients are intercept `-3.20`, systemic/card cardiac coefficients `-0.25/-0.25`, and time-on-support coefficient `0.055`. Respiratory coefficients are `-3.40`, `-0.25/-0.35`, and `0.050`. Minimum duration is one hour; the untruncated maximum is 72 hours; episode end truncates an active interval.

## Proposed rates

Rates use bounded lognormal draws and an hourly log-rate AR(1) coefficient of `0.85`, innovation SD `0.18`, and contemporaneous burden loading `0.12`:

| Agent | Bounds | Median | Log SD |
|---|---:|---:|---:|
| Norepinephrine | 0.01–0.50 | 0.08 | 0.60 |
| Epinephrine | 0.01–0.30 | 0.06 | 0.60 |
| Dopamine | 1–20 | 7 | 0.45 |
| Dobutamine | 1–15 | 5 | 0.40 |

These bounds structurally permit multiple existing cardiovascular SOFA categories without targeting category frequencies.

## Proposed intervention feedback

After a two-hour lag, vasopressor support shifts future hourly latent drift by `-0.015` systemic and `-0.025` cardiac; invasive ventilation shifts it by `-0.010` systemic and `-0.025` respiratory. A subject/episode response multiplier is drawn from a bounded lognormal distribution with median 1, log SD 0.25, and bounds `[0.5,1.5]`. Absolute shift is capped at 0.04 per latent coordinate per hour. Noninvasive categories have zero feedback in v1.

This is a modest stochastic simulation mechanism, not a causal claim. It never reads a target, split, eventual class, remaining time, total duration, or future outtime. No propensity or response multiplier is exported.

## Interval, duplicate, and overlap semantics

Intervals are `[start,end)`. State is ON at start and immediately before end, and OFF at end unless a same-concept adjacent interval is canonicalized into one continuous interval. Explicit ends are required; an end may equal episode outtime, but a start may not. Zero/negative duration and onset exactly at outtime are invalid.

Duplicate IDs and semantic duplicates with different IDs fail. Same-agent overlaps must be merged deterministically before export or fail. Different qualifying vasoactive agents may overlap and collapse to one binary ON state. Respiratory categories remain preserved; invasive state is ON whenever any invasive interval is active. Row order never resolves conflict.

## SOFA alignment

The endpoint and SOFA share `synthetic_support_vocabulary_v2_proposed`, `synthetic_support_closed_open_v1_proposed`, and the eventual dictionary hash. Cardiovascular SOFA consumes the four listed agent rates in `ug/kg/min`; the endpoint consumes only the three qualifying pressors. Respiratory SOFA and Pulkit both consume `INVASIVE`; NIV, HFNC, and ordinary oxygen remain excluded. No SOFA threshold or missingness rule changes.

## Regeneration and authorization plan

Option A—full replay—is required. The current Phase-3 generator creates latent physiology before returning an empty support table. Once approved support effects feed future latent drift, an interval-only extension would disagree with the physiology it should have influenced. Final generation must integrate support state into the hourly latent loop using the existing subject-local component-code-7 RNG, then regenerate raw data and every dependent Phase 4–9 artifact.

Proposed final engineering choices are root seed `20260921` and 2,000 independent subjects. The seed is not 42 or a fixture seed; the size is not the prohibited convenience size 100. The size is a benchmark/compute proposal, not a statistical-power or clinical-prevalence claim. Both require explicit approval.

After approval and promotion, the required order is: generator replay; raw/cohort/timeline validation; support adapter and dictionary validation; Pulkit states and labels; SOFA; Phase-7 feature rerun only if necessary for regenerated physiology (its schema remains unchanged); Phase-8 recovery/ICU targets; exact pre-split identity audit. Phase 10 remains out of scope.

## Compatibility/conflict audit

- Phase 2: compatible with one-hour latent process, Markov hazards, closed-open intervals, child stream 7, and delayed bounded effects.
- Phase 3: **implementation gap**—the orchestrator currently emits `[]` support rows and has no intervention feedback hook. Implementation follows only after approval.
- Phase 5: raw schema already represents typed support intervals; no schema redesign is proposed.
- Phase 6: concept/unit/category proposal matches the existing SOFA agents and invasive context. Its future-end handling still requires a cutoff-safe adapter during implementation.
- Phase 7: no support channels exist in frozen F=15, so none will be added. Full rerun is nevertheless needed if approved feedback changes physiology.
- Phase 8: formulas remain unchanged; recovery can be regenerated only after support-backed SOFA is valid.
- Pulkit: OFF/ON and composite algorithms remain unchanged. The loader currently hard-codes event dictionary v1 and a real/scientific block; after approval, the minimum compatibility work must add an explicitly authorized synthetic profile without weakening fail-closed checks.
- Governance: human reviews and final latent/population configuration remain genuine blockers; this proposal is not approval.

## Approval checklist

- [ ] Qualifying endpoint vocabulary: norepinephrine, epinephrine, dopamine.
- [ ] SOFA-only dobutamine distinction.
- [ ] Canonical `ug/kg/min` units and rate domains.
- [ ] Four-category respiratory taxonomy and invasive-only qualification.
- [ ] Closed-open interval and adjacency semantics.
- [ ] Duplicate and overlap policies.
- [ ] Onset exactly at outtime is invalid.
- [ ] Unknown state/category/unit fails closed and is never OFF.
- [ ] Shared SOFA/dictionary vocabulary and interval identity.
- [ ] Hourly initiation equations and coefficients.
- [ ] Subject random-effect and hourly-innovation distributions.
- [ ] Cessation equations and duration bounds.
- [ ] Agent rate distributions and AR(1) process.
- [ ] Lagged intervention-effect parameters.
- [ ] Final latent-process scale/config to which coefficients apply.
- [ ] Final root seed `20260921`.
- [ ] Final independent-subject count 2,000.
- [ ] Full Phase-3 deterministic replay strategy.
- [ ] Sanskruti source-semantics review.
- [ ] Pulkit endpoint-compatibility review.
- [ ] Vedant receiver-compatibility review.

## Exact approval boundary

The small decision set that still requires team action is: approve or revise (1) the vasoactive roles/ranges, (2) support hazard and cessation numbers, (3) intervention feedback, (4) tie/unknown/duplicate policies, (5) the final latent configuration, seed, and cohort size, and (6) the three ownership reviews. Until those decisions are recorded through versioned change control, production support generation and Phase 10 remain blocked.

## Verification

The focused proposal-validation suite passes `7 passed`. It proves complete non-authorizing schemas, absence of hidden null/TODO/TBD/draft markers in the proposed parameter sets, governance metadata for every numerical parameter group, unique vocabularies, explicit endpoint/SOFA roles, valid probability totals and numeric domains, explicit tie/unknown/duplicate policies, unchanged v1 hashes, the absence of newly generated support/label/final artifacts, and the full-replay pre-split plan. `compileall` succeeds.

The final integrated repository suite passes `1193 passed, 2 skipped` in 17.80 seconds. The two skips remain the pre-existing optional Captum and SHAP/XGBoost integrations. The governed system-evidence package was refreshed from a passing 1,187-test non-evidence run and audits successfully with 17 indexed outputs and zero orphans.
