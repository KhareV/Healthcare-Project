# Sanskruti Rebuild Phase 1 — Project Scope v2 Review

## Status

- Phase-1 governance artifacts: **IMPLEMENTED; TESTED**.
- `PROJECT_SCOPE_V2`: **DRAFT COMPLETE — TEAM FREEZE REQUIRED**.
- Human approvals recorded: **none**.
- Phase 2 or later work performed: **none**.
- Final data generated: **no**.
- Models trained or final test accessed: **no**.

## Deliverables

- Human scope: `docs/governance/project_scope_v2.md`
- Machine scope: `configs/governance/project_scope_v2.json`
- Requirement delta: `docs/governance/project_scope_v2_delta.csv`
- Contract impact: `docs/governance/project_scope_v2_impact.md`
- Agent context successor: `docs/CODEX_PROJECT_CONTEXT_V2.md`
- SHA-256 integrity manifest: `docs/governance/project_scope_v2_hashes.json`
- Semantic tests: `tests/test_project_scope_v2.py`

## Authorized scientific changes

1. MIMIC-IV → authorized final synthetic project data.
2. General ICU population → adult cardiac/heart-disease population domain.
3. MIMIC anchor-era split → a future reviewed synthetic subject split policy.
4. MIMIC provenance → generator/schema/config/seed/dataset provenance.
5. MIMIC source mappings → versioned synthetic severity/support event mappings.

Not changed: retrospective forecasting; prediction grid; 48h/eight-bin history; recovery, remaining-episode-time, and support tasks; eligibility/censoring; naive/XGBoost/GRU and LSTM sensitivity; information parity; exact-30 validation search; stay-balanced metrics/bootstrap; support calibration/threshold; PredictionPipeline; explanations; replay; final-test protection.

## Vedant review checklist

- [ ] Retained timestamp formula/boundaries exactly match `timestamp_spec_v1`.
- [ ] Synthetic split supersedes only era policy and retains subject isolation.
- [ ] Canonical tensor information contract remains valid.
- [ ] Recovery/remaining-time/support tasks are unchanged.
- [ ] Search/evaluation/selection/calibration governance is retained.
- [ ] Final-test isolation remains fail-closed.
- [ ] Generator provenance can enter registry/lineage without fabricated MIMIC fields.
- [ ] Engineering fixtures cannot be promoted to authorized final data.

## Pulkit review checklist

- [ ] Retrospective replay semantics and exact banner concept are retained.
- [ ] `PredictionPipeline` remains the only orchestration path.
- [ ] Support state/composite semantics are retained.
- [ ] Only MIMIC source mappings/provenance are superseded.
- [ ] Future synthetic events can feed existing state engines.
- [ ] `CurrentSOFAProvider` remains usable with synthetic provenance/six flags.
- [ ] `prediction_schema_v1` remains valid because tasks are unchanged.
- [ ] API/dashboard wording makes no clinical-validity or real-time claim.

## Sanskruti review checklist

- [ ] Final generator targets the adult cardiac/heart-disease domain, not silently HF-only.
- [ ] Explicit `intime`/`outtime` episode boundaries are generated.
- [ ] No trajectory class or other target-generating latent variable reaches inputs.
- [ ] Generator/schema/config/seed/dataset provenance is versioned and hashed.
- [ ] Canonical builder consumes Vedant's timestamps.
- [ ] Exact `(t-48h,t]`/8×6h boundaries are retained.
- [ ] Inputs support a versioned six-component 0–24 SOFA calculation.
- [ ] Support events satisfy Pulkit's state-engine input semantics.
- [ ] Preprocessing is unfitted until the split and fit on train subjects only.
- [ ] XGBoost receives only the canonical flattened view.

## Scope-amendment review required

Reviewers must either approve the complete package or record exact requested changes. Prose acknowledgement does not create a frozen state. Approval must not be inferred from passing tests, file presence, or later-phase activity.

## Reserved for Phase 2+

Cardiac subtypes/distribution, cohort size, latent dynamics, missingness, support frequency, episode durations, feature inventory, split parameters, generator seeds, SOFA input generation, package manager, and final environment remain unresolved by design.

## Stale-reference audit and disposition

The repository-wide audit separated references instead of rewriting scientific history:

1. **Historical records retained:** `docs/CODEX_PROJECT_CONTEXT_V1.md`, completed phase reviews, old evidence, and member handoff records remain evidence of the v1 baseline. The V1 context now points explicitly to V2.
2. **Active guidance amended:** `AGENTS.md`, `README.md`, and `RUNBOOK.md` now identify Project Scope v2 as active and distinguish final synthetic research data from engineering fixtures.
3. **Runtime dependencies deferred:** MIMIC-era logic and vocabulary in `src/data/real_adapter.py`, `src/data/schema.py`, `src/data/split.py`, `configs/split_spec_v1.yaml`, `configs/event_dict_v1.yaml`, label modules, registry/lineage fields, evidence generation, and serving history are implementation dependencies for later phases. Phase 1 deliberately does not patch them around an unfrozen contract.
4. **Candidate tree retained:** `candidate/` is historical/reference implementation material and was not changed.

## Verification evidence

- Pre-change full regression: **906 passed, 2 skipped**. Both skips were optional Captum/SHAP-XGBoost dependency paths.
- Phase-1 focused governance tests after initial implementation: **14 passed**.
- Final focused governance test: **15 passed**.
- Evidence/packaging/lineage repair subsets: **11 passed**, followed by **12 passed** after registry-chain propagation.
- Final full regression: **921 passed, 2 skipped** in 10.72 seconds. The two skips remain the optional Captum and SHAP/XGBoost integration paths; there were no failures.
- Syntax compilation: `python3 -m compileall -q src api dashboard tests` completed successfully.
- `docs/governance/project_scope_v2_hashes.json` was verified by the focused semantic suite.

Passing tests prove internal consistency of the draft; they do not constitute team approval or authorize Phase 2.
