# Pulkit Phase 9 — Integrated Gradients Review

## 1. Pulkit Phase 9 summary

A Captum-only Integrated Gradients adapter framework is implemented. Captum is unavailable in the current environment, so attribution execution and completeness evidence remain blocked; no substitute algorithm was introduced.

## 2. Repository state before changes

Phase 8 supplied model/hash/cutoff-bound routing. GRU models consume `CanonicalBatch`; values and observation masks are mandatory, TSLO is configuration-dependent, and production statics remain blocked. The pre-change suite passed 732 tests.

## 3. Captum dependency status

`import captum` fails with `ModuleNotFoundError`. The repository has no dependency/environment manifest through which to approve and pin it. Execution raises `BLOCKED — CAPTUM DEPENDENCY REQUIRED`.

## 4. Files created

`src/explainability/ig.py`, a synthetic development config, four focused test modules, a shared test helper, and this review record.

## 5. Files modified

The Phase-8 internal result can now retain optional adapter details. Stale phase-boundary assertions were advanced to prohibit Phase 10 rather than Phase 9.

## 6. IG adapter architecture

The adapter validates context, target, prepared input, feature order, baseline identity/shape, and finite tensors before lazy-loading `captum.attr.IntegratedGradients`. There is no fallback implementation.

## 7. Selected-GRU / model-hash binding

Only `family=gru` is accepted. Context and loaded-model hashes must agree. The Phase-8 router additionally binds exact selected predictor object and manifest identity.

## 8. Prepared-input integration

The exact prepared `CanonicalBatch` is reused. No builder, preprocessor, label, eligibility, raw timeline, or future event is accessed.

## 9. Model forward wrapper

The wrapper reconstructs only the immutable batch shell around attributed tensors while preserving padding and metadata. Its full output equals ordinary model forward on deterministic fixtures.

## 10. Baseline policy / freeze status

`SYNTHETIC_IG_BASELINE_V1` explicitly zeros each attributable tensor while leaving padding fixed/non-attributed. It is development-only and rejected in real scope. The real baseline remains unlocked.

## 11. IG config / integration settings

`SYNTHETIC_IG_CONFIG_V1` declares 32 Gauss-Legendre steps, no internal batch size, and convergence delta. These are synthetic development settings, not a scientific freeze.

## 12. Recovery 24h target result

The descriptor validates `delta_sofa_24h` against the model's explicit horizon order. Wrapper target selection passes; actual Captum attribution is blocked by the missing dependency.

## 13. Recovery 48h target result

The descriptor independently validates `delta_sofa_48h`; no chaining or combined output is used. Actual Captum attribution is blocked.

## 14. ICU-time target/domain status

Synthetic scalar descriptor support exists. Real log-space versus reported-hours attribution remains **UNLOCKED — ICU-TIME IG OUTPUT DOMAIN MUST BE FROZEN**.

## 15. Organ-support target/domain status

Synthetic scalar descriptor support exists. Real logit versus sigmoid-probability attribution remains **UNLOCKED — SUPPORT IG OUTPUT DOMAIN MUST BE FROZEN**. Isotonic calibration and thresholds are excluded.

## 16. Temporal attribution shape

The contract preserves `[B,8,F]` signed attribution separately for values, observation masks, and optional TSLO. Shape checks are strict and do not hard-code production `F`.

## 17. Static attribution status

Aggregation preserves statics as a separate vector and never repeats them over timesteps. Actual project GRU statics remain blocked by the upstream model/input contract.

## 18. Feature-schema mapping

Dynamic attribution uses the prepared batch's ordered feature names and matching schema version. Observation-mask and TSLO identities are explicitly prefixed. Production human-readable names remain blocked with the feature handoff.

## 19. Analytic sanity test

The deterministic two-output linear-style fixture and expected target outputs are verified. Captum attribution against that fixture is skipped until Captum is available.

## 20. Completeness diagnostic

The adapter requests Captum convergence delta and labels it solely as a numerical completeness diagnostic. The executable completeness assertion is currently skipped, not claimed passed.

## 21. Reproducibility result

Config/baseline hashes and forward results are deterministic. Repeated Captum attribution reproducibility remains blocked by the missing dependency.

## 22. Absolute-contribution aggregation

A pure utility sums absolute attribution by exact input feature while retaining raw signed tensors. It returns all features without normalization or top-k truncation.

## 23. Model-immutability / no-training result

The adapter snapshots model state, switches temporarily to evaluation mode, restores prior mode, and verifies parameters are unchanged. No optimizer, training callback, scaler fit, calibration fit, or threshold search exists in the module. Runtime verification awaits Captum.

## 24. Future-data nonaccess result

The interface accepts only the prepared batch and already-computed raw output. Tests contain no labels or eligibility requirement, and source audits find no raw-history or final-test access.

## 25. Noncausal terminology audit

Outputs contain feature identities and attribution magnitudes only. Documentation describes model contribution, not clinical cause, treatment, confidence, or prognosis.

## 26. Phase-5/8 integration result

The adapter implements the Phase-8 protocol and returns Phase-5-compatible items plus internal reproducibility details. Router integration can execute once Captum and explicit targets are supplied. The public schema is unchanged.

## 27. Tests added

Tests cover dependency blocking, GRU-only enforcement, hash/cutoff checks, wrapper equality, explicit targets, baseline/config identity, shape attacks, finite inputs, signed/absolute aggregation, static separation, and Phase-8 regressions. Captum-dependent execution is explicitly skipped.

## 28. Exact commands executed

```text
PYTHONPATH=src:tests python3 -m pytest -q
python3 -c "import captum; import torch"
rg -n "class .*GRU|def forward|..." src/models src/training configs docs tests
rg -n "IG BASELINE|Integrated Gradients|..." configs docs src artifacts tests
PYTHONPATH=src:tests python3 -m pytest -q <Phase-9 and Phase-8 focused tests>
PYTHONPATH=src:tests python3 -m compileall -q src tests
```

## 29. Test results

Final counts are recorded in the completion response. Captum-dependent tests are reported as skipped rather than passed.

## 30. Real selected-GRU explanation status

**BLOCKED — REAL SELECTED MANIFEST, MODEL, PHASE-7 INPUT, CAPTUM, BASELINE, FEATURE CONTRACT, AND OUTPUT DOMAIN REQUIRED.**

## 31. Blocked / unlocked parameters

Blocked/unlocked: real baseline, Captum dependency pin, final integration settings/tolerance, recovery standardized versus inverse-transformed domain, ICU-time domain, support domain, production statics, feature naming, and display normalization/top-k policy.

## 32. Vedant review notes

Verify exact model/hash, recovery order, task output domains, no target-scaler/refit, no training, prepared-input reuse, forward equality, baseline/config provenance, synthetic isolation, and continued G3 blocking.

## 33. Sanskruti compatibility notes

Verify feature order/names, value-mask-TSLO identities, padding behavior, static representation, aggregation semantics, no alternate preprocessing, and non-causal language.

## 34. G3 explanation dependency status

IG framework code exists, but Captum execution and scientific freeze are incomplete. TreeSHAP is absent. **G3 explanation dependency remains blocked.**

## 35. Definition-of-done checklist

Framework contracts, validation, target descriptors, synthetic baseline/config, wrapper, aggregation, blockers, and tests are implemented. Captum execution, completeness, runtime immutability, real artifacts, scientific freezes, and reviews remain incomplete and are not claimed.

## 36. Readiness for Pulkit Phase 10

The router and IG contract can hand off structurally, but Phase 10 must not change selected-family routing. TreeSHAP remains entirely unimplemented.
