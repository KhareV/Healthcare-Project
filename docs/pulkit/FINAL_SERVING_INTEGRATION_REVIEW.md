# Vedant Finalization — Stage 4 Review: Final Serving Integration

Binds the frozen Stage-3 outputs into the existing Pulkit serving stack: real
selected-model loading, real TreeSHAP and Integrated Gradients, a real
FastAPI composition root, and a real train/validation-only retrospective
replay dashboard. No model training, calibration, or threshold selection
occurred in Stage 4. Final-test data was never accessed.

## Mixed-family routing (unchanged from Stage 2/3 selection)

| Task | Family | Candidate | Explanation |
|---|---|---|---|
| Recovery | xgboost | `xgb-recovery-014` | tree_shap |
| ICU-like episode remaining time | gru | `gru-icu-time-026` | integrated_gradients |
| Organ-support initiation | xgboost | `xgb-support-024` | tree_shap |

LSTM is not serving-eligible; the Stage-4 resolver (`src/serving/real/bundle.py`)
fails closed if any task's `family == "lstm"`.

## Artifact resolver

`serving.real.bundle.resolve_stage4_bundle(root)` does **not** reuse
`serving.artifacts.load_serving_bundle`'s `scope="real"` path — that path
assumes a manifest shape (`serving_ready=True`, per-task
`serving_authorized`/`output_contract` sidecar metadata) the actually
accepted, G3-frozen `selected_models_calibrated_v1` manifest deliberately
never has (`serving_ready` stays `false` so its hash, and G3, never change).
Instead the resolver:

1. Calls `vedant_infra.g3.validate_g3_marker(g3_path, root, expected_scope="real")`,
   which independently re-verifies G1, Phase-14 handoff, Stage-2 family
   selection, LSTM sensitivity, split, preprocessor, recovery scaler, support
   class weight, `selected_models_v1`, support calibrator, support threshold,
   and code commit against **live** file state — failing closed on any drift.
2. Independently re-verifies every per-task model file's hash
   (`recovery_24h.json`, `recovery_48h.json`, `organ_support.json`,
   `model.pt`) against the hashes recorded inside the already-verified
   `selected_models_v1.json`.
3. Loads models only after every hash above has passed, via the existing
   `models.xgb_canonical.load_xgb_bundle` and
   `models.gru_icu_time.load_icu_time_bundle` (zero new loading logic).

No latest-file discovery, no directory scanning, no runtime family
switching, no fallback to a smoke or synthetic fixture model.

## Canonical input path (reused, not rebuilt)

`serving.real.history.Stage4CanonicalInputProvider` wraps the existing,
unmodified `serving.preprocessing.CanonicalHistoryInputProvider` (the same
class Phase 14's `pulkit_receiver_acceptance` used to prove offline/serving
feature equivalence) — it only adds the request cutoff to the returned view
so the frozen preprocessor's output can carry it forward for GRU/TreeSHAP
context binding. `serving.real.predictors.RealFrozenPreprocessor` wraps the
existing, unmodified `preprocess.synthetic_phase10.FrozenSyntheticFeaturePreprocessor`
transform-only; the XGBoost flattened row is the identical
`values, observation_mask, tslo_hours, padding_mask, statics` concatenation
`data.xgb_canonical.flatten_row` uses (block order confirmed against the
frozen flat feature map). Preprocessor fit calls: **0**.

## Test-partition guard

`serving.real.history.NonTestHistoryProvider` wraps the real
`CanonicalTimelineHistoryProvider` and refuses (`SealedTestSubjectError`,
a subclass of the existing `UnknownStayError` so a sealed subject is
indistinguishable from an unknown one) any stay whose subject is in the
`test` split, before the truncator or feature builder ever see it. This is
the only history path Stage-4 serving/API/dashboard/replay uses, so ordinary
requests cannot open final-test data. Verified in
`tests/test_stage4_real_serving.py` at both the pipeline and live-API layer.

## Recovery, ICU, support serving

- Recovery: `RecoveryXGBoostTaskPredictor` runs both frozen boosters, applies
  the frozen `RecoveryTargetScaler` inverse transform (already built into
  `models.xgb_canonical.RecoveryXGBBundle.predict`), and reconstructs
  `clip(SOFA(t) + Δ, [0,24])` per horizon via the existing
  `serving.recovery.RecoveryServingPostprocessor` / `reconstruct_absolute_sofa`
  — unmodified. 24h and 48h stay independent (never summed together).
- ICU: `ICUTimeGRUTaskPredictor` returns the raw log1p prediction; the
  existing `serving.postprocessing.CanonicalICUTimeServingPostprocessor`
  applies the unmodified `remaining_icu_hours_from_log_prediction`
  (`expm1(clamp_min(x,0))`).
- Support: raw XGBoost probability → frozen isotonic calibrator
  (`evaluation.calibrate.load_calibrator_artifact` /
  `apply_support_calibrator`, unmodified) → calibrated probability. The
  frozen threshold (`0.3608842426222367`, `>=`) is bound into the serving
  bundle and available via `/model-metadata`; it is not treated as the
  primary evaluation metric.
- Current SOFA: the unmodified `data.synthetic.sofa_provider.SyntheticCurrentSOFAProvider`
  (the same class Phase 14 proved against accepted `sofa_at(t)`), wrapped
  only to normalize the request cutoff's UTC spelling between two
  pre-existing, differently-strict ISO-8601 parsers
  (`serving.history` accepts `+00:00`; `data.synthetic.validation.parse_utc`
  requires literal `Z`) — no SOFA logic changes.

## Explainability

Both adapters below are the pre-existing `explainability/tree_shap.py` and
`explainability/ig.py` modules, additively extended to a real (non-synthetic)
scope — their prior synthetic-only code paths and tests are unchanged.

### TreeSHAP — recovery (independent per horizon)

`serving.real.explanations.RealTreeShapAdapter` runs `shap.TreeExplainer`
(shap 0.49.1) separately on each frozen booster
(`config.model_output="raw"`, `feature_perturbation="tree_path_dependent"`),
then maps SHAP values and the base value through the frozen affine
`RecoveryTargetScaler` (`shap * scale`, `base * scale + mean`) into original
DeltaSOFA-point units — exact under an affine transform, predictions
themselves are never altered. Feature identities are read directly from the
frozen `synthetic_xgb_flat_feature_map_v1` entries (bin/base-feature/channel
recoverable per `bin{n}__{feature}__{channel}`), not parsed heuristically.

### TreeSHAP — support

Explains the selected XGBoost model's raw margin output (pre-calibration),
using `output_margin=True` for the additivity check rather than the
calibration-input probability — required because shap's `tree_path_dependent`
perturbation only supports `model_output="raw"`. Every explanation's
`details` documents this explicitly: *"This TreeSHAP explanation decomposes
the selected XGBoost support model's raw margin output before isotonic
calibration... does not decompose the calibrated probability."*

### Integrated Gradients — ICU GRU

Captum 0.8.0, `IntegratedGradients`, `method="gausslegendre"`, `n_steps=64`,
`internal_batch_size=None`, convergence delta reported. Attribution target:
the postprocessed public prediction (`remaining_icu_hours_postprocessed_v1`)
via a differentiable wrapper that applies the same frozen
`remaining_icu_hours_from_log_prediction` transform after the model output —
provably the same function `.predict()` uses, not a second one.

Baseline (`RealICUBaselineProvider`): zero for the value and
observation-mask channels (the trained mean-normalized reference, and
"not observed"); the frozen `configs/synthetic/feature_schema_v2.json`
`tslo.no_observation_sentinel` (**54.0 hours**, not z-scored) for TSLO — the
value that channel already uses to mean "never observed," not zero, which
would instead mean "just observed." Static channel: zero (trained mean age,
"no category selected"). Padding stays a fixed, non-attributed forward
argument.

A real latent bug surfaced and fixed while wiring the real GRU through this
framework for the first time: Captum batches every interpolation step into
one forward call (batch dimension = `n_steps`), but `padding_mask` is not an
attributable input and stayed at its original batch-of-one shape, which the
real `GRUEncoder._validate_batch`'s strict shape check rejected (the
synthetic test fixture's toy model never hit this because it does not call
that validator). Fixed by expanding `padding_mask` to match inside
`GRUForwardWrapper._batch` — a correctness fix, not a scope change.

A second real bug: `ICUTimeGRUTaskPredictor` is itself the `nn.Module`
identity bound into the router (needed so `.predict()` and Captum's
differentiable forward share one object), but `nn.Module` starts in
*training* mode by default; the IG adapter restores whatever training mode
it found before explaining. Without setting this wrapper to `.eval()` at
construction, every explained prediction re-enabled dropout for all
subsequent predictions on the same pipeline — a real nondeterminism bug,
caught by comparing two direct `pipeline.predict()` calls to the same
request. Fixed.

`pipeline.py`'s `explanation_target` was hardcoded to `None` (never
exercised by the synthetic mock adapters, which ignore it); real adapters
require a non-`None`, task-matching target. Added an optional,
backward-compatible `explanation_targets` mapping to
`PredictionPipeline.__init__`/`.predict()` (default `{}`, preserving prior
behavior for every existing caller/test).

## Prediction-schema fix

`serving.prediction_schema._validate_metadata` rejected any `split_version`
containing the lowercase substring `"synthetic"` when `synthetic=False` —
but the accepted scientific project scope is itself a synthetic cardiac
benchmark, so the real, G3-frozen split version (`synthetic_split_v2`)
legitimately contains that word without being a Phase-6 mock placeholder.
Changed that one field to the same unmistakable-placeholder convention
(`SYNTHETIC_` uppercase prefix) `feature_version`/`label_version` already
correctly use; `manifest_version`'s check was left as-is (already correct in
both directions). Verified against the full existing synthetic-fixture test
suite before and after.

## FastAPI

`api/real_app.py::build_real_app(root)` resolves the real bundle and calls
the existing, **unmodified** `api.main.create_app(pipeline=...)` — every
endpoint (`/predict`, `/health`, `/model-metadata`), error handler
(404/422/503), and metadata cross-check was already fully implemented and
required zero changes. Verified: API response == direct
`pipeline.predict()` response, byte-for-byte, for the same request.

## Retrospective sequential replay / dashboard

`dashboard.catalog.DashboardCatalog` and `dashboard.app.create_dashboard_app`
are reused unmodified; only the `scope != "synthetic"` / `"SYNTHETIC_"`-prefix
gates were relaxed to also accept `scope="real"` with the cohort's own
identifiers (`dashboard/catalog.py`). `dashboard/real_catalog.py` builds a
small, deterministic demo catalog directly from the accepted cohort/timeline
artifacts, filtered against the accepted split *before* any other step
(never a sealed test subject). `dashboard.app.create_real_dashboard_app`
wires `dashboard.api_client.HTTPDashboardAPIClient` (genuine HTTP calls, no
precomputed lookup) against the live API. Verified against a real `uvicorn`
server in a background thread (`tests/test_stage4_dashboard.py`): the
rendered page contains the `RETROSPECTIVE SEQUENTIAL REPLAY` banner, and the
dashboard's own recomputed prediction is byte-identical to a direct
`/predict` call for the same stay/cutoff.

## Environment / dependencies

`shap` and `captum` were missing from this development environment; no
homemade fallback was implemented. Installed `shap==0.49.1` and
`captum==0.8.0` (numpy pulled down `2.0.2`→`1.26.4` via numba's pin — the
full regression was re-run clean after). Recorded in
`artifacts/serving/stage4_environment_lock_v1.txt`
(exact-pinned, referenced by hash from the final serving bundle) and in the
refreshed `observed_environment_phase15.json` development snapshot. This is
a serving-specific lock, distinct from the separate, still-`BLOCKED`
repository-wide training-reproduction lock
(`configs/reproducibility_v1.json` / `tests/test_environment_lock.py`),
which Stage 4 did not touch.

## Hostile compatibility tests

`tests/test_stage4_real_serving.py` tampers a fresh repository copy (cheap
APFS copy-on-write clone) per case and asserts `resolve_stage4_bundle` fails
closed *before* loading any model: wrong `selected_models_v1` hash, inactive
G3, changed feature schema, changed preprocessor, changed recovery scaler,
changed support calibrator, changed threshold, wrong recovery/ICU/support
model file, and an LSTM family substituted into the manifest. None repair
the mismatch automatically.

## Governance state (unchanged, re-verified live)

- `selected_models_v1.json`: unchanged, `3d4fb0b1e0341b0ebe29a54e9581bc47d3b80c536d3ee22f830276b4631fbb0b`
- G3: `G3_ACTIVE`, unchanged, `d94d49b0a438027842dbbf1c6130404297230f59b18bc863320ac753db770564`
- Final-test access state: `AUTHORIZED_NOT_RUN`, `test_accessed=false`,
  `FINAL_TEST_ACCESS_CONSUMED` absent throughout Stage 4.

## New Stage-4 artifacts

- `artifacts/serving/final_serving_bundle_v1.json` — `SERVING_BUNDLE_FROZEN_PRE_TEST`
- `artifacts/acceptance/stage4_serving_integration_freeze_v1.json` — `STAGE4_SERVING_INTEGRATION_FROZEN`
- `artifacts/serving/stage4_environment_lock_v1.txt`

## Stage-5 readiness

Stage 4 performed zero model training/search/calibration and zero final-test
access. Serving is fully wired (resolver, mixed-family prediction, real
TreeSHAP/IG, FastAPI, replay dashboard) and provably equivalent across
direct/API/dashboard call paths on train/validation-safe data only. Stage 5
may now perform the one-time final-test evaluation.
