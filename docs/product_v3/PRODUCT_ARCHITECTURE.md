# V3 product architecture

The effective release is defined by `artifacts/performance_v3_1/product_handoff.json`: all four heads use the frozen V3 bundle. The V3.1 candidate failed its development gate; privileged oracles are research diagnostics and have no serving path. The original SvelteKit shell and product workflows remain in use.

## Request path

`api.product_app:app --factory` constructs `serving.v3.backend.V3Backend` before accepting requests. The backend checks the V3.1 and V3 handoffs, model manifest and file hashes, and refuses startup on a mismatch. It composes the existing FastAPI product routes with the V3 predictor. There is no implicit V2 fallback. The canonical history builder truncates raw observations at the selected legal cutoff and uses the same eight six-hour bins as the scientific contract. The V3 runtime computes SOFA(t) for every legal cutoff; it does not derive feature availability from +24h target eligibility, preventing defect D1 from recurring.

The four heads are independent: +24h and +48h DeltaSOFA, remaining time in the current ICU-like episode, and 24h **new** organ-support initiation risk. Recovery +24h, ICU time, and support use XGBoost; recovery +48h uses CatBoost. Support probability is Platt calibrated, with the frozen alert threshold `0.3021480061849302`. TreeSHAP explanations are model-specific raw-output attributions, not causal claims or decompositions of the calibrated probability.

The existing response fields remain available alongside release, bundle, model-family, hash and data-revision metadata. Patient replay and showcase scenario cutoffs request real V3 inference. Research endpoints read saved artifacts; they neither train nor invoke the sealed final-test cohort.

## Records and identity

Clerk verifies ownership server-side for custom records and reports. Optional MongoDB persists profiles, conditions, encounters, observations, report metadata, audits and prediction snapshots. The snapshot key includes owner, encounter, cutoff, model bundle and data revision. Older V2 snapshots remain historical and labeled by their model release. A confirmed report candidate is validated as an observation only after explicit user selection of the encounter and relative hour; the runtime's next V3 forecast uses the updated source history. Unconfirmed extraction text is not a model feature. Conditions and report metadata in the longitudinal record are also not automatically model features.

Groq serves optional report extraction and Copilot commentary. Neither may change numerical predictions. Missing Groq or TTS leaves replay and V3 inference usable.

## Scientific evidence and limits

The primary sealed V3 result covers 1,500 independent synthetic subjects. Its authoritative metrics, denominators and CIs are in `artifacts/performance_v3/phase4/final_metrics_v3.json`. The V2-as-served comparison is operational; the repaired V2-recipe control is the modeling-method comparison. Historical V2-as-evaluated ICU/support metrics include [defect D1](../performance_v3/DEFECT_D1.md) and must be labeled compromised. V3.1's [latent-state diagnostic](../performance_v3_1/LATENT_STATE_DIAGNOSTIC.md) is privileged development evidence, never a deployable accuracy claim.

This is retrospective synthetic research software, with no clinical or external validation. Population bootstrap intervals are not individual prediction intervals. See [figure and table index](FIGURE_AND_TABLE_INDEX.md) for source hashes and export formats.
