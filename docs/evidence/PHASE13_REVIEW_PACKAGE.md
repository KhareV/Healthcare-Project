# Phase 13 review package

Status: **calibration and threshold framework implemented with synthetic fixtures only**.

Real execution is `BLOCKED — REAL SELECTED ORGAN-SUPPORT CLASSIFIER REQUIRED`.
No real selected support model, frozen validation prediction artifact, calibrator,
threshold, promoted serving manifest, registry lineage, or G3 marker exists.

## Implemented contracts

- Only one already-selected organ-support XGBoost/GRU model may enter calibration.
- The input artifact must contain raw uncalibrated validation probabilities and
  authoritative eligibility/labels; ineligible rows are ignored.
- Isotonic fitting records the implementation, explicit out-of-range policy,
  sample-weighting policy, selected model/prediction hashes, split/schema/label
  lineage, eligible population hash, and counts.
- Threshold search runs only on calibrated probabilities and calls the Phase-9
  stay-balanced weighted-confusion implementation. The comparator is `p >= t`.
- Calibrator and threshold artifacts are deterministic JSON with SHA-256 binding.
- Manifest promotion preserves all selected families, model identities, and
  explanation routes, and remains non-serving in synthetic mode.

## Pulkit review checklist

- Review calibration/threshold serialization and loading interfaces.
- Verify model-calibrator-threshold compatibility failures are fail-closed.
- Verify the manifest exposes the exact selected model, calibrator reference/hash,
  threshold reference/hash/value, and unchanged explanation routing.
- Verify no serving, PredictionPipeline, API, dashboard, or explanation computation
  was implemented in Phase 13.
- Before real execution, approve/freeze the remaining policies with the team.

## Unresolved before real execution

- `UNLOCKED ENGINEERING PARAMETER — ISOTONIC IMPLEMENTATION DETAIL`
- `UNLOCKED ENGINEERING PARAMETER — ISOTONIC CALIBRATION SAMPLE WEIGHTING`
- `UNLOCKED ENGINEERING PARAMETER — THRESHOLD SEARCH PROCEDURE`
- `BLOCKED — F1 THRESHOLD TIE POLICY REQUIRED`
- Real selected support model and raw validation prediction/label artifact
- Git commit provenance in a real Git checkout

No Phase 14 or Pulkit-owned serving work is included.
