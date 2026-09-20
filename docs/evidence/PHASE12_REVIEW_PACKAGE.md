# Phase 12 review package

Status: **selection framework implemented on synthetic evidence only**.

Real scientific selection is `BLOCKED — REAL VALIDATION SEARCH RESULTS REQUIRED`.
The repository contains six synthetic dry-run candidate manifests but no completed
scientific 30-candidate result sets, no valid real within-family winners, and no
real XGBoost model artifacts. Consequently, no `selected_models_v1.json`, task
selection artifact, registry decision, or G3 marker was created.

## Frozen comparison rules

| Task | Primary validation criterion | Direction |
|---|---|---|
| Recovery | Stay-balanced 24h MAE | minimum |
| Remaining ICU time | Stay-balanced weighted median absolute error in hours | minimum |
| Organ support | Stay-balanced AUPRC from raw uncalibrated probabilities | maximum |

Only XGBoost and GRU are eligible. Naive is reference-only and LSTM is
sensitivity-only. XGBoost routes to TreeSHAP metadata; GRU routes to Integrated
Gradients metadata. No explanations are computed in Phase 12.

## Pulkit review checklist

- Review `src/evaluation/select.py` and `selection_validation.py`.
- Verify mixed-family task routing and family-correct explanation metadata.
- Verify artifact hash and task/family/schema/split/preprocessor/label checks.
- Verify the organ-support calibrator and threshold remain null and explicitly
  pending Phase 13; selection-stage output is not serving-ready.
- Verify structural rejection of test-labelled sources and `test_accessed=false`.
- After real searches exist, review the three generated candidate comparisons,
  selected run/artifact hashes, and the Phase-13 support handoff.

## Sanskruti review checklist

- Verify real XGBoost candidates belong to each frozen exact-30 search.
- Verify XGBoost and GRU receive identical canonical information, differing only
  by flattened versus sequence representation.
- Verify feature, label, split, preprocessing, and eligible validation population
  provenance before any real selection is persisted.

## Unresolved before real selection

- `UNLOCKED PARAMETER — CROSS-FAMILY EXACT-TIE POLICY`
- `UNLOCKED ENGINEERING PARAMETER — METRIC TIE TOLERANCE`
- Phase-9 AUPRC integration convention review blocker
- Scientific search sampler/seed and GRU/XGBoost execution prerequisites from
  Phase 11, including the project seed policy for any required selected-GRU rerun
- Git commit provenance is unavailable because this workspace is not a Git checkout

Phase 13 must receive exactly one genuinely selected support classifier plus its
validation prediction/label eligibility references. This package contains no
calibration or threshold implementation.
