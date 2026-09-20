# Phase 9 Stay-Balanced Metrics — Sanskruti Review Package

**Owner:** Vedant Khare  
**Mandatory reviewer:** Sanskruti  
**Scope:** deterministic point estimates only; no bootstrap, selection, calibration, or test evaluation

## Formula sheet

For task or horizon `T`, stay `i`, and its `n_i` eligible snapshots:

```text
w_ij = 1 / n_i
sum_j w_ij = 1

weighted MAE  = sum(w * abs(y - yhat)) / sum(w)
weighted RMSE = sqrt(sum(w * (y - yhat)^2) / sum(w))
weighted median AE = weighted_quantile(abs(y - yhat), w, 0.5)
weighted Brier = sum(w * (p - y)^2) / sum(w)
```

Weights are recomputed independently for recovery 24h, recovery 48h, ICU
time, and organ support. Ineligible rows receive aligned weight zero and are
removed before value validation or estimation.

## Canonical unequal-snapshot calculation

```text
Stay A: one eligible error 10       -> weight 1
Stay B: three eligible errors 0     -> weights 1/3, 1/3, 1/3

ordinary row MAE = 10 / 4 = 2.5       (prohibited)
stay-balanced MAE = 10 / (1 + 1) = 5  (implemented)
```

Tests also construct stays with 1, 2, 7, and 12 eligible snapshots and verify
that each stay's weights sum to one.

## Recovery

The evaluator accepts raw delta targets and predictions and produces separate
24h and 48h results. Each horizon reports stay-balanced MAE, RMSE, eligible
examples, and eligible stays. It never accepts baseline SOFA or reconstructs
or clips dashboard SOFA values.

Directional agreement is mathematically unambiguous for strictly nonzero true
and predicted deltas and is implemented there. Exact-zero semantics are not
specified by either authoritative document:

**BLOCKED — DIRECTIONAL AGREEMENT ZERO/TIE SEMANTICS REQUIRED**

If an eligible zero occurs, directional agreement is `None` and the blocker is
returned in metadata; MAE and RMSE remain valid.

## Remaining ICU time

The evaluator consumes true and predicted hours. The test uses the shared
Phase 6 `expm1(max(0, raw_log_prediction))` function before building the
model-neutral records and proves errors are calculated in hours.

Primary weighted median AE uses this deterministic convention:

```text
sort values ascending
ignore zero-weight observations
return first value with cumulative_weight >= q * total_weight
```

At an exact half boundary this returns the lower observed value. This is an
**UNLOCKED ENGINEERING DETAIL — WEIGHTED MEDIAN TIE CONVENTION** requiring
freeze before real selection. Generic weighted percentiles are implemented;
final reporting levels remain an **UNLOCKED REPORTING PARAMETER**.

## Organ support

Probabilities must be finite and within `[0,1]`; labels must be binary. The
evaluator does not call models, apply sigmoid, calibrate, select thresholds,
or accept the training `pos_weight`.

- AUPRC path: weighted average precision using a step integral over recall;
  equal scores are processed as one threshold group.
- AUROC path: weighted positive-negative pair concordance, with half credit
  for equal-score ties.
- Brier: stay-weighted squared probability error.
- Optional threshold metrics: supplied threshold only, weighted TP/TN/FP/FN,
  sensitivity, specificity, precision, and F1.

One-class AUROC and zero-denominator threshold quantities return `NaN` and are
listed in `undefined_metrics`; they never return 0, 0.5, or 1 by fallback.
No-positive AUPRC is likewise `NaN`.

Because the source documents do not resolve average precision versus
trapezoidal PR area:

**BLOCKED — AUPRC INTEGRATION CONVENTION REVIEW REQUIRED BEFORE REAL MODEL
SELECTION**

## Hand-calculated classifier fixture

```text
Stay A: positive p=0.8, weight 1
Stay B: negatives p=0.9, 0.7, 0.1, each weight 1/3

weighted AP    = 0.75
weighted AUROC = 2/3
weighted Brier = (0.04 + (0.81 + 0.49 + 0.01)/3) / 2

at TEST threshold 0.5:
TP=1, FP=2/3, TN=1/3, FN=0
sensitivity=1, specificity=1/3, precision=0.6, F1=0.75
```

Duplicating the same legitimate negative prediction across additional
snapshots in Stay B and recomputing `1/n_i` leaves all point estimates
unchanged.

## Structural counts and corruption checks

Every result returns `N_examples` and `N_ICU_stays`. Classification additionally
returns `N_positive_examples` and the number of unique stays containing at
least one eligible positive row. Duplicate `stay_id + prediction_time` keys,
missing stay IDs, eligible NaN/infinity, invalid probabilities, nonbinary
labels, empty eligible sets, and invalid weights fail explicitly.

Ineligible targets and predictions are deliberately allowed to be null or
corrupt-looking because they are masked before numeric validation. Replacing
an ineligible row with a two-million-unit regression error or invalid support
values leaves metrics and counts exactly unchanged.

## Model-family and phase isolation

The API has no model-family argument affecting mathematics. Tests attach GRU
and XGBoost only as metadata and obtain identical metrics and counts. The
implementation version is `stay_balanced_metrics_v1`.

There is no bootstrap module, randomness, resampling, winner declaration,
selected-model manifest, calibrator, threshold optimizer, or test evaluation.
Phase 10 can call these same point estimators without copying formulas.

## Review checklist

- Recalculate the unequal-snapshot MAE and classifier fixture.
- Verify every eligible stay contributes total weight one.
- Verify recovery horizons use independent eligibility and raw deltas.
- Approve or revise the lower weighted-median boundary convention before real selection.
- Resolve recovery exact-zero directional semantics.
- Approve or replace the weighted average-precision AUPRC convention before real selection.
- Confirm training support `pos_weight` is absent from evaluation.
- Confirm no row-wise fallback, bootstrap, calibration, threshold selection, or model selection exists.
