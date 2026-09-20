# Stay-Balanced Metric Evidence

Status: **FINAL** method evidence. Implementation: `stay_balanced_metrics_v1`.

For task or horizon T, stay i has n_i eligible snapshots and each eligible row receives
`w_ij = 1 / n_i`; consequently each represented stay contributes total weight 1.

- Recovery: weighted MAE, weighted RMSE, and same-horizon directional agreement;
  24h and 48h remain separate.
- Remaining ICU stay time: weighted median absolute error in hours, weighted MAE,
  weighted RMSE, and frozen weighted error percentiles.
- New Organ-Support Initiation Risk within 24h: stay-balanced AUPRC, AUROC, Brier,
  sensitivity, specificity, precision, and F1. Selection uses validation AUPRC on raw
  probabilities, not F1.
