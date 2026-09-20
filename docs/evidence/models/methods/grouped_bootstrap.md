# Grouped Bootstrap Evidence

Status: **FINAL** method evidence. Implementation: `stay_cluster_percentile_bootstrap_v1`.

```text
sample ICU stay IDs with replacement
  -> carry every eligible snapshot for each sampled stay copy
  -> assign a distinct identity to duplicate sampled copies
  -> recompute weights so each sampled copy contributes total weight 1
  -> recompute the same point estimator
  -> repeat frozen B times with the frozen seed
  -> use the 2.5th and 97.5th percentiles
```

The resampling unit is the ICU stay, never an individual prediction row. A classification
replicate lacking required class diversity is skipped only for that metric, and valid and
invalid replicate counts are reported. The interval quantifies uncertainty in aggregate
performance. It is not an individual prediction interval or a confidence percentage.

Real B and seed remain unresolved, so no real CI evidence is generated.
