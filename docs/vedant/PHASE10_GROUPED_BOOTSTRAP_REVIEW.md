# Phase 10 Grouped Stay-Cluster Bootstrap — Sanskruti Review Package

**Owner:** Vedant Khare  
**Mandatory reviewer:** Sanskruti  
**Scope:** synthetic framework validation only; no final test access

## Algorithm

For each metric-specific eligible record set:

```text
original_stays = unique eligible stay IDs, in stable first-seen order
N = number of original_stays

for replicate b:
    draw N stays with replacement using explicit seeded RNG
    for each draw position d:
        copy every eligible row for the drawn original stay
        assign bootstrap_cluster_id = (version, b, d)
    call the same Phase-9 point estimator on copied records
    retain a finite result; otherwise count one invalid replicate for this metric

CI95 = Type-7 percentile(valid values, [0.025, 0.975])
```

Rows are never independently sampled. The engine pre-indexes eligible records
by original `stay_id`. Task wrappers call `evaluate_recovery_horizon`,
`evaluate_icu_time`, and `evaluate_organ_support`; bootstrap contains no MAE,
RMSE, weighted-median, AUPRC, AUROC, Brier, or confusion-metric formula.

## Duplicate stay-copy proof

For forced draws `[A, A]`, where A contains rows A1 and A2:

```text
cluster (b, draw 0): A1, A2 -> weights 1/2, 1/2 -> total 1
cluster (b, draw 1): A1, A2 -> weights 1/2, 1/2 -> total 1
```

The original ID is retained in `BootstrapCopiedRow.original_stay_id`, while
Phase-9 receives the distinct `bootstrap_cluster_id` as its grouping key.
Duplicate draws therefore contribute twice and cannot collapse back to one
original-stay group.

## Rows-together and long-stay proof

A forced `[B, A]` draw carries B1/B2 followed by A1/A2/A3, with no independent
row sampling. A second fixture draws a one-row stay and a ten-row stay: the
copy totals are 1 and 1 (within floating-point tolerance), not 1 and 10.

Task-specific filtering is applied before indexing. A recovery24 stay-copy
with four eligible records carries four rows; the corresponding recovery48
copy carries only its two eligible rows. Extreme changes to ineligible records
leave draws, distributions, and CI bounds identical.

## Two-stay analytic fixture

```text
Stay A: one row, absolute error 10
Stay B: one row, absolute error 0

forced draw  Phase-9 stay-balanced MAE
[A,A]        10
[A,B]         5
[B,A]         5
[B,B]         0
```

The original point estimate is 5. The distribution is `(10,5,5,0)`. With the
declared Type-7 convention, CI95 is `[0.375, 9.625]`.

An ICU weighted-median fixture separately proves the reported point estimate
is not the bootstrap mean: original median AE is 0 while the mean of four
forced replicate estimates is 2.5.

## Classification invalid-replicate proof

One stay contains only a positive and one only a negative. Ten forced AUROC
replicates contain seven mixed-class draws and three single-class draws.

```text
n_requested = 10
n_valid = 7
n_invalid = 3
valid distribution = seven values of 1.0
CI95 = [1.0, 1.0]
```

No invalid replicate is replaced by 0 or 0.5, and no redraw occurs. A separate
all-invalid fixture returns `ci_lower=None` and `ci_upper=None`. Other metrics
follow their Phase-9 defined/undefined behavior independently.

## Reproducibility

The engine uses a local `random.Random(seed)` instance, never global random
state. Identical records, callback, test `B`, and test seed reproduce the full
draw sequence, valid/invalid pattern, distribution, and bounds. A different
seed changes the sampled-cluster sequence on the nontrivial fixture.

The test seed `8675309` and small test replicate counts are test values only.
They are not scientific defaults.

## Result contract

Every `BootstrapResult` records:

- task, horizon, and metric name;
- original Phase-9 point estimate;
- CI lower/upper, level 0.95, and percentile method;
- requested, valid, and invalid replicate counts;
- original eligible example and stay counts;
- explicit bootstrap seed;
- Phase-9 metric and Phase-10 bootstrap implementation versions;
- valid replicate distribution and sampled original-stay sequences;
- aggregate-performance-only uncertainty statement;
- optional split/hash/run/label/probability/threshold provenance supplied by callers.

The metadata explicitly records no final test access, no independent row
sampling, and sampling with replacement.

## Frozen and unresolved parameters

The versioned [bootstrap configuration](../../configs/evaluation/bootstrap_v1.json)
freezes the method and CI level but intentionally leaves these unset:

- **UNLOCKED PARAMETER — NUMBER OF BOOTSTRAP REPLICATES B**
- **UNLOCKED PARAMETER — BOOTSTRAP RNG SEED**
- **UNLOCKED REPORTING / QA PARAMETER — MINIMUM VALID REPLICATES**

The Type-7 percentile interpolation is an **UNLOCKED ENGINEERING DETAIL** that
must be reviewed and frozen before final CIs. Phase-9 blockers for recovery
zero-direction and AUPRC integration also remain in force.

## Scientific isolation

This is an evaluation bootstrap over frozen predictions. It does not train or
refit a model, preprocessor, target scaler, calibrator, or threshold. Threshold
metrics require an externally supplied identifier and value and record
`threshold_reselected=false`; support metadata records
`calibrator_refit=false`.

No G3 marker or final selected-model manifest exists. No final test data or
real prediction artifact was opened. These intervals describe aggregate
metric uncertainty and must never be shown as individual-patient prediction
intervals.

## Review checklist

- Reproduce the four two-stay draw values and CI bounds.
- Verify duplicate original stays receive distinct cluster-copy IDs.
- Verify all eligible rows move together and each copy totals weight one.
- Confirm task/horizon eligibility is applied before cluster indexing.
- Confirm task wrappers directly call Phase-9 evaluators.
- Confirm invalid classification replicates are counted without redraw.
- Review Type-7 percentile interpolation.
- Freeze final `B`, seed, and valid-replicate QA policy before test access.
- Confirm no model/calibrator/threshold fitting or patient-level uncertainty claim exists.
