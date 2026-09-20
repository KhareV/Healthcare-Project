# Vedant Viva Evidence

1. **Why ICU stay?** It is the episode, prediction clock, remaining-time endpoint, and bootstrap cluster.
2. **First cutoff?** ICU hour 24, then every 6 hours, at most 12 cutoffs.
3. **Why an eight-bin tensor at hour 24?** Pre-ICU positions are padding, never floor-history values.
4. **Lookback?** Only `(t-48h,t]`; older observations cannot seed forward fill.
5. **DeltaSOFA48?** `SOFA(t+48)-SOFA(t)`, independent of DeltaSOFA24.
6. **Baseline SOFA leakage?** SOFA(t) uses only information through t and defines the target baseline.
7. **Split?** Subject-disjoint coarse temporal eras: train 2008–13, validation 2014–16, test 2017–19.
8. **XGB/GRU parity?** Same canonical values, masks, TSLO and permitted statics; representation differs.
9. **Stay balancing?** Each eligible row has weight `1/n_i`, giving each represented stay total weight 1.
10. **Bootstrap?** Resample whole ICU stays with replacement and preserve duplicate stay-copy identity.
11. **Support censoring?** OFF at t is required; incomplete 24h follow-up without an event is censored.
12. **Calibration versus threshold?** Isotonic maps probabilities; the separate validation-F1 threshold creates alerts.
13. **Why G3?** It proves all development decisions are frozen before one-time test access.
14. **Mixed families?** XGBoost versus GRU is selected independently per task.
15. **Bootstrap CI meaning?** Aggregate test-set performance uncertainty, not patient uncertainty.
16. **Selected manifest?** It binds each task to family, run, model, preprocessing, split, label and routing hashes.
17. **Registry lineage?** Results trace through predictions and checkpoints to run/config/data versions and commit.
18. **Main limitations?** Retrospective event time, informative horizon eligibility, single-center data, point forecasts,
no external/bedside validation, and non-causal explanations.
