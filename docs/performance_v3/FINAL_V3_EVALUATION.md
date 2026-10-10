# Performance V3 — Final Evaluation (SYN-V3-T, sealed cohort, one-time access)

All values are rendered from `artifacts/performance_v3/phase4/*.json` by `scripts/performance_v3_write_final_report.py`.
Predictions SHA-256 `7ab7000f3513429dcbf986295f90aa2fca6eee6c82aafe85e62c7e6ead6773f1`; evaluation freeze `artifacts/performance_v3/governance/v3_final_evaluation_freeze_v1.json`.

## Denominators

| Task | eligible rows |
|---|---:|
| icu_stay_time | 9008 |
| organ_support | 5703 |
| recovery24 | 5130 |
| recovery48 | 1816 |

## Primary metrics (stay-balanced, 95% stay-cluster bootstrap CI, 2,000 replicates)

| Task | Naive | V3 | V2 as served (primary comparator) | V2 as evaluated (historical harness) | V2-recipe refit (control) |
|---|---:|---|---|---|---|
| Recovery24 MAE | 1.2539 | 1.0261 [0.9815, 1.0716] | 1.0876 [1.0463, 1.1286] | 1.0876 [1.0463, 1.1286] | 1.0463 [1.0037, 1.0889] |
| Recovery48 MAE | 1.6320 | 1.3143 [1.2417, 1.3924] | 1.3663 [1.2945, 1.4382] | 1.3663 [1.2945, 1.4382] | 1.3330 [1.2632, 1.4048] |
| ICU weighted median AE (h) | 9.7545 | 9.1011 [8.8694, 9.4116] | 18.2471 [17.9251, 18.6024] | 5.2914 [5.1797, 5.4168] | 9.1011 [8.8694, 9.4116] |
| Support calibrated AUPRC | 0.4075 | 0.6238 [0.5890, 0.6562] | 0.5197 [0.4872, 0.5518] | 0.6930 [0.6580, 0.7249] | 0.6056 [0.5702, 0.6386] |

## Support detail

| Model | cal AUPRC | cal AUROC | Brier | F1 @ frozen threshold | raw AUPRC | raw AUROC |
|---|---:|---:|---:|---:|---:|---:|
| V3 | 0.6238 | 0.7210 | 0.2072 | 0.6437 | 0.6238 | 0.7210 |
| V2_AS_SERVED | 0.5197 | 0.6746 | 0.2484 | 0.5912 | 0.5367 | 0.6758 |
| V2_AS_EVALUATED | 0.6930 | 0.7822 | 0.1940 | 0.6656 | 0.7407 | 0.7889 |
| CONTROL | 0.6056 | 0.7157 | 0.2088 | 0.6430 | 0.6150 | 0.7166 |

## Paired differences, V3 minus V2 as served (primary)

| Task | metric | diff | 95% CI | 98.75% CI | decision |
|---|---|---:|---|---|---|
| recovery24 | mae | -0.0615 | [-0.0788, -0.0440] | [-0.0834, -0.0396] | PROMOTE_V3 |
| recovery48 | mae | -0.0520 | [-0.0881, -0.0161] | [-0.0981, -0.0072] | PROMOTE_V3 |
| icu_stay_time | weighted_median_ae_hours | -9.1460 | [-9.5543, -8.7356] | [-9.6854, -8.6397] | PROMOTE_V3 |
| organ_support | auprc | +0.1041 | [+0.0793, +0.1296] | [+0.0726, +0.1369] | PROMOTE_V3 |

## Paired differences vs the other comparators

| Comparison | diff | 98.75% CI |
|---|---:|---|
| icu_stay_time|weighted_median_ae_hours|V3_vs_CONTROL | +0.0000 | [+0.0000, +0.0000] |
| icu_stay_time|weighted_median_ae_hours|V3_vs_V2_AS_EVALUATED | +3.8097 | [+3.5484, +4.1333] |
| organ_support|auprc|V3_vs_CONTROL | +0.0182 | [+0.0096, +0.0268] |
| organ_support|auprc|V3_vs_V2_AS_EVALUATED | -0.0692 | [-0.1098, -0.0237] |
| recovery24|mae|V3_vs_CONTROL | -0.0202 | [-0.0359, -0.0037] |
| recovery24|mae|V3_vs_V2_AS_EVALUATED | -0.0615 | [-0.0834, -0.0396] |
| recovery48|mae|V3_vs_CONTROL | -0.0187 | [-0.0508, +0.0109] |
| recovery48|mae|V3_vs_V2_AS_EVALUATED | -0.0520 | [-0.0981, -0.0072] |

## Reading these results

* **Defect D1.** The frozen V2 models learned to use the *missingness* of `current_sofa` (null exactly when a stay ends within 24 h of the cutoff). The V2 offline harness preserved that pattern, which inflates its ICU-time and support numbers (`V2 as evaluated`); the V2 serving runtime always supplies SOFA(t), so the deployed behaviour is `V2 as served`. See `configs/performance_v3/protocol_amendment_v1_1.yaml` and `docs/performance_v3/DEFECT_D1.md`.
* V3 is *not* better than `V2 as evaluated` on ICU time or support; that comparator contains the leak and is reported only for continuity and attribution.
* The `clean subset` (>=24 h follow-up) is selected on the outcome for ICU time and support, so it is not leak-free for those tasks; it is reported in the JSON for recovery tasks only as a like-for-like check.
* V3 ICU time equals the V2-recipe control by the pre-registered replacement rule (no candidate beat the repaired V2 recipe beyond tolerance).
* Promotion decisions follow the frozen plan: icu_stay_time: PROMOTE_V3; organ_support: PROMOTE_V3; recovery24: PROMOTE_V3; recovery48: PROMOTE_V3.
* Absolute gains over the naive floor are modest, as predicted by the generator's irreducible noise; negative or null findings would have been reported identically.
