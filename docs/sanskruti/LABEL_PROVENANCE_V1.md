# Phase-8 Target Provenance

The machine-readable authority is `configs/labels/target_provenance_v1.json` (SHA-256 `2b43fc2aafdaec014bda7a04befd3cbf94cea3351129ee257568cbf440853cf3`). Every target is label-only and prohibited from model inputs.

| Target | Source | Formula | Eligibility | Leakage role |
|---|---|---|---|---|
| `delta_sofa_24` | Phase-6 `sofa_at(t)` and `sofa_at(t+24h)` | future minus baseline | Phase-4 24h flag plus executable SOFA | future label only |
| `delta_sofa_48` | Phase-6 `sofa_at(t)` and `sofa_at(t+48h)` | future minus the same baseline | Phase-4 48h flag plus executable SOFA | future label only |
| `remaining_hours` | current retained episode `outtime` and cutoff | elapsed hours | Phase-4 ICU-time flag | structural label only |
| `icu_time_log1p` | `remaining_hours` | `log1p(remaining_hours)` | Phase-4 ICU-time flag | label only |

SOFA scores and horizon cutoffs belong in label audit evidence, never in Phase-7 values, masks, TSLO, padding, statics, or family views. `outtime` is target truth and is likewise prohibited as a feature. No support target, split, fitted preprocessor, recovery scaler, or model artifact belongs to Phase 8.
