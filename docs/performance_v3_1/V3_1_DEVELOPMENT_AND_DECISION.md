# V3.1 Candidate Development and Terminal Decision

Protocol `configs/performance_v3_1/scientific_protocol_v1.yaml` (SHA-256 `29556d28bf5dc36b73cdef8f70c171a9146227caa2ab4ebe50c6134fe6ba5257`).

## Search (5-fold grouped CV, 32 pre-declared trials)

| Feature family | best OOF AUPRC |
|---|---:|
| F0 | 0.6605 |
| F1 | 0.6617 |
| F2 | 0.6617 |
| F3 | 0.6619 |

All 32 trials lie within the 0.003 tie tolerance of the best; by the frozen tie rule the selected set is **F0** (557 features, AUPRC 0.6599).

## Held-out development decision set V31_HOLD (single comparison, frozen candidate)

| | AUPRC | AUROC | Brier | Log loss | ECE |
|---|---:|---:|---:|---:|---:|
| CANDIDATE | 0.6559 | 0.7335 | 0.2041 | 0.5911 | 0.0221 |
| NEW_MODEL_ONLY_CALIBRATED | 0.6549 | 0.7340 | 0.2037 | 0.5900 | 0.0262 |
| V3_FROZEN | 0.6553 | 0.7325 | 0.2046 | 0.5925 | 0.0287 |

Paired difference candidate − frozen V3: AUPRC +0.0006 [-0.0011, +0.0022], AUROC +0.0010, Brier -0.0006, log loss -0.0014.
Per-seed AUPRC gaps [0.0008, 0.0003, 0.0011]; thirds [0.0006, 0.0006, 0.0004].

## Gate C

```
{
 "auprc_gap_ge_0.015": false,
 "brier_ok": true,
 "lb95_gt_0": false,
 "log_loss_ok": true,
 "passes_gate_c": false,
 "stable_thirds_ge_0.005": false
}
```

**Decision: V3 retained for all four heads. No V3.1 sealed test was generated or opened.**
