# Phase 5 Recovery GRU — Scientific Review Package

**Owner:** Vedant Khare  
**Mandatory scientific reviewer:** Sanskruti Satish Shete  
**Compatibility reviewer:** Pulkit (checkpoint bundle only)  
**Scope:** synthetic recovery GRU implementation; no search or scientific result

## Review decision requested

Confirm that the two output slots and independent eligibility masks match the
frozen recovery labels. The implementation does not derive or modify SOFA,
does not invent real labels, does not use test rows, and does not run any part
of the future 30-configuration search.

## Model diagram

```text
canonical values [B,8,F] ---------+
                                    +--> concat per bin --> [B,8,2F]
observation mask [B,8,F] -- float -+            |
                                                 | exclude prefix padding
padding mask [B,8] ------------------------------+ via packed valid suffix
                                                 |
                                         unidirectional GRU
                                                 |
                                      final hidden state [B,H]
                                                 |
                                      configured dropout once
                                                 |
                                           linear head
                                                 |
                      standardized output [B,2]
                      index 0: DeltaSOFA24
                      index 1: DeltaSOFA48
```

The real contract also requires TSLO. The current Phase-3 fixture carries
`tslo=None` because its no-observation sentinel is unresolved, so the synthetic
config explicitly disables that channel. The encoder supports a contract-
supplied `[B,8,F]` TSLO tensor when enabled. Static input is rejected rather
than guessed until the static-input contract is frozen.

## Frozen target semantics

- Output 0 is `SOFA(t+24h) - SOFA(t)`.
- Output 1 is `SOFA(t+48h) - SOFA(t)`.
- The head emits both values from the shared final representation; neither
  output is an input to the other.
- Original-unit reconstruction uses the same baseline independently:
  `SOFA_hat_24 = SOFA(t) + Delta24` and
  `SOFA_hat_48 = SOFA(t) + Delta48`.
- Raw deltas remain unclipped. Optional `[0,24]` clipping exists only in the
  pure display reconstruction function.

## Masked Huber loss

Targets are standardized separately and Huber loss is calculated only after
indexing eligible rows for each horizon. A horizon with no eligible target
contributes nothing; a batch with no eligible recovery target returns no loss
and the shared engine skips optimization.

**UNLOCKED ENGINEERING PARAMETER:** the current aggregation first averages
eligible pairs within each available horizon, then averages the available
horizon means equally. This avoids weighting a horizon merely because it has
more available rows. It is not claimed as a frozen scientific choice and
requires Sanskruti review before real training.

The censoring attack changes an ineligible 48-hour target to `999999` and
proves loss, gradients, and the optimizer update remain bit-identical.

## Target-scaler provenance

`RecoveryTargetScaler` refuses any loader whose dataset partition is not
`train` and also checks every batch identifier says `train`. Statistics use
eligible targets only and are separate by horizon.

Synthetic fitted values:

| Horizon | Eligible train count | Mean | Population standard deviation |
|---|---:|---:|---:|
| 24h | 2 | -1.5 | 0.5 |
| 48h | 1 | -3.0 | 0.0, replaced by unit scale |

**UNLOCKED ENGINEERING PARAMETER:** population standard deviation (`ddof=0`)
and unit scale for a zero-variance horizon are explicit, serialized engineering
choices requiring review before real training. Validation and test loaders are
rejected for fitting. The deterministic scaler artifact records partition,
eligible counts, versions, split hash, variance convention, and zero-variance
policy.

## Architecture/configuration

The reusable encoder enforces the frozen search bounds without executing a
search: hidden dimension `{64,128,192}`, layers `{1,2}`, dropout `[0.10,0.40]`,
unidirectional only, and explicit observation masks. The smoke configuration
uses `H=64`, one layer, dropout `0.1`, AdamW, batch size `64`, and three epochs.
These are one legal development configuration, not a selected configuration.
Dropout is applied once to the final GRU representation; recurrent dropout is
zero. This exact placement is an engineering interpretation to review before
the search-space freeze.

## Synthetic smoke evidence

Command:

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m training.recovery_smoke --config configs/smoke/gru_recovery_phase5_v1.json
```

- two train examples and three validation examples;
- zero test examples accessed;
- three epochs, best epoch 3;
- inverse-transformed prediction shape `[3,2]`, all finite;
- raw deltas not clipped;
- checkpoint SHA-256:
  `8c3c3efe140a683899692f7605cd07f42a5bc6ee7cfc547d5003ab0654527303`;
- scaler SHA-256:
  `c34ce85b8b5e1e04ced3ceead6a9fc4ff6e39d5727191180c44c8035a49ab72f`.

Repeated runs produced identical logs, metrics, scaler bytes, and checkpoint
bytes. The observed standardized validation Huber value is infrastructure
evidence only and is not reportable scientific performance.

## Checkpoint bundle contract

The checkpoint metadata contains task/family, full model config, seed,
tensor/feature versions, split hash, scaler path/hash, explicit horizon order,
Huber loss and aggregation, validation monitor/value, code-commit status,
synthetic flag, and `participates_in_30_config_search=false`.

`load_recovery_bundle` constructs the model from metadata, hash-verifies and
strict-loads the checkpoint, confirms recovery/GRU identity and exact horizon
ordering, then hash-verifies and version-checks the target scaler. Smoke
inference uses this fresh-loaded bundle.

## Sanskruti review checklist

- Confirm outputs match the independent label fields she supplies.
- Confirm no 24-hour prediction or target enters 48-hour reconstruction.
- Confirm horizon-specific eligibility behavior and censoring attack.
- Review equal-horizon loss aggregation before real training.
- Review target standard-deviation convention and zero-variance policy.
- Confirm values plus genuine-observation masks are canonical information;
  approve TSLO integration only after its upstream representation is frozen.
- Confirm this implementation does not redefine SOFA or recovery labels.

## Blocked before real recovery training

- **BLOCKED — INFORMATION REQUIRED:** final feature schema/order/dimension and
  train-derived dynamic-feature preprocessing artifact.
- **BLOCKED — INFORMATION REQUIRED:** frozen TSLO no-observation encoding and
  actual canonical TSLO tensors.
- **BLOCKED — STATIC INPUT CONTRACT REQUIRED:** names/order/dimension,
  preprocessing, and reviewed model-input design.
- **BLOCKED — INFORMATION REQUIRED:** real canonical recovery labels and
  split-compatible train/validation artifacts.
- **BLOCKED — REVIEW REQUIRED:** Sanskruti approval of loss aggregation,
  target-scaler conventions, dropout placement, and label compatibility.

These blockers do not invalidate the synthetic implementation. They prevent
real training or a scientific claim.
