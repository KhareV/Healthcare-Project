# Phase 8 Fixed LSTM Sensitivity — Sanskruti Review Package

**Owner:** Vedant Khare  
**Mandatory reviewer:** Sanskruti  
**Compatibility notification:** Pulkit  
**Scope:** reusable implementation plus synthetic smoke evidence only

## Decision requested

Verify that the implementation preserves the Phase 5–7 scientific and data
contracts and changes only the recurrent cell from GRU to LSTM. No selected
GRU configurations exist yet, so this package contains no final scientific
sensitivity result.

**BLOCKED — FINAL SELECTED GRU CONFIGURATIONS REQUIRED FOR SCIENTIFIC LSTM
RUNS**

## Architecture delta

```text
canonical [B,8,F] values + observation mask (+ TSLO when enabled)
                              |
                  identical validation/assembly
                              |
                +-------------+-------------+
                |                           |
          source GRU cell             LSTM cell
                |                           |
       same final hidden size       same final hidden size
                |                           |
                +------ same task head -----+
```

The LSTM subclasses the shared GRU encoder solely to reuse its exact input
validation, padding alignment, and temporal-field assembly. It replaces the
unidirectional `torch.nn.GRU` with a unidirectional `torch.nn.LSTM`. Hidden
dimension, layer count, final-representation dropout, feature inputs, and task
heads remain unchanged. There is no attention, bidirectionality, residual
block, convolution, extra hidden layer, or LSTM-only feature.

## Current parent configurations

The only available parents are explicitly development smoke runs; none is
called selected:

| Task | Source GRU run | Config SHA-256 | Checkpoint SHA-256 |
|---|---|---|---|
| Recovery | `phase5_synthetic_recovery_gru_smoke_v1` | `79a2f26abbe339754bce18d9df9ea696ac6cfb9df8748c8eb4862c5cfed82d87` | `8c3c3efe140a683899692f7605cd07f42a5bc6ee7cfc547d5003ab0654527303` |
| ICU time | `phase6_synthetic_icu_time_gru_smoke_v1` | `47d05831c49b10137591793b937bb85236505e5a75966eef88782dc154c07194` | `382433b5ff30f6d4cc4ceeef6dfd21eea7ad674a67ce8c6855a461058ea80fa0` |
| Support | `phase7_synthetic_support_gru_smoke_v1` | `f482c5d01d517853225e0b3d802b449cdda4c8afd6402612c14cffd2c9079771` | `d2ae7f884826645eb71a6af50ce6d60c855910dacf301b0bb82604140d8eba81` |

Each derived config validates the parent config and checkpoint hashes before
training. It embeds the canonical fixture hash
`9269c98a0e5942f464d80f29054c174fc462223621c9f9ec8e136b0705c900df`
and split hash
`217184e2dc4e9e2092aec962bf5523660006e3e4ffa35d2f807bafdfbb871b37`.

## Mapping and parity evidence

The derivation utility copies seed, batch size, worker count, maximum epochs,
device, optimizer name/learning rate/weight decay, and the complete early
stopping mapping. The model mapping copies feature dimension, hidden dimension,
layer count, dropout, observation-mask inclusion, TSLO inclusion, static
dimension, and directionality. Tests fail on any difference in these fields.

Both families use the same Dataset/DataLoader and tensorization path. The
LSTM and GRU input assemblers produce bitwise-identical packed inputs and
sequence lengths. Train/validation subject identifiers come from the same
frozen split artifact; no split generation exists in the LSTM path. No test
partition is loaded, and every smoke metric records `test_examples_accessed=0`.

Task artifacts are shared, not refit:

- Recovery reuses target scaler SHA-256
  `c34ce85b8b5e1e04ced3ceead6a9fc4ff6e39d5727191180c44c8035a49ab72f`
  and the unchanged independent-horizon masked Huber adapter.
- ICU time reuses the shared `expm1(max(0, raw_log_prediction))`
  postprocessor and unchanged eligible log-space Huber adapter.
- Support reuses class-weight SHA-256
  `e03d9f7eac8411cae7ed9b93d45f5e40d5c75abec0ae35aae909904769245b7f`,
  unchanged eligibility-first BCE-with-logits, and one external sigmoid. It
  remains uncalibrated and has no threshold.

## No-search and no-serving evidence

There are exactly three synthetic comparison configs and no LSTM search
module, candidate array, distribution, trial count, random/grid/Optuna setting,
or configuration budget. Each config and checkpoint records sensitivity-only,
not scientific, no LSTM search, and no serving participation. No G3 marker or
`selected_models_v1.json` exists, and no routing/API code was changed.

## Parameter-count evidence

Counts are descriptive only. Hidden dimension remains 64 and was not changed
to equalize parameter counts.

| Task | GRU parameters | LSTM parameters |
|---|---:|---:|
| Recovery | 13,570 | 18,050 |
| ICU time | 13,505 | 17,985 |
| Organ support | 13,505 | 17,985 |

## Synthetic smoke evidence

All runs trained for three epochs, reloaded a fresh model through the shared
hash/version-checked checkpoint loader, and produced finite validation
predictions of the expected shape.

| Task | Best epoch | Shape | LSTM checkpoint SHA-256 |
|---|---:|---:|---|
| Recovery | 3 | `[3,2]` | `87f211427c2d15d13ee60e28e6842bfcd367b8fa67e721c133c05b468d875fb2` |
| ICU time | 3 | `[3,1]` | `9e61cbd87b726c85a98c59181a5abeebdc4e433cd675c4e1ed9349fa471936f6` |
| Support | 1 | `[3,1]` | `4973b6cd58e23fff36fae47a86d5a4ae921ca70e6dc307f600c60206e875db0a` |

Validation values are smoke monitors only and are not reportable scientific
performance or architecture comparisons.

## Reviewer checklist

- Confirm the canonical value/mask/padding assembly is unchanged.
- Confirm target, eligibility, loss, transform, split, and preprocessing
  artifacts remain the task-specific Phase 5–7 implementations.
- Confirm the mapping allowlist contains only cell/family and run-output
  provenance differences.
- Confirm parameter counts are recorded descriptively without hidden-size
  adjustment.
- Confirm the smoke parents are not mistaken for selected GRU runs.
- Confirm no search, calibration, threshold, serving selection, test access,
  or Phase 9 work is present.
- After the real selected GRU configurations are frozen, review the generated
  one-per-task comparison configs before authorizing exactly three scientific
  runs.
