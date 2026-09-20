# Phase 4 Training Infrastructure — Reviewer Package

**Owner:** Vedant Khare  
**Required reviewers:** Pulkit; Sanskruti for feature-contract compatibility  
**Scope:** synthetic Dataset/DataLoader and model-neutral training framework only

## Review decision requested

Approve the engineering interface for later Phase 5–7 task adapters without
interpreting the smoke model or smoke metric as a research result. No GRU,
LSTM, XGBoost, final preprocessing, model selection, calibration, threshold,
bootstrap, explanation, or test evaluation is included.

## End-to-end proof

The configured path is:

`Phase-3 fixture -> CanonicalTensorDataset -> CanonicalBatch -> SmokeTestModel
-> eligible-only loss -> optimizer -> validation monitor -> early stopping ->
best checkpoint restore -> SHA-256 -> experiment registry`.

The command

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m training.smoke --config configs/smoke/phase4_smoke_v1.json
```

was run twice with identical results and checkpoint bytes. The retained result
used CPU, two train examples, three validation examples, zero test examples,
stopped after epoch 3, restored epoch 1, and has checkpoint SHA-256
`ec121653422647e1e6313a2bee7becd85cf58142a97151d41944b21b0c13adc9`.
The observed validation value is infrastructure output only and is explicitly
non-reportable scientific performance.

## Batch and serialization contract

The normative field/shape table is in
`docs/contracts/PHASE4_BATCH_CONTRACT.md`. Input is the deterministic Phase-3
synthetic JSON transport. It is validated before tensorization against the
canonical schema, `tensor_contract_v1`, feature-schema reference, duplicate
key rule, and supplied subject split. This JSON format and the zero-valued
synthetic numerical fill are test-only engineering choices, not the production
serialization or preprocessor.

## Checkpoint and load contract

The `.pt` payload contains `model_state_dict`, `optimizer_state_dict`, and
provenance metadata. Its adjacent `.metadata.json` sidecar repeats the payload
metadata and adds the checkpoint filename and SHA-256. Required provenance
includes run/task/model family, seed, epoch, config hash, tensor-contract and
feature-schema versions, split hash, code commit status, validation value, and
the mandatory `synthetic_smoke_test: true` flag.

`load_checkpoint` verifies the sidecar, file hash, filename, expected tensor
and feature versions, safe payload structure, exact payload/sidecar metadata,
and strict model state compatibility before restoration. Optimizer restoration
is optional for serving-style model loads. Corruption and incompatible versions
fail closed with `CheckpointError`. Unit and smoke tests demonstrate save,
hash, load, optimizer/model roundtrip, best-checkpoint restoration, and
corruption/version rejection.

## Reproducibility and configuration

One utility seeds Python, NumPy, PyTorch CPU, and all available CUDA generators
and requests deterministic algorithms. DataLoader shuffle uses its own seeded
generator; workers receive deterministic derived seeds. Device selection is
`cpu`, `cuda`, or `auto`, with no hard-coded GPU index. The smoke CLI validates
that its referenced tensor-contract artifact and version agree.

The project-wide primary seed and final runtime lock remain unresolved. Seed
`314159`, batch size 2, four maximum epochs, `num_workers=0`, CPU,
AdamW learning rate `0.01`, zero weight decay, validation-loss monitor, and
patience 2 are explicitly labelled engineering test defaults only.

## Sanskruti compatibility review

- Confirm the future canonical artifact supplies exact feature order and
  feature-schema version; the loader never sorts or derives features.
- Confirm values remain `[8,F]` oldest-to-newest, with separate `[8]` padding
  and `[8,F]` genuine-observation masks.
- Supply the frozen TSLO no-observation representation and static names/order;
  current values remain `None` rather than guessed.
- Supply the real train-derived preprocessing interface/artifacts and recovery
  and ICU-time labels. Phase 4's synthetic fill is not reusable as production
  preprocessing.
- Preserve the Phase-2 subject assignment artifact; the loader rejects a
  canonical row whose split conflicts with it.

## Pulkit compatibility review

- Confirm the canonical identifiers and version metadata are sufficient for
  future PredictionPipeline inputs without rebuilding masks or feature order.
- Confirm the explicit support target plus independent eligibility tensor can
  represent positive, negative, and censored/ineligible examples; ineligible
  values are `NaN` after collation and cannot enter loss.
- Confirm the `.pt` state plus JSON metadata sidecar, strict version checks,
  hash verification, and repository-relative registry references meet the
  eventual selected-model loading boundary.
- Nominate/freeze the production serialization contract later. Phase-3 JSON is
  accepted here only as a synthetic fixture transport.

## Blocked for real integration

- **BLOCKED — INFORMATION REQUIRED:** final feature schema/order/dimension,
  TSLO sentinel/encoding, static feature contract, and production canonical
  serialization from the owning upstream contract.
- **BLOCKED — INFORMATION REQUIRED:** real canonical feature/label artifacts
  and train-fitted preprocessing inputs/artifacts.
- **BLOCKED — INFORMATION REQUIRED:** Pulkit's final support event/label
  contract version and future serving artifact contract.
- **UNLOCKED ENGINEERING PARAMETER:** final package/environment lock,
  project-wide seed, DataLoader worker count, logging cadence, and later-task
  scheduler wiring remain to be frozen through reviewed configuration.

These blockers do not prevent Phase 4 completion or synthetic Phase 5
development. They do prevent claiming production/real-data readiness.
