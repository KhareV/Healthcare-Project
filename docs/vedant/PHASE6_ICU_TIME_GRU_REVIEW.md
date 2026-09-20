# Phase 6 Remaining ICU-Stay-Time GRU — Reviewer Package

**Owner:** Vedant Khare  
**Primary serving reviewer:** Pulkit  
**Target-semantics reviewer:** Sanskruti Satish Shete  
**Scope:** synthetic ICU-time GRU implementation; no search or scientific result

## Review decision requested

Confirm that the scalar output, current-ICU target language, and one shared
postprocessor match the frozen contract. No hospital-discharge, recovery,
survival, or mortality target is implemented. No real target labels are
derived or modified in this phase.

## Model and data flow

```text
canonical values + observation masks + padding
                    |
           Phase-5 shared GRUEncoder
                    |
             final hidden [B,H]
                    |
              scalar head [B,1]
                    |
     raw y_hat = predicted log1p remaining
             current-ICU hours
                    |
      clamp y_hat to >= 0 in log space
                    |
                 expm1
                    |
       finite nonnegative hours [B,1]
```

`src/models/gru.py` and the Recovery GRU were not modified. Phase 6 adds only
the task-specific scalar head, target checks, masked loss, postprocessor,
bundle loader, smoke command, tests, and configuration.

## Frozen target semantics

For prediction time `t` and the same ICU stay's `outtime`:

```text
remaining_hours = (outtime - t) in hours
y = log1p(remaining_hours)
```

The model trains directly on the canonical `icu_time` log target. No fitted
target normalization or scaler is added. The pure reference function used by
semantic tests accepts only `prediction_time` and `icu_outtime`; hospital
discharge is not an argument and is not referenced in model code.

Synthetic tests vary hospital discharge while holding ICU outtime fixed and
obtain the same target. They also alter the target while holding the canonical
history fixed and prove the exact assembled GRU input remains bit-identical.
Neither `outtime` nor hospital discharge exists in `CanonicalBatch` model
inputs.

## Loss and eligibility

Huber loss is calculated on raw log1p targets only after indexing
`icu_time_eligible`. Ineligible targets are not zero-filled or validated. An
attack replacing an ineligible target with `-999` leaves loss, gradients, and
the optimizer update identical. An all-ineligible batch returns no loss and
the shared engine skips optimization.

Huber `delta=1.0` in the smoke config is an engineering development setting,
not a selected hyperparameter.

## One shared postprocessor

`remaining_icu_hours_from_log_prediction` is the only implemented conversion:

```text
hours = expm1(clamp_min(raw_log_prediction, 0))
```

The clamp is explicitly before `expm1`. Non-finite raw values and overflowed
hours fail closed. Tests cover `[-1, 0, log(2), log(11)] -> [0, 0, 1, 10]` and
call the exact same function for validation-, inference-, and
serving-compatible paths.

Postprocess version:
`icu_time_nonnegative_log_expm1_v1`.

## Checkpoint contract

Metadata includes:

- task `icu_stay_time` and family `gru`;
- raw-output meaning `log1p_remaining_current_icu_hours`;
- postprocess version;
- full GRU config and seed;
- tensor/feature/split/config provenance;
- explicit absence of a target scaler;
- log-space Huber validation monitor/value;
- synthetic flag and exclusion from the later search.

`load_icu_time_bundle` reconstructs the model from metadata, verifies the
checkpoint hash and schema versions, strictly loads state, and rejects the
wrong task, family, raw-output meaning, or postprocess version.

## Synthetic smoke evidence

Command:

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m training.icu_time_smoke --config configs/smoke/gru_icu_time_phase6_v1.json
```

- train rows: 2, both ICU-time eligible;
- validation rows: 3, all ICU-time eligible;
- test rows accessed: 0;
- epochs: 3; best epoch: 3;
- postprocessed shape: `[3,1]`;
- all resulting hours finite and nonnegative;
- checkpoint SHA-256:
  `382433b5ff30f6d4cc4ceeef6dfd21eea7ad674a67ce8c6855a461058ea80fa0`.

The observed validation Huber loss is non-reportable synthetic infrastructure
output and is not a correctness threshold.

## Pulkit review checklist

- Confirm raw output and postprocess metadata are sufficient for a future
  serving artifact.
- Confirm serving will import the shared postprocess function rather than
  duplicate clamp/expm1 logic.
- Confirm the bundle loader's task/family/version checks meet the future model
  routing boundary.
- Preserve the phrase “remaining time until the current ICU stay ends”; do not
  expose this as hospital discharge, recovery, or survival time.

## Sanskruti review checklist

- Confirm the canonical ICU-time field is already
  `log1p((outtime-t) hours)` for the current ICU stay.
- Confirm every legal real canonical cutoff has the intended explicit
  eligibility value.
- Confirm the semantic reference tests match her upstream label
  implementation without replacing it.
- Confirm the leakage proof: outtime changes the target only, never the model
  input.

## Blocked before real training

- **BLOCKED — INFORMATION REQUIRED:** final feature/preprocessing contract and
  split-compatible canonical artifacts.
- **BLOCKED — INFORMATION REQUIRED:** frozen real ICU-time label artifact and
  Sanskruti confirmation of target spot checks.
- **BLOCKED — INFORMATION REQUIRED:** TSLO no-observation encoding and static
  input contract, inherited from prior phases.
- **BLOCKED — REVIEW REQUIRED:** Pulkit approval of postprocessing and bundle
  compatibility before serving integration.

These do not block synthetic Phase-6 completion or later synthetic model work.
