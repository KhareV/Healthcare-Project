# Phase 7 Organ-Support GRU — Pulkit Review Package

**Owner:** Vedant Khare  
**Mandatory endpoint reviewer:** Pulkit  
**Scope:** synthetic uncalibrated classifier only; no state-machine implementation

## Review decision requested

Confirm that the classifier consumes Pulkit-style final label/eligibility fields
without deriving or altering OFF/ON states, event mappings, onset timestamps,
or censor reasons. The real Pulkit contract is not present, so real training is
blocked; this package uses a separately labelled manual synthetic handoff.

## Architecture and output

```text
canonical values + observation masks + padding
                    |
           unchanged shared GRUEncoder
                    |
             final hidden [B,H]
                    |
                linear head
                    |
             one raw logit [B,1]
                    |
       external sigmoid, exactly once
                    |
       raw UNCALIBRATED probability
```

The trainable model never applies sigmoid. No calibration or operating
threshold exists in this phase.

## Label handoff boundary

`support_labels_synthetic_v1.json` is marked
`SYNTHETIC_MANUAL_ORGAN_SUPPORT_LABEL_HANDOFF_NOT_REAL_LABELS`. It contains
manual final labels, eligibility, and audit-only metadata. The overlay returns
only label and eligibility to the canonical item. OFF-at-t evidence, future
onset, censor reason, and future support state are never returned in a model
batch.

This fixture demonstrates, without implementing detection logic:

- positive before early ICU exit -> label 1, eligible;
- early exit with no positive event -> null label, ineligible;
- already-on continuation is not a new initiation; the other OFF component can
  still make the combined row classifiable;
- vasopressor-only initiation -> positive;
- invasive-ventilation-only initiation -> positive;
- both initiate -> positive;
- neither with complete follow-up -> negative.

Pulkit must replace this manual fixture with the frozen real handoff before
real training. Contradictory incoming labels must be reported as a cross-member
contract failure, not patched in the model.

## Eligibility and BCE loss

The task adapter indexes `organ_support_eligible` before validating labels or
calling binary cross entropy with logits. Eligible labels must be exactly 0 or
1. Censored targets remain null/NaN in canonical batches and cannot be coerced
to zero.

An attack changes an ineligible target to `999999`; loss, gradients, and the
optimizer update remain bit-identical. An all-ineligible batch returns no loss
and is skipped by the shared engine.

## Positive-class-weight provenance

The proposed formula is:

```text
pos_weight = eligible training negatives / eligible training positives
```

**UNLOCKED ENGINEERING PARAMETER — POSITIVE CLASS WEIGHT FORMULA:** the
authoritative documents require a training-derived weight but do not freeze
the numerical formula. `N_negative/N_positive` is the standard proposed
default and requires approval before final real training.

The fitter rejects non-train loaders, checks every row's split, indexes
eligibility before inspecting labels, and fails with
`SCIENTIFIC FEASIBILITY ISSUE — NO POSITIVE TRAINING EXAMPLES` when the
eligible training positive count is zero.

The deterministic artifact records split hash, eligible/positive/negative
counts, formula/version, weight, label and event-dictionary versions, and
source-config hash. It contains no validation/test prevalence.

Synthetic smoke counts are one positive and one negative, producing
`pos_weight=1.0`; this number has no scientific meaning.

## Leakage and information parity

Tests duplicate one canonical feature row and assign a different synthetic
future outcome. Sequence, padding mask, observation mask, TSLO, and static
fields remain identical. Label-audit metadata is accessible only through a
separate review method and never through `CanonicalBatch`.

The support GRU therefore sees the same canonical value/mask information as
the other GRUs and eventual flattened XGBoost path, subject to the previously
documented TSLO/static blockers.

## Checkpoint bundle

Metadata records:

- task `organ_support`, family `gru`, output type `logit`;
- probability transform `sigmoid_uncalibrated_v1`;
- class-weight artifact and SHA-256;
- label contract and event-dictionary versions;
- model/tensor/feature/split/config provenance;
- validation smoke monitor;
- `calibration=null` and `operating_threshold=null`;
- synthetic flag and exclusion from the later 30-run search.

`load_support_bundle` verifies checkpoint identity/hash/versions and the
referenced class-weight artifact/hash/label/event versions before inference.

## Synthetic smoke evidence

Command:

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m training.support_smoke --config configs/smoke/gru_support_phase7_v1.json
```

- train examples: 2; eligible positives/negatives: 1/1;
- validation examples: 3;
- test examples accessed: 0;
- best epoch: 1 of 3;
- output probability shape: `[3,1]`, finite and in `[0,1]`;
- probabilities calibrated: false;
- threshold selected: false;
- checkpoint SHA-256:
  `d2ae7f884826645eb71a6af50ce6d60c855910dacf301b0bb82604140d8eba81`;
- class-weight SHA-256:
  `e03d9f7eac8411cae7ed9b93d45f5e40d5c75abec0ae35aae909904769245b7f`.

The weighted validation BCE is a smoke monitor only, not the future
stay-balanced AUPRC selection criterion and not reportable performance.

## Pulkit checklist

- Approve or correct the expected real handoff fields and versions.
- Confirm manual positive-before-censor, early-exit censoring, continuation,
  and endpoint-union cases match the frozen state machine.
- Confirm audit metadata must remain unavailable to model inputs.
- Review the proposed `N_negative/N_positive` weighting formula.
- Confirm the checkpoint bundle is compatible with later selected-model
  loading.
- Confirm later calibration occurs only after selection and is not present
  here.

## Blocked before real support training

- **BLOCKED — PULKIT FROZEN ORGAN-SUPPORT LABEL ARTIFACT REQUIRED FOR REAL
  TRAINING:** final label, eligibility, OFF-at-t evidence, onset, censor reason,
  and label/event-dictionary versions.
- **BLOCKED — INFORMATION REQUIRED:** final feature/preprocessing artifacts,
  TSLO encoding, and static-input contract.
- **BLOCKED — REVIEW REQUIRED:** positive-class-weight formula approval.
- **UNLOCKED ENGINEERING PARAMETER — SMOKE EARLY-STOP MONITOR:** weighted BCE
  is used only to exercise infrastructure; final selection remains the later
  frozen stay-balanced validation AUPRC path.
