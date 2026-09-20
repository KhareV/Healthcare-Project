# Phase 11 Validation-Only Search Governance — Sanskruti Review Package

**Owner:** Vedant Khare  
**Mandatory reviewer:** Sanskruti  
**Scope:** framework and synthetic dry-run governance only

## Scientific search spaces

The serialized GRU specification exactly contains hidden dimensions
`{64,128,192}`, layers `{1,2}`, dropout `[0.10,0.40]`, log-uniform learning
rate `[1e-4,3e-3]`, batch size `{64,128}`, and log-uniform weight decay
`[1e-6,1e-3]`. Training is capped at 60 epochs with patience 8, AdamW, and
best-epoch restoration.

Canonical GRU search-space SHA-256:

`057c19d8ffe176ea56dbb798dc369f385ff272bca44924ffec5be54117f8a997`

The serialized XGBoost specification exactly contains max depth 3–8,
log-uniform learning rate 0.01–0.20, subsample and colsample 0.60–1.00,
min-child weight 1–10, alpha 0–1, lambda 0.1–10, and at most 1000 estimators
with validation early stopping and best-iteration retention.

Canonical XGBoost search-space SHA-256:

`90c54ad6c4c523ac8cbeae3078f6cd1c520a3c5439520320f2f94c1322c551c3`

Canonical hashing sorts mapping keys, so key reordering is stable while any
range mutation changes the hash and fails exact scientific validation.

## Sampler and manifests

No scientific sampler or seed is frozen. The candidate materializer therefore
accepts only `run_type=synthetic_dry_run`; requesting scientific generation
hard-fails. Six deterministic dry-run manifests provide equivalent auditable
generation recipes and hashes:

| Family/task | Candidate-list SHA-256 |
|---|---|
| GRU recovery | `8fbf6cd4188a73fa2730ee3112dbf8ade0969af9f09b535ffb17271baf168e85` |
| GRU ICU time | `0b7dc4d2f5c5318c9e79e9f427b3dd67f5072c3a48ab7f46a2815de13f32ea48` |
| GRU support | `93fd264847d41f0373c1a6e9c33c51bf2ee0270363fda6fdf850291621c5aad7` |
| XGB recovery | `2be738abafd8db750c46f61eb07a3c2002b2552376669677dc0e416ecc0b0286` |
| XGB ICU time | `8d4c40a2b568237815e0420438663b8beb1116229c80d95dc2f67681a2a3b8ad` |
| XGB support | `083bda7f9dd0b1421951eb6503aa1e951166b0042d850af9ea54438084786f8f` |

Every recipe materializes 30 unique IDs and 30 unique configuration hashes.
Lists of 29 or 31 fail. Dry-run identifiers cannot be confused with final
scientific candidate IDs and each manifest records zero scientific slots
consumed.

## Budget, retry, and immutability governance

`candidate_id` identifies one of the frozen 30 configurations. `run_id`
identifies an execution attempt. A software retry must reference a failed run
of the same candidate and use the next sequential attempt number. Audit counts
unique candidates separately from attempts, failures, and retries. Epoch or
boosting-iteration history never creates candidates.

Registry attempts outside the manifest, including candidate 31, fail. Once a
search is running/completed/failed/cancelled, changing its search-space hash,
candidate-list hash, validation objective, split hash, or feature/label
versions fails under the same search version.

The existing experiment registry was extended without creating a second
registry. Optional fields now include candidate ID, search version,
candidate-list hash, attempt number, retry reference, detailed failure status,
and validation objective. Existing Phase 4–8 rows remain valid.

## Validation objectives

- Recovery minimizes stay-balanced validation MAE24 only. MAE48 is secondary.
- ICU time minimizes stay-balanced weighted median absolute error in hours.
- Support maximizes stay-balanced validation AUPRC from raw uncalibrated
  probabilities. Calibrated candidates are rejected.

Only valid `COMPLETE` candidates are ranked. Exact primary-metric ties fail
until a tie-break policy is approved. Ranking selects only within one family;
there is no GRU-versus-XGBoost serving decision.

## Test isolation and parity

The access guard allowlists only `train` and `validation`; requesting `test`
raises `HARD FAIL — PHASE 11 TEST ACCESS FORBIDDEN`. A sentinel test proves
test content never enters allowed IDs, metrics, ranking, or registry inputs.

Information-parity manifests require identical row keys, eight bins, dynamic
features, masks, TSLO, static features, preprocessing, split, and eligibility.
Only GRU `sequence_8xf` versus XGBoost `flattened_8xf` representation may
differ. Extra features, nine bins, or different eligibility fail.

This parity is proven synthetically. Real parity remains blocked because no
Sanskruti-owned XGBoost entrypoint or real canonical manifest exists.

## Dry-run audit

The audit command validates all six manifests and reports for each:

```text
planned candidates = 30
unique configurations = 30
candidate hash verified = true
test_accessed = false
scientific_searches_executed = 0
```

No actual model performance or final candidate was fabricated.

## Required decisions and dependencies

- Approve a reproducible scientific configuration sampler and seed.
- Freeze task-specific GRU early-stopping monitors before candidate 1.
- Provide/approve XGBoost entrypoint, flattening, metric, early-stopping rounds,
  artifact contract, and support class weighting.
- Provide real canonical data, split hash, schemas, preprocessing, labels, and
  support endpoint artifacts.
- Approve a deterministic within-family tie-break policy.

Until then:

- **BLOCKED — REAL GRU SEARCH PREREQUISITES INCOMPLETE**
- **BLOCKED — SANSKRUTI XGBOOST SEARCH ENTRYPOINT REQUIRED**

## Reviewer checklist

- Recompute both canonical search-space hashes.
- Confirm every scientific bound and distribution matches the Master Plan.
- Confirm the test sampler cannot authorize scientific candidates.
- Verify all six dry manifests produce exactly 30 unique configurations.
- Verify candidate 31 and post-start mutations fail.
- Verify retry attempts retain candidate identity.
- Confirm validation objectives and raw support probability policy.
- Confirm synthetic information parity and real-parity blocker.
- Confirm no test access, LSTM search, calibration, thresholding, or serving selection occurred.
