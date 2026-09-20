# Phase 4 Canonical Batch Contract

**Status:** synthetic infrastructure contract; production feature, TSLO, static,
and preprocessing contracts remain blocked upstream.

`CanonicalTensorDataset` accepts only Phase-3 canonical records that validate
against `tensor_contract_v1`, the supplied feature-schema reference, and the
subject-level split artifact. `collate_canonical_examples` emits an immutable
`CanonicalBatch` with the following fields.

| Field | Shape/type | Meaning |
|---|---|---|
| `identifiers` | mapping of tuples of length `B` | `subject_id`, `stay_id`, `prediction_time`, `grid_index`, and `split`; retained on CPU by device transfer |
| `sequence` | float32 `[B,8,F]` | oldest-to-newest canonical temporal values in exact feature-schema order |
| `padding_mask` | bool `[B,8]` | canonical pre-ICU padding mask; never inferred from values |
| `observation_mask` | bool `[B,8,F]` | genuine in-bin observations; never regenerated after numerical filling |
| `tslo` | float32 `[B,8,F]` or `None` | canonical TSLO only; `None` in Phase-3 fixtures because its sentinel is not frozen |
| `static_features` | float32 `[B,S]` or `None` | separate static vector only; never repeated over time; `None` in current fixtures |
| `targets` | four float32 tensors `[B]` | `recovery24`, `recovery48`, `icu_time`, and `organ_support`; unavailable values are `NaN`, never zero |
| `eligibility` | four bool tensors `[B]` | independent masks for the four target slots |
| `versions` | string mapping | tensor-contract, timestamp-specification, and feature-schema versions |
| `feature_names` | tuple of length `F` | exact feature order |

Collation rejects inconsistent feature order, versions, optional-field
presence, eligible null targets, and populated ineligible targets. Numerical
null filling is allowed only through the explicitly labelled
`SyntheticTensorizationPolicy`; it does not modify padding or observation
masks and is not a production preprocessor.

`CanonicalBatch.to(device)` transfers all tensor fields and leaves identifiers
and schema metadata intact. Training loss indexes eligible rows before calling
the per-item loss and averages over eligible rows only. A zero-eligible batch
returns no loss and skips its optimization step.

Train loaders may shuffle through their seeded generator. Validation and test
loaders reject shuffling. Worker processes, when configured, receive seeds
derived from the PyTorch loader generator. Phase 4 creates no test loader in
the smoke path and records `test_examples_accessed = 0`.
