# Pulkit Phase 7 — Raw-History Truncation and Preprocessing Review

## 1. Pulkit Phase 7 summary

Raw-history truncation and canonical-builder/preprocessor integration are implemented and synthetically verified. This is an engineering framework result, not a real-data scientific equivalence claim.

## 2. Repository state before changes

The Phase-6 immutable artifact pipeline existed and the pre-change suite passed 680 tests. No production `build_features()`, production raw-timeline contract, frozen `feature_schema_v1`, production `processed_schema`, or real selected-model manifest was present.

## 3. Sanskruti feature-handoff audit

The repository was searched for builders, schemas, dictionaries, preprocessors, masks, TSLO, statics, raw timelines, flattening, and training manifests. `data.real_adapter` can validate a future handoff but does not build features. The required production handoff remains absent.

## 4. Files created

- `src/serving/history.py`
- `src/serving/preprocessing.py`
- Four Phase-7 test modules
- Explicit synthetic timeline and feature-schema fixtures
- This review record

## 5. Files modified

`src/serving/interfaces.py` and `src/serving/pipeline.py` now request canonical cutoff-built input with selected family and feature version. Phase-5/6 tests and shared test helpers were advanced to the Phase-7 boundary.

## 6. Raw timeline contract

Storage remains an unlocked parameter. A narrow `HistoryProvider` protocol is defined. The only concrete adapter is an explicitly synthetic, in-memory fixture adapter; it is not a CSV, JSON, database, Parquet, or MIMIC production decision.

## 7. Verified event-time semantics

Production semantics are unresolved. The fixture explicitly maps `synthetic_available_time` and declares synthetic point-event availability at that time. The concrete truncator refuses real or stateful-interval contracts rather than guessing a field or interval rule.

## 8. History truncation algorithm

Retrieve one retained stay, call the existing timestamp generator, match the requested legal cutoff without rounding/localizing, and retain mapped point events with time `<= t`. No lookback, binning, aggregation, imputation, masks, TSLO, tensorization, or prediction is performed in `history.py`.

## 9. Lookback boundary result

Synthetic canonical-builder tests verify `(t-48h,t]`: exact `t-48h` is excluded, `t-48h+1s` can populate the first bin, exact `t` is included, and `t+1s` is excluded before feature construction.

## 10. Pre-ICU padding result

At the earliest legal cutoff, the first four nominal bins precede ICU admission and remain structural padding. They are distinct from the first active, unobserved ICU bin.

## 11. Observation mask result

Masks come only from the injected canonical builder. Synthetic tests verify genuine observations alone set the mask and old/pre-ICU values do not.

## 12. TSLO result

TSLO comes only from the injected canonical builder. The synthetic fixture freezes `999.0` for engineering tests; this is not the production sentinel. A sole `t-60h` observation leaves all in-window values missing, masks false, and TSLO at the synthetic sentinel.

## 13. Canonical feature-builder integration

`CanonicalHistoryInputProvider` invokes one injected builder object after truncation. Direct/modeling and serving tests use that identical object, and a spy proves both calls occur. No serving-specific scientific builder exists.

## 14. Preprocessor artifact status

The Phase-6 task bundle still loads the exact hash-verified preprocessor artifact selected by the manifest. The synthetic artifacts are identity transforms for orchestration tests and explicitly unauthorized for real serving. Production fitted artifacts are absent.

## 15. Train-fit provenance status

Phase-6 requires `fit_partition == "train"`. A missing provenance field blocks bundle construction before loaders, transformation, or prediction.

## 16. No-refit evidence

A hostile transform-only spy exposes `fit` and `fit_transform` traps. Serving calls only `transform`; fit counters remain zero. No preprocessing statistics or mappings are recomputed.

## 17. Feature-schema and order validation

The builder's bound `FeatureSchemaReference` must exactly equal the serving schema. Canonical tensor validation rejects wrong dimensions; tests reject wrong version, reordered channels, extra features, and missing channels. Production schema-hash binding remains blocked with the missing handoff.

## 18. Full-history versus explicit-truncation test

Full stored history with internal truncation produces exactly the same canonical family view as a separately constructed timeline containing only rows through `t`.

## 19. Future-row perturbation test

Deleting, reversing, mutating, and adding extreme synthetic rows after `t` leaves the model-facing input at `t` exactly equal. The assertion compares input structures, not predictions.

## 20. Modeling versus serving equality test

Direct `builder -> frozen transform` equals `store -> truncator -> same builder -> same frozen transform` exactly on the synthetic fixture. This does not establish real equality.

## 21. XGBoost / GRU information parity

The existing `temporal_information_views()` utility is used. Unflattening XGBoost values, observation masks, and TSLO reproduces the GRU sequence exactly; padding and statics are identical and neither family receives extra fields.

## 22. Prediction-lookup nonuse

Repository/source audit and tests find no serving dependency on `predictions.csv`, `final_predictions`, or a prediction lookup. Every request rebuilds its cutoff-specific input; no cache was added.

## 23. Data-quality output status

Counts are derived from canonical padding and observation masks. They remain missingness/data-quality metadata and contain no confidence or reliability score. Non-sensitive history trace objects record cutoff, source version/hash, counts, and maximum retained time.

## 24. Tests added

Tests cover legal/illegal cutoffs, boundary behavior, isolation, immutability, stateful fail-closed behavior, future perturbation, explicit truncation, row ordering, lookback, padding/missingness, TSLO, targets/audit fields, schema attacks, parity, no lookup, no refit, provenance, artifact drift, and Phase-6 pipeline integration.

## 25. Exact commands executed

```text
pytest -q ...                                      # unavailable executable
python3 -m pytest -q ...                           # demonstrated missing PYTHONPATH
PYTHONPATH=src:tests python3 -m pytest -q <focused Phase-6/7 files>
PYTHONPATH=src:tests python3 -m pytest -q
rg -n -i "predictions..." src/serving tests/fixtures/serving
rg -n "def build_features|build_features" src docs configs
find ... feature/processed schema and selected-model manifest audits
PYTHONPATH=src:tests python3 -m compileall -q src tests
```

The first two invocations are retained as environment diagnostics; the repository requires `python3 -m pytest` and `PYTHONPATH=src:tests` in this shell.

## 26. Test results

Focused integration: 41 passed before the final added missing-channel case. Full regression before that final case: 707 passed with one stale Phase-5 assertion, which was advanced to prohibit Phase 8+. Final results are recorded in the completion response after the last full run. Compileall passed.

## 27. Real training-serving equivalence status

**BLOCKED — SANSKRUTI PRODUCTION FEATURE/PREPROCESSOR HANDOFF REQUIRED.** Synthetic equality must not be represented as real equality.

## 28. Blocked / unlocked parameters

Blocked: production event-time and stateful semantics, `processed_schema`, raw fixture schema, production builder, feature order/dimension, units/aggregations, mask convention, TSLO sentinel, statics, model-facing serialization, fitted preprocessors, and real direct-path fixture. Unlocked: demo fixture storage format only.

## 29. Sanskruti review notes

Reviewer must verify: raw schema; mapped availability time; truncate-before-build; exact builder identity; future exclusion; `(t-48h,t]`; no pre-lookback or pre-ICU fill; no duplicated unit/duplicate/aggregation logic; schema order; and adapter compatibility. Production approval is pending.

## 30. Vedant review notes

Reviewer must verify: reuse of timestamp legality; preserved Phase-6 checks; exact transform artifact; zero fitting; unchanged tensor contract; family parity; exclusion of support audit/future fields; version bindings; and direct/serving equality. Approval is pending.

## 31. Definition-of-done checklist

Framework items are complete: audit, Phase-6 reuse, history module, stay isolation, pre-build cutoff, structural leakage tests, shared builder injection, frozen transform use, schema checks, parity, immutability, trace/data quality, tests, and no Phase-8+ implementation. All real-data items and both production reviews remain unchecked because their authoritative artifacts are unavailable.

## 32. Readiness for Pulkit Phase 8

The synthetic Phase-7 framework is ready to hand off after review. Real serving is not ready. Phase 8 must not treat these fixtures as selected production models or implement explanations against them as clinical artifacts.
