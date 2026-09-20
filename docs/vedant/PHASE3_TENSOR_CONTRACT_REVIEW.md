# Phase 3 Tensor Contract — Reviewer Package

**Owner:** Vedant Khare  
**Mandatory reviewers:** Sanskruti Satish Shete and Pulkit  
**Contract:** `configs/tensor_contract_v1.json`  
**Contract SHA-256:** `d998456430d632b81778d52f33a9b98214e3599263d443b191f5f8f75e307e9e`

## Review decision requested

This phase freezes the shape and semantic interface without inventing the upstream feature schema. Sanskruti should approve feature-builder compatibility; Pulkit should approve serialization, identifiers, version metadata, and eventual serving consumption.

## Frozen interface

- Model-facing temporal shape is `[B,8,F]`, ordered oldest-to-newest across eight six-hour bins covering `(t-48h,t]`.
- `F` and feature order come only from Sanskruti's `feature_schema_v1`; the real value/order remain unresolved.
- Padding mask is `[B,8]` boolean with `true = pre-ICU padded bin`; padding must be a prefix.
- Observation mask is `[B,8,F]` boolean and is true only for a genuine in-bin observation. Numerical imputation must never flip it true.
- TSLO is logically `[B,8,F]` in hours and feature-aligned. Its no-observation representation is not frozen, so fixture `tslo_hours` remains null and explicitly blocked.
- Static inputs are logically `[B,S]`, never repeated silently across time. `S` and static order are not frozen, so fixture static inputs remain null and explicitly blocked.
- Four eligibility masks are independent: recovery24, recovery48, ICU time, and organ support.
- Ineligible synthetic targets must be null; eligible fixture targets must be present. Eligibility is never inferred from nullness.
- Canonical uniqueness key is `(stay_id,prediction_time)`.
- GRU preserves each temporal field as `[8,F]`; XGBoost's comparison view row-major flattens exactly the same values/masks. No XGBoost-only information exists.

## Authoritative implementation

`src/data/schema.py` (SHA-256: `4d5dde4338428962361cab2b23b098c6f586f0371441aa1e13c2afe369c49057`) provides:

- external `FeatureSchemaReference` with symbolic/dynamic `F`;
- immutable canonical example/dataset, eligibility, and target contracts;
- shape, dtype, padding, observation-mask, version, target-mask, identifier, split, and uniqueness validation;
- deterministic canonical ordering and UTF-8 JSON serialization/deserialization;
- reversible GRU/XGBoost temporal information views for parity assertions;
- no feature construction, forward fill, imputation, normalization, label derivation, batching, or model code.

The JSON choice is **UNLOCKED ENGINEERING PARAMETER — SYNTHETIC FIXTURE SERIALIZATION**. It does not select the final production data format. JSON `null` represents only unprocessed synthetic missing values; it does not claim a fitted imputation value or TSLO sentinel.

## Synthetic fixture

`tests/fixtures/canonical/canonical_synthetic_v1.json` is explicitly tagged `SYNTHETIC_CANONICAL_FIXTURE_NOT_REAL_CLINICAL_DATA`.

- SHA-256: `9269c98a0e5942f464d80f29054c174fc462223621c9f9ec8e136b0705c900df`.
- Deterministic canonical-serialization SHA-256: `36155cac6c4c146618f9f57910f8471b6b0b65c12d670c2ccd037f39ee97764a`.
- Placeholder schema: `synthetic_feature_schema_v1` with `feature_0`, `feature_1`; neither `F=2` nor these names are final.
- Six examples across three synthetic subjects/stays and train/validation/test era metadata.
- Snapshot counts: Stay A = 2, Stay B = 3, Stay C = 1.
- Source examples are intentionally out of order; deserialization canonicalizes them deterministically.

Covered cases include earliest-cutoff four-bin padding, full 48-hour ICU history, genuine in-ICU missingness, padding versus missingness, a feature with no in-lookback observation, boundary assertions at `t-48h`/just after/`t`, independent recovery availability, ICU-time eligibility with recovery censoring, manually assigned support positive/negative/ineligible targets, unequal stay snapshot counts, and multiple coarse-era splits.

The support values are manual interface fixtures only. No OFF-to-ON logic is implemented.

## Blocked upstream items

- **BLOCKED — FINAL FEATURE CONTRACT FROM SANSKRUTI REQUIRED:** real `feature_schema_v1`, feature order, actual `F`, observation-mask convention confirmation, feature units/aggregation metadata, and schema version/hash.
- **BLOCKED — TSLO NO-OBSERVATION REPRESENTATION REQUIRED:** exact sentinel/encoding and provenance from the frozen feature contract.
- **BLOCKED — STATIC FEATURE CONTRACT REQUIRED:** static feature names/order, dimension `S`, missingness representation, and preprocessing boundary.
- **BLOCKED — LABEL/EVENT CONTRACT REFERENCES REQUIRED:** final `label_spec_v1` and Pulkit's `event_dict_v1` versions for real integration metadata.
- **BLOCKED — PRODUCTION SERIALIZATION FORMAT REVIEW REQUIRED:** current canonical JSON is synthetic-test-only.
- Source commit is unavailable because the directory is not a Git repository.

## Sanskruti checklist

- Confirm `[8,F]` and exact feature-schema reference mechanism.
- Confirm oldest-to-newest bin order and `(t-48h,t]` boundary wording.
- Confirm padding-prefix and genuine-observation-mask semantics.
- Supply/freeze TSLO no-observation representation and static-feature contract.
- Confirm no feature name/dimension in this fixture is interpreted as real.

## Pulkit checklist

- Confirm `(subject_id,stay_id,prediction_time,grid_index)` and version metadata are sufficient for serving handoff.
- Confirm deterministic JSON is acceptable for mocks only and nominate the production serialization contract later.
- Confirm the support target/eligibility pair supports positive, negative, and censored/ineligible states without implementing event logic here.
- Confirm PredictionPipeline will consume the same validated representation rather than rebuilding masks or feature order.

Production Phase 4 integration cannot freeze until the feature, TSLO, and static contracts are supplied. Synthetic Dataset/DataLoader work may use this fixture only when separately authorized.
