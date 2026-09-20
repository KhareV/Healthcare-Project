# Phase 1 Timestamp Contract — Reviewer Package

**Owner:** Vedant Khare  
**Mandatory reviewer:** Sanskruti Satish Shete  
**Informed downstream owner:** Pulkit  
**Specification:** `configs/timestamp_spec_v1.yaml`  
**Specification SHA-256:** `de3a424fcd60affda178afbde49a0c83fd51610f7682cf73306d124454b18a9b`

## Review decision requested

Sanskruti should verify that the typed retained-stay input and emitted-row contract are compatible with the cohort, feature-builder, recovery-label, and ICU-time-label boundaries. No scientific change is proposed.

External MIMIC schema mapping remains **BLOCKED — UPSTREAM SCHEMA MAPPING REQUIRED**. `subject_id`, `stay_id`, `intime`, and `outtime` are internal **ENGINEERING CONTRACT FIELD NAME** values. The generator does not query MIMIC or select the first ICU stay.

## Authoritative API

The only implementation is `src/data/timestamps.py` (SHA-256 at completion: `c7bd20e6d475ab3699fad5cbf829049311b010407bb387338dc35db5ed718e2c`). It exposes:

- `RetainedICUStay(subject_id, stay_id, intime, outtime)` — typed upstream input;
- `validate_retained_stay` / `validate_retained_stays` — identifiers, datetimes, positive ICU duration, one retained stay/subject, and duplicate-stay rejection;
- `generate_candidate_cutoffs` — exactly `k=0..11` candidates;
- `retain_legal_cutoffs` — validates the frozen grid and applies inclusive `t <= outtime-6h`;
- `calculate_horizon_availability` — time-only 24h/48h follow-up flags;
- `generate_prediction_rows_for_stay` — one-stay canonical rows;
- `generate_prediction_timestamps` — collection validation and deterministic subject/stay/time ordering;
- `validate_prediction_rows` — independent output/grid/boundary/follow-up validation.

Naive and timezone-aware Python `datetime` values are supported, but a stay's `intime` and `outtime` must use consistent awareness. Arithmetic is elapsed-time arithmetic; no rounding or wall-clock snapping occurs.

## Emitted row schema

`PredictionTimestamp` contains:

- `subject_id`
- `stay_id`
- `prediction_time`
- `grid_index`
- `icu_elapsed_hours`
- `recovery24_followup_available`
- `recovery48_followup_available`
- `support24_full_followup_available`
- `icu_time_temporally_eligible`
- `timestamp_spec_version`

The support flag means only complete 24-hour temporal follow-up. It is not `organ_support_eligible`, is not a negative label, and contains no OFF/ON state logic.

## Mathematical proof of the cap

Candidate indices are the inclusive integer set `{0,...,11}`, whose cardinality is 12. Filtering by `t_k <= outtime-6h` can only remove candidates; it cannot add any. Therefore each retained stay emits between 0 and 12 rows. No post-cap sampling or secondary cadence exists.

## Synthetic boundary evidence

`tests/fixtures/timestamps/cases.json` is a manually readable, non-clinical fixture set covering:

- exact first cutoff and `t == outtime-6h` equality;
- exactly 12 candidates and a much longer stay still capped at 12;
- exactly four legal cutoffs;
- no legal cutoff when the first candidate is one microsecond beyond the boundary;
- non-wall-clock-aligned ICU intimes such as `03:17`;
- exact `t+24h == outtime` and `t+48h == outtime` equality;
- 24h available/48h unavailable and both unavailable;
- multiple subjects, duplicate input, and invalid duration.

The test suite additionally verifies deterministic repeated calls, malformed-candidate rejection, timezone-aware relative arithmetic, missing times, one-stay-per-subject enforcement, and independence from arbitrary future labs/SOFA/support-event payloads.

## Explicit exclusions

The implementation contains no feature binning/padding tensor, MIMIC mapping, hospital admission/discharge anchor, SOFA or recovery calculation, ICU-time target calculation, organ-support event/label logic, split logic, model dependency, performance inspection, randomness, or test access.

## Reproducibility status

- Timestamp spec is versioned and exact-byte hashed above.
- Synthetic fixture SHA-256: `a8b72b51be52e3739c37cc0256a81076f4727970f3746aab99294d878199d051`.
- Test module SHA-256 at completion: `2c36f107ed250c98bc338a09891ef069ed78cfb8b1323a43dcbc00e468db9515`.
- No timestamp dataset artifact was written, so no generated-artifact hash applies.
- Source commit is unavailable because the project directory is not yet a Git repository.

Pulkit should consume the same emitted cutoff contract for support-state labeling and replay cutoff validation; he must not reinterpret full follow-up as final support eligibility.

