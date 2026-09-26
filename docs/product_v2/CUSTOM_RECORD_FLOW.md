# Custom Record Flow ("Enter My Own Record")

How a manually-entered record becomes a real prediction through the frozen
V2 pipeline. Code: `src/serving/v2/custom_record.py`,
`src/serving/v2/runtime.py` (`register_ephemeral_stay`/
`register_ephemeral_support`), `api/v2_app.py` (`/custom-records*`),
`frontend/src/routes/patients/custom/+page.svelte`.

## What a user provides

- Patient alias, age, sex (display metadata only — not model features
  beyond what the frozen static contract already uses for demo subjects).
- One or more **vitals/labs observations**: a canonical concept (from the
  14 supported — see below), a value, and *hours since a synthetic ICU
  admission* (not a calendar timestamp — the calendar date is never
  meaningful anywhere in this pipeline).
- Optionally, **organ-support intervals**: a vasopressor (agent + rate +
  start/end hour) or invasive-ventilation interval (start/end hour). Left
  empty, the record is modeled as "no organ support observed" — a real,
  valid state, not an error.

## What happens on submit (`build_custom_record`)

1. Validate every observation against the frozen feature schema
   (`configs/synthetic/feature_schema_v2.json`): concept must be supported,
   value finite, `glasgow_coma_scale` a whole number (SOFA's neurological
   component requires an integer), hour strictly `> 0` (an hour-0 event is
   indistinguishable from pre-admission padding) and `<= 90` (the legal
   cutoff grid never extends past `24 + 6*11 = 90` hours regardless of
   anything else — an observation past that could never be visible to any
   servable cutoff).
2. Generate a `CUSTOM-<random>` stay id and `CUSTOM-SUBJ-<random>` subject
   id — a namespace that can never collide with the demo manifest
   (`SYN-*`) or the sealed fresh-test cohort.
3. Build canonical event rows (unit + provenance_id looked up
   automatically from the schema — the caller never supplies or can get
   these wrong) and, if any were entered, support-interval rows in the
   exact `support_intervals.jsonl` shape (`configs/event_dict_v2.yaml`'s
   `CLOSED_OPEN` convention: active iff `start <= cutoff < end`; "currently
   active" is represented as an end far beyond the grid ceiling, which the
   SOFA support provider clamps to the query cutoff automatically — never
   a future-data leak).
4. Compute real legal cutoffs via the same
   `data.timestamps.generate_prediction_rows_for_stay` the frozen demo
   manifest was built with.
5. Register everything into the running `V2ServingRuntime`: the same
   `_events_by_stay`/`_statics_by_stay`/`_supports_by_stay` dicts, the same
   `feature_builder._statics` dict, and the same `sofa_provider.stays`/
   `.history`/`.support_provider` objects the frozen corpus already
   populated at process start. From this point, `/predict`, `/history`,
   `/ai/recommendation`, and `/assistant` serve the record identically to
   a demo subject — no branch anywhere in those endpoints treats it
   differently except the ownership check (see
   [`AUTH_AND_DATA_BOUNDARIES.md`](AUTH_AND_DATA_BOUNDARIES.md)).
6. Run one real prediction at the first legal cutoff to compute an honest
   readiness summary (see below) — reusing the real pipeline rather than
   re-deriving bin/observation math separately, so these numbers can never
   drift from what `/predict` itself reports.

## The 14 supported concepts

`heart_rate`, `mean_arterial_pressure`, `systolic_blood_pressure`,
`diastolic_blood_pressure`, `respiratory_rate`, `oxygen_saturation`,
`temperature`, `pao2`, `fio2`, `creatinine`, `platelet_count`,
`bilirubin_total`, `glasgow_coma_scale`, `lactate` — every raw temporal
channel in the frozen feature schema *except* `urine_output_volume`.

## Why urine output is not supported

The renal SOFA component (`data/synthetic/sofa.py::_urine_candidate`)
requires a **fully gapless chain** of interval-shaped readings whose union
tiles *exactly* the 24-hour window before each cutoff — any partial/sparse
coverage is fine (it silently falls back to creatinine alone), but any
malformed or non-contiguous row for this concept raises a hard error for
**every** cutoff, not just the ones near it. Building genuine gapless
24-hour coverage from episodic manual entry would mean synthesizing volumes
for hours the user never actually reported — fabricating data, which this
project does not do anywhere. Renal SOFA still scores correctly from
creatinine alone; this is a documented, deliberate scope boundary, not a
bug.

## Data readiness (never a confidence score)

The response's `readiness_summary` reports facts, not a probability of
correctness:

- `observations_entered`, `concepts_represented` / `concepts_total`
- `earliest_observation_hour` / `latest_observation_hour`
- `legal_cutoffs` (always `>= 1` once any valid observation exists)
- `first_cutoff_observed_bins` / `first_cutoff_total_bins` — from a real
  `/predict` call, not re-derived
- `sofa_components_observed` / `sofa_components_missing` — which of the six
  SOFA subcomponents have at least one contributing concept represented
- `support_state_entered` — whether vasopressor/ventilation intervals exist
- `missing_concepts`, plus free-text `warnings` for the UI

`data_readiness` is always `"READY"` once validation passes — the model is
explicitly designed to tolerate missing bins (the same
observation-mask/data-quality machinery every prediction already reports),
so a sparse record is a real, valid, low-completeness case, not an error
state requiring a fake pass/fail gate.

## Ephemerality and ownership

In-memory only (`V2ServingRuntime`'s plain Python dicts/tuples) — a process
restart discards everything, and nothing is ever written to a git-tracked
path. Every record is stamped with the `owner_user_id` of the authenticated
caller who created it (never a value the request body can supply); see
[`AUTH_AND_DATA_BOUNDARIES.md`](AUTH_AND_DATA_BOUNDARIES.md) for how that's
enforced on every subsequent read/predict/history/assistant call.
