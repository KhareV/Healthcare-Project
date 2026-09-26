# Health Record Architecture (Product V2)

**This is a synthetic/research record system, not a production clinical
EHR.** Nothing stored here is real patient data; every record a user builds
is either a demo subject from the frozen scientific manifest or a
manually-entered synthetic episode. There is no encryption-at-rest, no
formal audit-compliance program, and no HIPAA/GDPR posture — treat every
document in this system exactly like the rest of the project's synthetic
benchmark data.

## Conceptual shape

```
Authenticated Clerk user (owner_user_id)
        |
        v
Patient Profile (patient_profiles, one per user this pass)
        |
        +-- Conditions           (conditions)
        +-- Encounters           (encounters, each with embedded
        |                         observations + support_intervals)
        |       |
        |       +-- Prediction runs (prediction_runs, one per replayed cutoff)
        |
        +-- Reports              (reports: metadata only; binary content
        |                         lives in object storage, never Mongo)
        |
        +-- Audit events         (audit_events: a lightweight mutation trail)
```

A patient profile is **not** an encounter. One profile owns many
conditions, many encounters, many reports, and (through its encounters)
many prediction-run histories. See
[`CUSTOM_RECORD_FLOW.md`](CUSTOM_RECORD_FLOW.md) for how an encounter
itself is built and served.

## Ownership

Every document in every collection is scoped by `owner_user_id`, the same
verified-Clerk-token identity used everywhere else in this product
(`serving/v2/auth.py::ClerkAuthenticator`). No endpoint ever accepts an
owner or patient id from a request body — `api/v2_app.py` derives
`owner_user_id` from the verified bearer token on every call, and every
Mongo query is filtered by it. A request for another user's data returns
404, identical to a nonexistent record (the same fail-closed pattern used
throughout this product — see
[`AUTH_AND_DATA_BOUNDARIES.md`](AUTH_AND_DATA_BOUNDARIES.md)).

This pass deliberately keeps one profile per Clerk user (not a
multi-patient/clinician mode) — but the unique index is already
`(owner_user_id, patient_id)`, so a future multi-profile mode needs no
index changes, only a UI for selecting among profiles.

## Mongo collections

| Collection | Key fields | Notes |
|---|---|---|
| `patient_profiles` | `patient_id`, `owner_user_id`, `display_name_or_alias`, `age_years`, `sex_category`, `blood_group`, `height_cm`, `weight_kg` | Lazily created on first use (`get_or_create_profile`). Demographics use `age_years`, not a date of birth — consistent with every other demographic field in this synthetic-benchmark product (demo patients, custom records), and a calendar birth date is never otherwise meaningful in this pipeline. |
| `conditions` | `condition_id`, `patient_id`, `owner_user_id`, `name`, `code`, `diagnosed_date`, `status` (`active`/`resolved`/`historical`), `notes` | Pure display context. Never fed into the forecasting models. Editable (PATCH), not just add/delete. No formal medical coding required — `code` is optional free text. |
| `encounters` | `stay_id` (= `encounter_id`), `patient_id`, `owner_user_id`, `patient_alias`, `age_years`, `sex_category`, `intime`/`outtime`, `encounter_type`, `source`, embedded `observations`, embedded `support_intervals` | One document per custom record built via "Enter My Own Record". Observations and support intervals stay embedded (matching the existing schema) rather than a separate collection — see [`CUSTOM_RECORD_FLOW.md`](CUSTOM_RECORD_FLOW.md) for the full build/rehydration mechanics. |
| `prediction_runs` | `owner_user_id`, `stay_id`, `prediction_time`, SOFA/recovery/ICU/support fields, `created_at`/`updated_at` | One snapshot per distinct cutoff actually replayed, upserted (never duplicated) on re-visit. Compact scalar fields only — no raw TreeSHAP matrices are persisted. |
| `reports` | `report_id`, `patient_id`, `owner_user_id`, `title`, `document_type`, `original_filename`, `mime_type`, `storage_key` (internal only, never returned to a client), `size_bytes`, `report_date`, `processing_status` | Metadata only. Binary content lives in `serving/v2/report_storage.py`, addressed by `storage_key` — a client never sees or supplies this key. |
| `audit_events` | `event_id`, `owner_user_id`, `patient_id`, `action`, `target_type`, `target_id`, `timestamp` | Meaningful mutations only (profile updated; condition created/edited/deleted; encounter created; report uploaded/deleted). Never raw content, tokens, or secrets. Best-effort: a write failure here never blocks the mutation it records. |

### Migration-on-read (backward compatibility)

Two shapes predate this pass and must keep working without data loss:

- **Conditions** created before `condition_id`/`name`/`patient_id` existed
  (addressed only by their Mongo `_id`, with a `label`/`diagnosed_year`
  shape). `_normalize_condition` maps these transparently on every read
  (`name` falls back to `label`, `diagnosed_date` is derived from
  `diagnosed_year`, `condition_id` falls back to the `_id` hex string) —
  the stored document is never rewritten just to read it, and both
  `get_condition`/`update_condition`/`delete_condition` accept either the
  new `condition_id` or a legacy Mongo-`_id` string.
- **Encounters** created before `patient_id` existed. The first time such
  an encounter is listed (`list_encounters`), it is backfilled with the
  caller's profile's `patient_id` via a real, additive `update_one` — not
  a destructive rewrite of anything else on the document, and not merely
  an in-memory patch (the next read sees the same `patient_id` without
  re-deriving it).

## Object storage: report files

Binary report content never touches MongoDB. `serving/v2/report_storage.py`
defines a small `ReportStorageProvider` protocol (`save`/`open`/`delete`/
`exists`) and ships exactly one implementation, `LocalReportStorage`, which
writes under a local, gitignored directory (`runtime/uploads/` by default).
**The application runs entirely locally with zero cloud dependency** — an
S3-compatible adapter is a natural drop-in behind the same protocol, not
built in this pass.

Security, enforced in `api/v2_app.py`'s report endpoints before storage is
ever touched:

- **Type allowlist**: `application/pdf`, `image/png`, `image/jpeg` only —
  anything else is rejected with `415`.
- **Size limit**: 10MB (`MAX_REPORT_SIZE_BYTES`) — rejected with `413`.
- **Filename sanitization**: `sanitize_filename` strips any directory
  component and restricts to a conservative safe charset before a
  filename is ever used as part of a storage key.
- **Server-generated storage keys**: a client never supplies or sees the
  key a file is stored under (`f"{owner_user_id}/{random}_{safe_name}"`);
  it is generated fresh on every upload.
- **Defense in depth**: `LocalReportStorage._resolve` independently
  refuses to resolve any key outside its configured root (`Path.
  is_relative_to`), so even a malformed or malicious key can never escape
  the storage directory — this holds regardless of how a key was produced.
- **Owner-scoped access**: every report read/download/delete first looks
  up the Mongo metadata document filtered by the caller's verified
  `owner_user_id`; a mismatched or absent record is `404`, and the
  underlying file is never touched unless that lookup succeeds. A raw
  storage path or key is never returned in any API response — downloads
  always go through the authorized `/health-record/reports/{id}/download`
  endpoint, which streams file bytes with a `Content-Disposition` header
  naming the original filename.

## Report parsing: explicitly out of scope for this pass

Every uploaded report is stored with `processing_status = "NOT_PARSED"`
and stays that way. **Uploaded report content is never used as a model
input.** The forecasting pipeline continues to consume only validated
structured observations/support intervals entered through "Enter My Own
Record" — nothing a report-parsing step might one day extract is inserted
into the clinical timeline automatically. If parsing is implemented in a
future pass, it must produce a separate candidate/extracted-data payload
and require explicit user confirmation before any extracted value could
become a real observation.

## Medication history: deliberately narrow

This project does not implement a general medication-management
subsystem — that is outside its core purpose (forecasting recovery
trajectories from ICU-style vitals/labs/support state, not managing a
pharmacy record). The only "medication" concept the system represents is
the four qualifying vasopressors already used by the frozen scientific
model (`norepinephrine`, `epinephrine`, `dopamine`, `dobutamine`), visible
under **Support / Interventions** in My Health Record. A general
medications table is deferred.

## Prediction history stays derived evidence

`prediction_runs` records what a forecast *was* at a given cutoff, for
display and history browsing — it is never treated as the source of truth
for interactive replay. Opening a cutoff in Patient Replay always
recomputes through the frozen model pipeline
(`V2ServingRuntime.predict`); the persisted snapshot is written as a
side effect of that same call, never read back to short-circuit it.

## Storage boundaries (unchanged from the rest of this product)

1. **Raw record** — MongoDB (profile, conditions, encounter observations/
   support intervals, report metadata) + local report file storage. What a
   user actually entered or uploaded, and nothing else.
2. **Canonical model input** — generated on demand by the frozen
   `SyntheticCanonicalFeatureBuilder`/SOFA provider from that raw record,
   for a specific encounter and cutoff. Never stored as the source of
   truth; regenerated every time.
3. **Derived intelligence** — predictions, SHAP attributions, Trajectory
   Copilot narration. Never fed back into the raw record, and never used
   to "improve" or reinterpret stored observations.

Trajectory Copilot may ground its answers in a user's structured record —
selected encounter, current forecast, prediction history, conditions, and
report *metadata* — but never raw report file content (binary or
extracted text) unless a future parser explicitly supports it and a user
has confirmed the extraction. See
[`TRAJECTORY_COPILOT.md`](TRAJECTORY_COPILOT.md).

## Export and deletion

`GET /health-record/export` returns one structured JSON document: profile,
conditions, encounters (with embedded observations/support intervals),
report *metadata* (never binary content), and prediction history. Nothing
about the export changes model behavior — it is a read-only projection of
the same collections described above.

Individual conditions and individual reports can be deleted
(`DELETE /health-record/conditions/{id}`, `DELETE /health-record/reports/
{id}` — the latter also deletes the underlying stored file). A full
"delete my entire health record" cascade is deliberately **not**
implemented in this pass — it touches five collections plus the report
storage directory and needs careful cascade semantics (in-flight
predictions, the demo/custom stay namespace, etc.) that deserve their own
review rather than being bolted on here.

## Scientific boundary (unchanged)

Nothing in this document touches `src/serving/v2/runtime.py`'s frozen
imports (feature builder, SOFA provider, the four model artifacts, the
isotonic calibrator, the support threshold) or any frozen artifact under
`artifacts/performance_v2/`. Every health-record collection stores raw
input or derived-evidence snapshots; the scientific pipeline is only ever
*served through*, never modified, by any of this. See
[`PRODUCT_ARCHITECTURE.md`](PRODUCT_ARCHITECTURE.md) and the final
evaluation freeze this product layer sits on.
