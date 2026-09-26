# Product Architecture (Product V2)

How the frozen Performance-V2 scientific system became the authenticated,
end-to-end product on top of it. See
[`CURRENT_UI_INTEGRATION_AUDIT.md`](CURRENT_UI_INTEGRATION_AUDIT.md) for what
existed before this work and the smallest-surface-area plan that guided it.

## Layers

```
 ┌─────────────────────────────────────────────────────────┐
 │  Clerk (auth)                                            │
 │  sessions, sign-in/up, onboarding metadata (unsafeMetadata) │
 └───────────────────────────┬───────────────────────────────┘
                              │ session JWT (short-lived, ~60s)
                              ▼
 ┌─────────────────────────────────────────────────────────┐
 │  SvelteKit 5 frontend (frontend/)                        │
 │  hooks.server.ts   — hard-reload/direct-URL route guard  │
 │  AuthGate.svelte   — soft-navigation route guard (SPA)   │
 │  AuthGate.svelte   — also refreshes the bearer token      │
 │  api.ts            — attaches Authorization: Bearer ...  │
 └───────────────────────────┬───────────────────────────────┘
                              │ HTTP (JSON), same-origin via Vite proxy
                              ▼
 ┌─────────────────────────────────────────────────────────┐
 │  FastAPI (api/v2_app.py)                                 │
 │  ClerkAuthenticator — verifies the JWT via Clerk's public │
 │                        JWKS (serving/v2/auth.py)          │
 │  demo-subjects, /predict, /history, /ai/*, /assistant,    │
 │  /custom-records* — auth is a no-op for demo stay_ids,    │
 │  mandatory + ownership-checked for CUSTOM- stay_ids       │
 └───────────────────────────┬───────────────────────────────┘
                              │
                              ▼
 ┌─────────────────────────────────────────────────────────┐
 │  V2ServingRuntime (serving/v2/runtime.py)                 │
 │  frozen: feature builder, SOFA provider, 4 XGBoost models │
 │  + register_ephemeral_stay / register_ephemeral_support   │
 │    (in-memory only, additive, never touches frozen state) │
 └───────────────────────────┬───────────────────────────────┘
                              │ raw observations/support-intervals only
                              │ (never derived features, never SHAP, never
                              │  model internals) -- optional
                              ▼
 ┌─────────────────────────────────────────────────────────┐
 │  MongoDB (serving/v2/persistence.py) -- optional          │
 │  conditions, encounters (raw input), prediction_runs      │
 │  rehydrate_if_needed() rebuilds the runtime's in-memory    │
 │  representation from raw input after a restart             │
 └─────────────────────────────────────────────────────────┘
```

## Two authorization boundaries, deliberately layered

1. **Frontend route protection** (UX, not security): `hooks.server.ts` for
   hard reloads/direct URLs, `AuthGate.svelte` for in-app soft navigation
   (this app is `ssr = false`, so a link click never touches the server
   hook — see the git history of this file for the bug this caused before
   `AuthGate` existed). Both redirect to `/sign-in` or `/onboarding` as
   appropriate.
2. **Backend authorization** (the real security boundary):
   `serving/v2/auth.py`'s `ClerkAuthenticator` verifies the bearer token's
   signature against Clerk's public JWKS (derived from the publishable
   key — no secret needed for verification) and extracts the `sub` claim
   as the caller's user id. `api/v2_app.py`'s `_authorize_custom_stay`
   helper checks that id against a custom record's stored `owner_user_id`
   for every endpoint that can touch a `CUSTOM-` stay (`/custom-records*`,
   `/predict`, `/history`, `/ai/recommendation`, `/assistant`). Demo
   patients are a no-op for this check — they remain globally readable, as
   before. A missing/invalid/mismatched token gets a **404**, identical to
   an unknown stay — the fail-closed pattern already used for the
   fresh-test guard, so a caller can never learn whether a record exists.

Frontend guards can be bypassed by anyone hitting the API directly; backend
authorization cannot. Both exist because they serve different purposes (UX
vs. security), not because either alone would suffice.

## Local-development mode

If the backend has no `PUBLIC_CLERK_PUBLISHABLE_KEY` configured,
`ClerkAuthenticator.configured` is `False` and every call succeeds as a
fixed `local-dev-user` owner, with no token required. `GET /health` reports
`"auth_mode": "local_dev_no_auth"` in this case so the frontend (or a
developer) can tell at a glance. This must only ever happen in local
development — see [`AUTH_AND_DATA_BOUNDARIES.md`](AUTH_AND_DATA_BOUNDARIES.md).

## Custom-record storage: in-memory serving, optional durability

`V2ServingRuntime.register_ephemeral_stay`/`register_ephemeral_support` add
entries to plain in-memory dicts/tuples already used to hold the frozen
corpus — the same feature builder, SOFA provider, and models serve both
without knowing the difference. This part is unconditional: serving never
reads from a database.

Durability across restarts is a separate, optional layer
(`serving/v2/persistence.py`, MongoDB). When `MONGODB_URI` is configured,
the *raw* structured input a user entered (never a derived representation)
is saved alongside registration, and the next request for that stay_id
after a restart calls `rehydrate_if_needed`, which rebuilds the in-memory
representation from that raw input via the exact same registration path
used at creation. Without `MONGODB_URI`, nothing changes from the original
design: nothing is written to disk, and a process restart discards every
custom record. See [`CUSTOM_RECORD_FLOW.md`](CUSTOM_RECORD_FLOW.md) for the
full mechanics, and `GET /health`'s `persistence_mode` field to check which
mode is active.

## What stays untouched

`src/serving/v2/runtime.py`'s frozen imports (feature builder, SOFA
provider, `data.timestamps`, the four model artifacts, the isotonic
calibrator, the support threshold) are never modified by anything in this
document — only *served through*, for a wider set of `stay_id`s. See
[`PERFORMANCE_V2_COMPLETE`'s final evaluation freeze](../performance_v2/FINAL_V2_TEST_EVALUATION.md)
for the immutable scientific baseline this product layer sits on.
