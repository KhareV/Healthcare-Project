# Authentication & Data Boundaries

## Two separate concepts: application auth vs. data ownership

**Clerk authenticates application users** — who is allowed to use the
workspace at all. It has no idea what a "custom record" is.

**`owner_user_id` on a custom record** is a data-ownership fact, derived
*only* from a verified Clerk token's `sub` claim
(`serving/v2/auth.py::ClerkAuthenticator.verify`) — never from a request
body field, a custom header, or anything else the client supplies. A
request body containing `"owner_user_id": "someone-else"` is simply
ignored; `api/v2_app.py`'s `create_custom_record` never reads such a field
from the request model at all (`CustomRecordRequest` has no such field).

## Enforcement points

| Endpoint | Custom-stay behavior | Demo-stay behavior |
|---|---|---|
| `POST /custom-records` | requires a valid token; record stamped with the verified `sub` | n/a |
| `GET /custom-records/{id}` | requires a valid token whose `sub` matches the record's owner, else 404 | n/a |
| `POST /predict` | same ownership check | unchanged — no auth required |
| `GET /history` | same ownership check | unchanged — no auth required |
| `POST /ai/recommendation` | same ownership check | unchanged — no auth required |
| `POST /assistant` | same ownership check | unchanged — no auth required |

"Custom-stay" is detected structurally
(`V2ServingRuntime.ephemeral_owner(stay_id) is not None`), not by string
prefix — so this can never accidentally apply to (or skip) a demo subject.
An unauthorized or unauthenticated caller touching someone else's custom
record gets **404**, not 403 — identical to a stay that was never created,
mirroring the fresh-test guard's existing "never let a caller distinguish
sealed from nonexistent" philosophy (`serving/v2/guard.py`).

## Token verification, concretely

`ClerkAuthenticator` derives Clerk's public JWKS URL from the publishable
key alone (`pk_{test,live}_<base64(host)>$` — a public value, safe to
commit as a template) and verifies: RS256 signature against a fetched (and
cached) signing key, `exp`/`nbf` (via PyJWT's default validation), and
`iss` matches the derived host. No secret key is needed for verification —
this is a public-key operation, the same as how any JWT-based backend
verifies a token issued by an identity provider it doesn't control.

## Local-development mode (no Clerk key on the backend)

If `PUBLIC_CLERK_PUBLISHABLE_KEY` is unset in the backend's `.env`,
`ClerkAuthenticator.configured` is `False` and `verify()` always returns
the fixed string `"local-dev-user"` without requiring or inspecting any
token. `GET /health` reports `"auth_mode": "local_dev_no_auth"` in this
state. This is intended only for local development without a Clerk
account configured; it is not a production posture. The frontend makes an
independent, consistent decision (`frontend/src/hooks.server.ts`'s
`clerkConfigured`) based on its own `PUBLIC_CLERK_PUBLISHABLE_KEY` +
`CLERK_SECRET_KEY`, so the two sides can't silently disagree about whether
real authentication is active — but an operator running with the frontend
Clerk-enabled and the backend Clerk-disabled (or vice versa) is a
misconfiguration outside this project's scope to detect automatically;
keep the two `.env` files in sync (see RUNBOOK.md).

## Multi-user isolation, verified

`tests/test_v2_custom_record.py`'s `authed_client_factory`-based tests
create two independent simulated users (Alice, Bob) against a real
signature-verification path (a locally-generated RSA keypair standing in
for Clerk's own key, so no network call is made in tests) and assert:
Alice can read/predict/get-history for her own record; Bob gets 404 on
every one of the same calls against Alice's record; an unauthenticated
caller gets 404 too. Demo-stay predictions succeed with no token at all,
even when Clerk is fully configured — proving the ownership check is
correctly scoped to custom records only.

## What is never logged or persisted

No raw bearer token, no Clerk secret key, and no custom-record content is
ever written to a log statement or a git-tracked file. Custom records
always live in the API process's memory for its lifetime; when
`MONGODB_URI` is configured, the raw structured input (never a derived
representation, and never through a log line) is additionally saved to a
MongoDB collection scoped by the caller's verified `owner_user_id` — see
[`CUSTOM_RECORD_FLOW.md`](CUSTOM_RECORD_FLOW.md)'s "Ephemerality,
durability, and ownership" section for exactly what that does and does not
change.
