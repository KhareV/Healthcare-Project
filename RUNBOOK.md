# Operational Runbook

## V2 final release demo application

This section is self-contained: it does not require reading the historical
Benchmark-v1 sections below, and needs no developer-specific local paths.

### 1. Prerequisites

```bash
python3 --version   # tested on CPython 3.9
python3 -c "import fastapi, xgboost, shap, matplotlib, httpx, jwt"   # must import cleanly
```

If any import fails, install the missing package with your environment's
package manager (no project-wide lock is currently pinned for these).

`pymongo` (`pip3 install --user pymongo`) is needed only if you configure
`MONGODB_URI` (step 6, persistent longitudinal records) -- without it, the
persistence layer is simply disabled and everything else is unaffected.

### 2. Verify V2 scientific artifacts are present and unchanged

```bash
PYTHONPATH=src:tests:. python3 -m pytest -q tests/test_v2_serving.py tests/test_v2_dashboard.py
```

### 3. Start the V2 API (terminal 1)

```bash
PYTHONPATH=src:. python3 -m uvicorn api.v2_app:app --factory --host 127.0.0.1 --port 8010
```

```bash
curl --fail http://127.0.0.1:8010/health
curl -s -X POST http://127.0.0.1:8010/predict \
  -H "Content-Type: application/json" \
  -d '{"stay_id":"SYN-E-00000001","prediction_time":"2101-05-31T06:17:26Z"}'
```

The demo fixture's exact stay IDs and legal cutoffs are listed in
`configs/performance_v2/v2_demo_manifest_v1.json`.

### 4. Start the V2 dashboard — SvelteKit frontend (terminal 2)

The primary dashboard is the SvelteKit app in `frontend/`, reusing its
existing component library and design system, restyled for this project.

```bash
cd frontend
npm install        # first time only
npm run dev
```

Open `http://localhost:5173/`. The dev server proxies `/api/*` to
`http://localhost:8010` (see `frontend/vite.config.ts`); no extra
configuration is needed as long as the V2 API from step 3 is running.

For a production-style build: `npm run build && npm run preview` (serves
the static build; `VITE_API_BASE_URL` can be set at build time to point at
a non-default API origin).

**Alternative — server-rendered Python dashboard (no Node required):**

```bash
V2_API_BASE_URL=http://127.0.0.1:8010 \
PYTHONPATH=src:. python3 -m uvicorn dashboard.v2_app:app --factory --host 127.0.0.1 --port 8511
```

Open `http://127.0.0.1:8511/`. This lightweight, dependency-free HTML
dashboard remains fully functional and covers the same five required views;
it is kept as a Node-free fallback alongside the SvelteKit frontend above.

Either dashboard exposes: Patient Replay, Forecast Details, Model
Performance, Explainability (+ an AI-generated research-note synthesis in
the SvelteKit frontend), and Data Quality & Provenance, plus a patient/stay
selector across the 3+ demo subjects.

### 5. Configure authentication (optional)

The dashboard routes (`/overview`, `/patients`, `/trends`, `/research/*`,
`/ai/*`, `/system/*`, `/demo`) sit behind Clerk-based authentication using
the community SvelteKit SDK `svelte-clerk`, enforced on **both** soft
client-side navigation (`AuthGate.svelte`) and hard server-side reload
(`hooks.server.ts`). Authentication is **optional at the infrastructure
level**: with no keys configured, the app runs fully open (no redirect to
sign-in) so the product remains inspectable without a Clerk account — the
same graceful-degradation pattern used for Groq/Kokoro.

To enable it, configure the key in **both** the frontend and the backend
(they must agree — see the note at the end of this step):

```bash
cd frontend
cp .env.example .env
# create a free application at https://dashboard.clerk.com, then paste:
# PUBLIC_CLERK_PUBLISHABLE_KEY=pk_test_...
# CLERK_SECRET_KEY=sk_test_...
```

```bash
# repository root
cp .env.example .env
# paste the SAME publishable key (this is a public value, not a secret):
# PUBLIC_CLERK_PUBLISHABLE_KEY=pk_test_...
```

Restart both the API (terminal 1) and `npm run dev` (terminal 2) after
editing either `.env`. With both configured:

1. Visiting any dashboard route while signed out redirects to `/sign-in`
   (enforced both client- and server-side — a real security boundary, not
   just a UI nicety).
2. After signing up/in, a first-time user is nudged to `/onboarding`
   (role → research disclaimer → data-source mode → done), tracked in
   Clerk's `unsafeMetadata` on the user object — no extra backend or
   database needed for this.
3. `GET /health` on the API reports `"auth_mode": "clerk_verified"`. Every
   custom-record endpoint (`/custom-records*`, and `/predict`/`/history`/
   `/assistant` when the target is a custom record) now verifies a real
   Clerk session token server-side (RS256 against Clerk's public JWKS, no
   secret key needed for verification) and binds/checks the record's
   `owner_user_id` — see [`docs/product_v2/AUTH_AND_DATA_BOUNDARIES.md`](docs/product_v2/AUTH_AND_DATA_BOUNDARIES.md).
4. Sign out from the user menu at the bottom of the sidebar.

Both frontend keys are required together — setting only one leaves the
client trying to render Clerk UI while the server-side redirect stays
disabled. If the backend's `PUBLIC_CLERK_PUBLISHABLE_KEY` is left unset
while the frontend has Clerk enabled, the API falls back to
`"auth_mode": "local_dev_no_auth"` and stamps every custom record with a
single fixed local-dev owner (`local-dev-user`) instead of a real per-user
identity — fine for solo local development, but keep the two `.env` files
in sync for anything resembling a real multi-user demo.

### 6. Configure My Health Record persistence (optional)

By default, custom records ("Enter My Own Record") live only in the API
process's memory and are discarded on restart, and My Health Record
(`/health-record`) has nothing to show. To make everything durable,
configure a MongoDB connection string:

```bash
pip3 install --user pymongo
# repository root .env:
cp .env.example .env   # if you have not already, from step 5
# add: MONGODB_URI=mongodb+srv://user:password@your-cluster.mongodb.net/
```

Never commit a real `MONGODB_URI` or paste one into a chat, issue, or log --
`.env` is gitignored specifically so this stays local. Restart the API.
`GET /health` now reports `"persistence_mode": "mongodb"` (it reports
`"in_memory_only"`, and the original behavior, if the cluster is unset or
unreachable at startup -- persistence is never a hard dependency; a
connection failure at startup is logged and the server keeps running
exactly as before). With it configured, **My Health Record**
(`/health-record`) becomes a consolidated, longitudinal record for the
signed-in user:

1. A **patient profile** (display name, age, sex, blood group, height,
   weight) is created automatically on first use and editable from the
   Overview tab -- pure display context, never a model input.
2. **Conditions** (name, optional code, diagnosed date, status, notes) can
   be added, edited, and deleted.
3. **Encounters** -- every custom record built via "Enter My Own Record" --
   are linked to that one profile and listed with a link back into Patient
   Replay. If this process later restarts, the *next* request for an
   encounter's stay_id transparently rebuilds it into the in-memory serving
   runtime from its stored raw input -- through the exact same registration
   path used at creation time. Scientific derived state (features, SHAP,
   model internals) is never stored; only the raw input a user actually
   typed in.
4. **Vitals & Labs** and **Support / Interventions** tabs show every
   observation and organ-support interval across every encounter,
   chronologically, with filters -- a read-side view over the same embedded
   data, not a new source of truth.
5. **Reports** (PDF/PNG/JPEG, 10MB limit) can be uploaded, downloaded, and
   deleted. Binary content is stored locally under `runtime/uploads/`
   (gitignored), never in MongoDB and never as a model input -- every
   report's `processing_status` stays `NOT_PARSED` in this pass. See
   [`docs/product_v2/HEALTH_RECORD_ARCHITECTURE.md`](docs/product_v2/HEALTH_RECORD_ARCHITECTURE.md)
   for the storage-security details (type/size limits, filename
   sanitization, path-traversal defenses, owner-scoped downloads).
6. **Prediction History** lists every cutoff actually replayed, across
   every encounter -- derived evidence, never the source of truth for
   replay (which always recomputes through the frozen model pipeline).
7. **Export my record** downloads one structured JSON file (profile,
   conditions, encounters, report metadata, prediction history -- never
   report binaries).

Everything stored is scoped by the same verified Clerk `owner_user_id` used
everywhere else in this product; a request body can never set it. Keep all
persisted data synthetic/demo-only -- this layer has no encryption-at-rest
or formal compliance program, so treat it the same as the rest of this
project's synthetic benchmark data, not real patient information.

### 7. Start the Kokoro narration service (terminal 3, optional)

The demo and AI + SHAP pages can read AI research notes aloud with
[Kokoro](https://github.com/thewh1teagle/kokoro-onnx), a small open-weight
TTS model. It runs as its own isolated FastAPI process because
`kokoro-onnx`'s ONNX runtime needs a newer Python than the rest of this
project is pinned to — it never touches the main `src/` environment or any
scientific artifact.

```bash
# one-time setup
brew install python@3.12   # any Python 3.11+ works
/opt/homebrew/bin/python3.12 -m venv .venv-tts
source .venv-tts/bin/activate
pip install kokoro-onnx soundfile "fastapi" "uvicorn[standard]"

mkdir -p models/kokoro
curl -L -o models/kokoro/kokoro-v1.0.onnx \
  https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx
curl -L -o models/kokoro/voices-v1.0.bin \
  https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin
```

```bash
# every run
source .venv-tts/bin/activate
python3 -m uvicorn tts.server:app --factory --host 127.0.0.1 --port 8020
```

```bash
curl --fail http://127.0.0.1:8020/health
```

The SvelteKit dev server proxies `/tts/*` to `http://localhost:8020` (see
`frontend/vite.config.ts`). If this service is not running, the "Listen
(Kokoro)" buttons show a narration error but everything else — predictions,
TreeSHAP, the AI text note itself — is entirely unaffected; narration is
strictly optional and downstream of the AI note, never a dependency of it.
The model files (~340MB) are not committed; re-download them with the
commands above (see `.gitignore`).

### 8. Replay steps

1. On **Patient Replay**, use "Previous cutoff" / "Next cutoff" / the cutoff
   dropdown to step through the patient's legal replay timestamps.
2. Each step issues a new `POST /predict` call recomputed from history
   truncated at that cutoff — never a cached value.
3. Watch the recovery trajectory chart, the ICU/support trend charts, and
   the "Replay history" table accumulate as you advance.
4. Visit **AI + SHAP / Explainability** at any cutoff for TreeSHAP
   contributors per task, and click "Generate summary" for an AI-written
   research note (Groq-hosted `openai/gpt-oss-120b`, called server-side —
   see step 9 for configuration), then "Listen (Kokoro)" to hear it read
   aloud (step 7). The guided demo auto-generates and auto-narrates a note
   for each new patient it steps to.
5. Visit **Model Performance** for the frozen, one-time fresh-test evaluation
   (no inference is triggered by this page).
6. Open **Trajectory Copilot** from the header ("COPILOT" button, top right)
   or "Ask Copilot" on Patient Replay — it opens grounded to whichever
   patient/cutoff you were just viewing, auto-generates a default trajectory
   summary, and offers suggested prompts (e.g. "What changed since the
   previous cutoff?"). It calls `POST /assistant`, which reuses the exact
   same `/predict` payload as the dashboard — it never predicts
   independently, never sees the fresh-test cohort, and refuses treatment/
   diagnosis questions by design (see `configs/product_v2/
   trajectory_copilot_prompt_v1.txt`).
7. From the workspace home or Patient Replay, click **Enter my own record**
   (`/patients/custom`) to type in your own vitals/labs (heart rate, MAP,
   SpO2, GCS, lactate, etc., each at a chosen number of hours since a
   synthetic admission), optionally add **organ-support intervals**
   (vasopressor agent + rate, or invasive ventilation, with a start hour and
   either an end hour or "currently active"), and get a **real** forecast
   through the exact same frozen models, feature builder, SOFA computation,
   and support-state logic as the demo cohort — not a mock. The result shows
   a honest **DATA READINESS** summary (observations entered, SOFA
   components observed vs. missing, legal cutoffs, support state entered) —
   never a confidence score — and explicitly notes that urine output cannot
   be entered (SOFA's renal component needs a fully gapless 24-hour reading
   chain that episodic manual entry can't honestly provide). The record
   exists only in the API server's memory for that process's lifetime
   (never written to disk, never mixed with the demo manifest or the
   fresh-test cohort — its `CUSTOM-` stay-id namespace can't collide with
   either) and is bound to the signed-in user who created it (when Clerk is
   configured on both sides, per step 5) — another signed-in user gets a 404
   on it, identical to a nonexistent stay. Without `MONGODB_URI` configured
   (step 6), restarting the API server discards every custom record for
   every user; with it configured, records and their prediction history
   survive a restart and are visible on **My Health Record**
   (`/health-record`). See
   [`docs/product_v2/CUSTOM_RECORD_FLOW.md`](docs/product_v2/CUSTOM_RECORD_FLOW.md),
   `src/serving/v2/custom_record.py`, `src/serving/v2/persistence.py`, and
   `tests/test_v2_custom_record.py`.

### 9. AI recommendation configuration

The "AI + SHAP" page's research-note synthesis calls Groq's OpenAI-compatible
chat completions API server-side (the key never reaches the browser). Set:

```bash
cp .env.example .env
# edit .env: GROQ_API_KEY=<your key>, GROQ_MODEL=openai/gpt-oss-120b (or another chat-capable model this key can access)
```

`api/v2_app.py`'s `app()` factory loads `.env` automatically. If the key is
absent or the request fails for any reason, the endpoint returns a
structured `{"status": "UNAVAILABLE", ...}` response — the rest of the
dashboard (predictions, TreeSHAP, model performance) is entirely unaffected,
since the LLM is called strictly after prediction and never influences it.

### 10. Final demo flow (reference checklist)

The complete end-to-end path this product supports, in order:

1. Start the V2 API (step 3).
2. Start the frontend (step 4).
3. Configure Clerk locally, on both frontend and backend (step 5).
4. Open the app at `http://localhost:5173/`.
5. Sign in (or sign up) through the real Clerk widget.
6. Complete onboarding (role → research disclaimer → data-source mode).
7. Explore a demo patient (`DEMO-CARDIAC-00N`) on Patient Replay.
8. Replay several cutoffs with "Next cutoff" and watch the trajectory chart
   and status cards update from a freshly recomputed prediction each time.
9. Open TreeSHAP (AI + SHAP page) for the current cutoff's top contributors.
10. Open Trajectory Copilot and ask "What changed since the previous
    cutoff?"
11. Click **Enter My Own Record** to start a custom record.
12. Add vitals/labs (and, if desired, organ-support intervals — see the
    "Enter my own record" step in step 8's Replay steps above).
13. Submit and review the DATA READINESS summary.
14. The submission runs an actual frozen-V2 prediction immediately — no
    separate "run prediction" step exists; the result is already the real
    forecast.
15. Show **Model Performance** — the frozen final-evaluation metrics; this
    page triggers zero inference.
16. Show **Data Quality & Provenance** — model/hash metadata and the
    temporal-window observation heatmap, plus the documented limitations
    (synthetic benchmark, retrospective replay, no urine-output support for
    custom records, no FHIR/EHR connector yet).

If `MONGODB_URI` is configured (step 6), add: 17. Restart the API and reopen
the custom record from **My Health Record** (`/health-record`) or Patient
Replay — it rehydrates from MongoDB and serves identically, proving
persistence rather than just describing it.

See [`docs/product_v2/FINAL_DEMO_SCRIPT.md`](docs/product_v2/FINAL_DEMO_SCRIPT.md)
for timed 4-minute and 7-minute narrated versions of this same flow.

### 11. Automated browser tests (Playwright)

```bash
cd frontend
npm install        # first time only; installs @playwright/test
npx playwright install chromium   # first time only; downloads the browser
cp .env.test.example .env.test
# edit .env.test: PLAYWRIGHT_TEST_EMAIL/PASSWORD for a dedicated Clerk test
# user (do not reuse a real personal account), CLERK_SECRET_KEY for the
# same Clerk instance (used only to reset that test user's onboarding state
# and is never sent to the browser)
npm run test:e2e
```

This drives the real Clerk sign-in widget (no auth bypass anywhere in the
codebase — see [`docs/product_v2/AUTH_AND_DATA_BOUNDARIES.md`](docs/product_v2/AUTH_AND_DATA_BOUNDARIES.md)),
completes onboarding, replays a demo patient, submits a custom record,
opens TreeSHAP and Trajectory Copilot, and checks the Model Performance and
Data/Provenance pages — starting its own API and dev-server instances if
they are not already running (`playwright.config.ts`). Screenshots land in
`docs/evidence/product_v2/`.

If `MONGODB_URI` is configured (step 6), the custom-record and
`/health-record` specs create real, durable records/conditions under the
dedicated e2e test account on every run (there is no delete-record UI, by
design — see step 6). This is expected and harmless (it's the same
synthetic-only test account every time), but it does mean repeated runs
accumulate documents in that cluster over time; periodically clean them up
directly if that matters for your deployment, e.g.:
`python3 -c "from serving.v2.persistence import MongoPersistence; p = MongoPersistence(uri='...'); p._db.encounters.delete_many({'owner_user_id': '<e2e-test-user-id>'})"`.

### 12. Troubleshooting

- `503` from `/predict`: a Phase-3 model/calibrator/threshold hash mismatch
  was detected; the server refuses to serve rather than guess. Re-verify
  `artifacts/performance_v2/phase3/selected_models_v2.json` is unchanged.
- `404` on a stay you expected to work: only demo-manifest stays are
  servable, by design; check `configs/performance_v2/v2_demo_manifest_v1.json`.
  A fresh-V2-test subject ID (`SYN-V2-...`) will always 404 here.
- `422` on a cutoff: the timestamp is not one of that stay's legal cutoffs —
  use the dropdown rather than typing an arbitrary time.
- Dashboard shows "Prediction unavailable": the API is not running or
  `V2_API_BASE_URL` does not point at it.
- First request after startup is slow (~15-20s): the runtime loads and
  hash-verifies the DEV canonical timeline once at process start, not per
  request.
- SvelteKit dashboard shows a network/CORS error: confirm the V2 API
  (terminal 1) is running on port 8010 — the dev proxy and the API's CORS
  middleware both assume that port by default.
- `npm run dev` fails to start: this frontend targets Node 18+; run
  `node --version` and update if needed. `npm install` must complete before
  `npm run dev`/`npm run build`.
- "AI summary unavailable" on the AI + SHAP page: expected if `.env` has no
  `GROQ_API_KEY`, the key/model combination is invalid, or the network is
  unreachable — this is a deliberate graceful-degradation path, not a bug;
  check the API server's log for the specific reason.
- "Narration unavailable"/"Browser blocked autoplay audio": the Kokoro
  service (step 7) is not running, or the browser blocked an unprompted
  autoplay — click "Listen (Kokoro)" directly, which is a user gesture and
  is never blocked.
- Signed-in but stuck bouncing to `/onboarding`: onboarding state lives in
  Clerk's `unsafeMetadata` on the browser-side user object; clearing
  cookies/local storage or using a different browser resets it — this is
  expected for a demo-scale app with no separate onboarding database.
- Dashboard routes are reachable without signing in: this is by design when
  `PUBLIC_CLERK_PUBLISHABLE_KEY`/`CLERK_SECRET_KEY` are unset (step 5) —
  add both to `frontend/.env` and restart `npm run dev` to enable the
  server-side redirect.
- "Trajectory Copilot is unavailable right now": it shares the same
  `GROQ_API_KEY` configuration as step 9 — if the AI research note also
  says unavailable, the cause is the same.
- A custom record ("Enter my own record") 404s after a while: if
  `MONGODB_URI` is not configured (step 6), this is expected — the API
  server was restarted, which discards every in-memory-only custom record
  by design — create it again. If `MONGODB_URI` *is* configured, this
  should not happen (the record is transparently rehydrated from MongoDB on
  first access after a restart); check `GET /health`'s `persistence_mode`
  and the API server's log for a MongoDB connection error.
- Custom-record creation rejects a `glasgow_coma_scale` value: it must be a
  whole number (3-15) — SOFA's neurological component requires an integer.
- Custom-record creation rejects an observation time: it must be strictly
  greater than 0 (an hour-0 event is indistinguishable from pre-admission
  padding) and at most 90 (the legal-cutoff grid never extends further,
  regardless of how the record is otherwise configured).
- A custom record 404s for a user who didn't create it: expected — records
  are bound to the creating user's verified Clerk identity and a different
  user gets a 404, identical to a nonexistent stay (step 5).
- Custom-record creation rejects a vasopressor/ventilation interval: the
  agent must be one of `norepinephrine`/`epinephrine`/`dopamine`/
  `dobutamine`, the rate must be positive, and `end_hour` (if not "ongoing")
  must be after `start_hour`.
- Urine output cannot be entered on a custom record: this is a deliberate,
  documented limitation, not a bug — see
  [`docs/product_v2/CUSTOM_RECORD_FLOW.md`](docs/product_v2/CUSTOM_RECORD_FLOW.md).
- Playwright fails at the sign-in step: confirm `.env.test` has valid
  credentials for a real (dedicated test) Clerk user on the same Clerk
  instance the frontend/backend `.env` files point at.

### 13. Stop

`Ctrl-C` in each terminal.

---

## Benchmark-v1 operational runbook (historical)

> **Active scope amendment:** Project Scope v2 replaces MIMIC-IV as the final data source with an authorized synthetic adult cardiac/heart-disease dataset. That dataset has not yet been generated. Existing synthetic demo inputs remain engineering fixtures only. See `docs/governance/project_scope_v2.md`; its status is **DRAFT COMPLETE — TEAM FREEZE REQUIRED**.

All commands below run from the repository root. This document distinguishes the currently testable synthetic input workflow from blocked real serving and from the unapproved final-environment freeze.

## 1. Prerequisites and environment status

Observed development platform: CPython 3.9.6 on macOS arm64, CPU-only. This is not an approved final policy. No package manager has been selected and no final dependency lock exists. Captum and SHAP are absent from the observed runtime, although the real declared stack requires them.

The supported final install command is therefore **blocked pending Vedant/team approval of a package manager and exact environment versions**. Do not use `pip freeze` as a substitute for that decision. Inspect `observed_environment_phase15.json` only as a non-final diagnostic snapshot.

## 2. Open the repository

Open a shell at the repository root. Do not edit source paths or copy files from a developer machine. No credentials are required for synthetic fixture validation.

## 3. Environment verification

In the currently provisioned development environment:

```bash
python3 --version
PYTHONPATH=src:tests:. python3 -m reproducibility.reproduce audit --root .
```

The reproducibility audit is expected to remain blocked on the missing final environment lock and real scientific prerequisites.

## 4. Validate the official synthetic demo input

```bash
PYTHONPATH=src:tests:. python3 -m demo.fixture validate --root .
```

The command verifies the fixture hash, provenance, history contract, safe identifiers, feature names, current-SOFA contract, first +24-hour cutoff, six-hour cadence, three cutoff-specific histories, and a future row after t3.

## 5. Test suite and Phase-14 integration

```bash
PYTHONPATH=src:tests:. python3 -m pytest -q tests/test_integration.py tests/test_artifact_compatibility_e2e.py tests/test_serving_lineage_e2e.py
PYTHONPATH=src:tests:. python3 -m pytest -q
```

## 6. API startup and smoke

Start the fail-closed application locally:

```bash
PYTHONPATH=src:. python3 -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

In another shell:

```bash
curl --fail http://127.0.0.1:8000/health
curl -i http://127.0.0.1:8000/model-metadata
```

Health reports unavailable scope and metadata returns 503 until an approved bundle is composed. A t1/t2/t3 live prediction smoke is blocked; Phase-14 tests exercise the same API/pipeline contracts with explicitly synthetic artifacts without using precomputed predictions.

## 7. Dashboard startup and smoke

```bash
PYTHONPATH=src:. python3 -m uvicorn dashboard.app:app --host 127.0.0.1 --port 8501
```

Open `http://127.0.0.1:8501/`. The exact retrospective warning must render. The default dashboard intentionally shows unavailable state rather than fake predictions. Packaged t1/t2/t3 live replay remains blocked with the final environment and serving composition.

## 8. Replay cutoffs in the input fixture

The official synthetic fixture provides:

1. `2030-01-02T00:00:00+00:00`
2. `2030-01-02T06:00:00+00:00`
3. `2030-01-02T12:00:00+00:00`

Fixture and integration tests verify exact requests and that the timeline contains no later rows at each cutoff. Predictions are recomputed through `PredictionPipeline`; the input contains no prediction lookup table.

## 9. Artifact and governance audits

```bash
PYTHONPATH=src:tests:. python3 -m experiments.registry_cli audit --root .
PYTHONPATH=src:tests:. python3 -c "from pathlib import Path; from vedant_infra.g3 import audit_g3; r=audit_g3(Path('.'), scope='real'); print(r.overall, r.test_data_accessed)"
```

The second command is non-authorizing and does not write a G3 marker.

## 10. Stop services

Use `Ctrl-C` in each uvicorn shell. No background service or external API is required.

## 11. Common failures

- Import failure: confirm the documented `PYTHONPATH` and repository-root working directory. A proper installed-package path is pending the package-manager decision.
- Demo hash mismatch: do not update the hash silently. Review the semantic change, version the fixture when required, and rerun Phase-14/15 tests.
- API/dashboard unavailable: expected without an approved serving bundle; do not create fallback predictions.
- Missing Captum/SHAP: expected in the observed development snapshot and a blocker for a complete real-stack lock.
- Reproducibility audit blocked: expected until the environment lock and real frozen artifacts exist.

## 12. Historical MIMIC research mode (superseded by Project Scope v2)

The v1 baseline described a MIMIC-IV v2.2 research mode requiring external authorization and MIMIC-specific provenance. Project Scope v2 supersedes that final-data-source requirement. These details remain historical migration evidence only and must not be treated as prerequisites for the active synthetic study.

Do not copy raw clinical data, credentials, patient exports, or local database paths into this repository.

## 13. Change control

After an environment lock is approved, dependency changes require a versioned declaration, regenerated lock/hash, compatibility review, Phase-14 rerun, and Vedant approval. Demo semantic changes require a fixture-version change, new hash, and replay/integration rerun.
