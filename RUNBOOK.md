# Operational Runbook

## V2 final release demo application

This section is self-contained: it does not require reading the historical
Benchmark-v1 sections below, and needs no developer-specific local paths.

### 1. Prerequisites

```bash
python3 --version   # tested on CPython 3.9
python3 -c "import fastapi, xgboost, shap, matplotlib, httpx"   # must import cleanly
```

If any import fails, install the missing package with your environment's
package manager (no project-wide lock is currently pinned for these).

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

### 5. Replay steps

1. On **Patient Replay**, use "Previous cutoff" / "Next cutoff" / the cutoff
   dropdown to step through the patient's legal replay timestamps.
2. Each step issues a new `POST /predict` call recomputed from history
   truncated at that cutoff — never a cached value.
3. Watch the recovery trajectory chart, the ICU/support trend charts, and
   the "Replay history" table accumulate as you advance.
4. Visit **AI + SHAP / Explainability** at any cutoff for TreeSHAP
   contributors per task, and click "Generate summary" for an AI-written
   research note (Groq-hosted `openai/gpt-oss-120b`, called server-side —
   see step 6a for configuration).
5. Visit **Model Performance** for the frozen, one-time fresh-test evaluation
   (no inference is triggered by this page).

### 6. AI recommendation configuration

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

### 7. Troubleshooting

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

### 8. Stop

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
