# Personalized Patient Recovery Trajectory

> **Scientific lineage:** the current product uses the frozen Performance V3 release on an authorized synthetic adult cardiac cohort. The separate [Project Scope v2](docs/governance/project_scope_v2.md) rebuild remained a draft and does not supersede the executed V1/V2/V3 lineage. See [V3 governance](docs/performance_v3/GOVERNANCE_AND_LINEAGE.md).

Retrospective research software for forecasting independent +24/+48-hour SOFA changes, remaining time until the current ICU stay ends, and calibrated new organ-support initiation risk.

> RETROSPECTIVE SEQUENTIAL REPLAY — NOT REAL-TIME CLINICAL PREDICTION OR CLINICAL DECISION SUPPORT

## Current product: frozen V3 research platform

The [V3.1 product handoff](artifacts/performance_v3_1/product_handoff.json) retains **all four frozen V3 heads**. V3.1's privileged latent-state diagnostic and observed-data candidate are research evidence; neither is deployed. Start the current FastAPI composition root and existing SvelteKit frontend:

```bash
PYTHONPATH=src:. python3 -m uvicorn api.product_app:app --factory --host 127.0.0.1 --port 8010
cd frontend && npm install && npm run dev
```

The product loads the hash-verified [selected V3 bundle](artifacts/performance_v3/phase3/selected_models_v3.json) and fails startup if it cannot load; it never silently falls back to V2. Recovery +24h, remaining ICU time, and new organ-support initiation use XGBoost; recovery +48h uses CatBoost. Organ-support probabilities use the frozen Platt calibrator and threshold. [Defect D1](docs/performance_v3/DEFECT_D1.md) explains why historical V2 offline ICU/support metrics cannot represent deployed V2 performance.

The existing patient replay, Clerk authentication, MongoDB-backed health records, report confirmation, and Copilot remain part of the product. V3 adds [cohort analytics](frontend/src/routes/cohort/+page.svelte), a [model research workbench](frontend/src/routes/research/analytics/+page.svelte), [eight selected development-cohort scenarios](configs/performance_v3_product/showcase_manifest_v1.json), and [research export](frontend/src/routes/reports/+page.svelte). See the [V3 runbook](RUNBOOK.md#current-v3-product) and [product architecture](docs/product_v3/PRODUCT_ARCHITECTURE.md). Scientific evaluations and privileged diagnostics remain distinct from patient-level forecasts.

## Performance V2 — historical product release

Performance V2 is complete: four frozen XGBoost models (recovery +24h, recovery +48h, remaining ICU stay time, 24h new organ-support initiation risk), a one-time fresh-test evaluation, and a real, model-backed demo application. See [Performance V2 final report](docs/performance_v2/FINAL_V2_TEST_EVALUATION.md) for the frozen scientific numbers and [`V2_DEMO_SCRIPT.md`](docs/performance_v2/V2_DEMO_SCRIPT.md) for a guided walkthrough.

The commands in this section reproduce the historical V2 application only. Use the V3 entry point above for the current product; V2's offline ICU and support evaluation is affected by [defect D1](docs/performance_v3/DEFECT_D1.md).

```bash
PYTHONPATH=src:. python3 -m uvicorn api.v2_app:app --host 127.0.0.1 --port 8010 --factory
cd frontend && npm install && npm run dev   # dashboard at http://localhost:5173
```

This is a real, model-backed demo: every prediction is recomputed by `src/serving/v2` from raw history truncated at the selected cutoff, through the exact frozen Phase-3 V2 models — never a lookup table, and never the sealed fresh-V2-test cohort (structurally excluded; see [Test/fresh-cohort protection](#v2-fresh-test-protection) below). The dashboard (`frontend/`) is a SvelteKit app reusing its existing component library, restyled for this project, with an AI-generated (Groq) research-note synthesis on the Explainability page and voice narration via a local Kokoro TTS service. A dependency-free, server-rendered Python dashboard (`dashboard/v2_app.py`, port 8511) remains available as a Node-free fallback covering the same five views. See the [V2 section of RUNBOOK.md](RUNBOOK.md#v2-final-release-demo-application) for exact commands, health checks, AI-key configuration, and troubleshooting.

### Product layer: authentication, onboarding, Trajectory Copilot

On top of the frozen scientific system (branch `product-v2`, built from the same commit as the frozen `performance-v2` branch — see [`docs/product_v2/CURRENT_UI_INTEGRATION_AUDIT.md`](docs/product_v2/CURRENT_UI_INTEGRATION_AUDIT.md)), the dashboard now has:

- **Authentication** (Clerk, via the community `svelte-clerk` SvelteKit SDK) gating the dashboard routes both client-side (soft navigation) and server-side (hard reload), with a graceful "auth disabled" fallback when no keys are configured.
- **Onboarding** (role → research disclaimer → data-source mode) for first-time signed-in users.
- **Human-readable demo patient aliases** (`DEMO-CARDIAC-001`, …) instead of raw synthetic subject IDs, from a new additive product artifact (`artifacts/performance_v2/product/demo_patients_v1.json`) that never touches the frozen demo manifest's selection criteria.
- **Trajectory Copilot**, a grounded Q&A drawer over the existing prediction/SHAP output (`POST /assistant`) — never predicts independently, refuses treatment/diagnosis questions by design, and is fully optional to the core dashboard.
- **Enter My Own Record**, a manually-entered custom clinical record (vitals/labs plus vasopressor/ventilation organ-support intervals) served through the exact same frozen V2 pipeline as the demo cohort. Every record is bound server-side to its creating user's verified Clerk identity (`owner_user_id`, never client-supplied) — a second user gets a 404 on someone else's record, identical to a nonexistent stay. See [`docs/product_v2/CUSTOM_RECORD_FLOW.md`](docs/product_v2/CUSTOM_RECORD_FLOW.md) and [`docs/product_v2/AUTH_AND_DATA_BOUNDARIES.md`](docs/product_v2/AUTH_AND_DATA_BOUNDARIES.md).
- **My Health Record** (`/health-record`), a consolidated longitudinal record per signed-in user, backed by an optional MongoDB layer (`src/serving/v2/persistence.py`): a patient profile (editable demographics), conditions (add/edit/delete), every custom record's raw structured input linked to that profile, chronological vitals/labs and organ-support history across encounters, uploaded reports (metadata in MongoDB, binary content in a local object-storage abstraction — `src/serving/v2/report_storage.py` — never as a model input), prediction history across every replayed cutoff, a JSON export, and a lightweight audit trail of meaningful mutations. When `MONGODB_URI` is configured, everything survives an API restart (encounters are transparently rehydrated into the in-memory serving runtime from their raw input on first access); without it, behavior is unchanged from the original design (in-memory only, discarded on restart). MongoDB is never read by the scientific pipeline itself. See [`docs/product_v2/HEALTH_RECORD_ARCHITECTURE.md`](docs/product_v2/HEALTH_RECORD_ARCHITECTURE.md).
- **Report Intelligence** (`src/serving/v2/report_parser.py`): a PDF report can be analyzed (text extraction via `pypdf`, then Groq mapping free-text lab lines onto the frozen canonical concept vocabulary) to propose *candidate* measurements — never inserted automatically. A user reviews the candidates and explicitly confirms which ones, at which relative hour, become real observations on a named encounter, through the exact same validation path direct entry uses. A candidate the model isn't confident about is shown but can never be confirmed.

A full Playwright end-to-end suite (`frontend/tests/e2e/`) drives the real Clerk sign-in flow, onboarding, demo replay, custom-record entry, explainability, Copilot, and the performance/provenance pages against a real running API and frontend, and captures the screenshot evidence in `docs/evidence/product_v2/`.

A synthetic FHIR upload/export and a SMART-on-FHIR EHR sandbox connector remain explicitly **not implemented** — only a documented future-extension interface exists (`src/serving/v2/external_record_adapter.py`). "Connect EHR Sandbox" in the UI is labeled "SOON" because it does not work yet; manual entry above is the only working data-entry path. See the six documents in `docs/product_v2/` for the full architecture, auth boundaries, Copilot grounding, and demo scripts.

The sections below describe the earlier Benchmark-v1 baseline this project builds on; they remain historically accurate for that baseline and are unaffected by the V2 work above.

## Current status (Benchmark v1, historical)

The scientific and product contracts are implemented and structurally tested with engineering-only synthetic fixtures. Under Project Scope v2, the accepted final research dataset will also be synthetic, but it has not yet been specified or generated and must not be conflated with those fixtures. Model-backed serving remains blocked because an approved selected-model bundle is absent. The package/environment manager and final dependency versions are also unresolved, so **no final environment lock currently exists**. [The observed runtime snapshot](observed_environment_phase15.json) is explicitly non-final.

## Repository layout

- `src/`: data, modeling, evaluation, serving, explainability, governance, and reproducibility packages.
- `api/`: FastAPI facade.
- `dashboard/`: server-rendered retrospective replay UI.
- `data/demo/`: non-sensitive Phase-15 synthetic input fixture.
- `configs/`: versioned contracts and governance configuration.
- `tests/`: unit, adversarial, and integration tests.
- `docs/`: implementation reviews and reproducibility/governance documentation.

## Development verification

From the repository root, using the currently provisioned development environment:

```bash
PYTHONPATH=src:tests:. python3 -m demo.fixture validate --root .
PYTHONPATH=src:tests:. python3 -m pytest -q
PYTHONPATH=src:tests:. python3 -m reproducibility.reproduce audit --root .
```

These commands do not constitute installation from a final lock. Selecting and approving a package manager is required before a clean-environment installation command can be published.

## Local application entrypoints

```bash
PYTHONPATH=src:. python3 -m uvicorn api.main:app --host 127.0.0.1 --port 8000
PYTHONPATH=src:. python3 -m uvicorn dashboard.app:app --host 127.0.0.1 --port 8501
```

The default applications fail closed until an approved serving composition is supplied. They never substitute synthetic predictions silently. The fixture validator and Phase-14 integration tests prove the synthetic input path; a packaged live prediction demo remains blocked with the environment/serving bundle.

See [RUNBOOK.md](RUNBOOK.md), [reproducibility guidance](docs/REPRODUCIBILITY.md), and the [Phase-15 review](docs/pulkit/PHASE15_PACKAGING_REVIEW.md).

## Privacy boundary

The committed demo is a manually constructed engineering fixture and is not the authorized final synthetic research dataset. Project Scope v2 does not authorize or require MIMIC-IV access or MIMIC-derived experimental claims. Raw restricted clinical records, identifying extracts, credentials, and local database paths must remain outside this repository.

## V2 fresh-test protection

`src/serving/v2/guard.py` restricts every V2 dashboard/API prediction request to the small, versioned demo set in `configs/performance_v2/v2_demo_manifest_v1.json` (deterministic DEV/TRAIN subjects, selected without regard to model performance). Any other `stay_id` — including every subject in the sealed fresh-V2-test cohort (`artifacts/performance_v2/phase3/fresh_test_cohort/`, access state `FINAL_V2_RUN_COMPLETED`) — is rejected identically to an unknown stay (`404`). The Model Performance dashboard page reads only the already-frozen `artifacts/performance_v2/phase4/` evaluation artifacts and never triggers inference.
