# Personalized Patient Recovery Trajectory

> **Active scope amendment:** the final academic dataset is synthetic and the population domain is adult cardiac/heart-disease. The repeated-cutoff three-task architecture remains unchanged. See [Project Scope v2](docs/governance/project_scope_v2.md). The amendment is a complete draft pending team freeze.

Retrospective research software for forecasting independent +24/+48-hour SOFA changes, remaining time until the current ICU stay ends, and calibrated new organ-support initiation risk.

> RETROSPECTIVE SEQUENTIAL REPLAY — NOT REAL-TIME CLINICAL PREDICTION OR CLINICAL DECISION SUPPORT

## Performance V2 — final release (start here)

Performance V2 is complete: four frozen XGBoost models (recovery +24h, recovery +48h, remaining ICU stay time, 24h new organ-support initiation risk), a one-time fresh-test evaluation, and a real, model-backed demo application. See [Performance V2 final report](docs/performance_v2/FINAL_V2_TEST_EVALUATION.md) for the frozen scientific numbers and [`V2_DEMO_SCRIPT.md`](docs/performance_v2/V2_DEMO_SCRIPT.md) for a guided walkthrough.

```bash
PYTHONPATH=src:. python3 -m uvicorn api.v2_app:app --host 127.0.0.1 --port 8010 --factory
cd frontend && npm install && npm run dev   # dashboard at http://localhost:5173
```

This is a real, model-backed demo: every prediction is recomputed by `src/serving/v2` from raw history truncated at the selected cutoff, through the exact frozen Phase-3 V2 models — never a lookup table, and never the sealed fresh-V2-test cohort (structurally excluded; see [Test/fresh-cohort protection](#v2-fresh-test-protection) below). The dashboard (`frontend/`) is a SvelteKit app reusing its existing component library, restyled for this project, with an AI-generated (Groq) research-note synthesis on the Explainability page. A dependency-free, server-rendered Python dashboard (`dashboard/v2_app.py`, port 8511) remains available as a Node-free fallback covering the same five views. See the [V2 section of RUNBOOK.md](RUNBOOK.md#v2-final-release-demo-application) for exact commands, health checks, AI-key configuration, and troubleshooting.

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
