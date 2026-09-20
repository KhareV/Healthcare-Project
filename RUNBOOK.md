# Operational Runbook

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
