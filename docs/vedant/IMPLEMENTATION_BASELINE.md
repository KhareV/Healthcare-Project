# Vedant Phase 0 Implementation Baseline

**Status:** Phase 0 infrastructure/governance baseline  
**Scientific authority:** Final Implementation Master Plan v1.0  
**Ownership authority:** Final Work Division v1.0  
**Detailed context:** [`../CODEX_PROJECT_CONTEXT_V1.md`](../CODEX_PROJECT_CONTEXT_V1.md)

This record prevents implementation drift. It does not implement prediction timestamps, splits, features, labels, models, metrics, calibration, or test evaluation.

## Repository audit at Phase 0 start

The project directory existed but was not a Git repository. Before Phase 0 changes it contained only:

- `AGENTS.md`
- `docs/CODEX_PROJECT_CONTEXT_V1.md`

Audit findings:

| Area | Finding |
|---|---|
| Source roots | None existed. |
| Configuration roots | None existed. |
| Test roots/framework | None existed or was configured. |
| Artifact roots | None existed. |
| Model/evaluation code | None existed. |
| Package/project configuration | None existed. |
| Environment files/locks | None existed. |
| Git status/history/branch | Not available; directory was not a Git repository. |
| Duplicate/conflicting files | None. |
| Existing unrelated work | The initialized context documents; preserved unchanged. |

The observed interpreter was system Python 3.9.6 with pip 21.2.4. This is an observation, not a final environment lock or package-version decision.

## Ownership and interface map

| Area | Owner | Vedant's relationship |
|---|---|---|
| MIMIC extraction, first-stay cohort, timeline/provenance, SOFA, canonical features | Sanskruti | Contract consumer and mandatory reviewer for time/provenance/SOFA semantics. |
| Recovery and ICU-time labels | Sanskruti | Formula/eligibility reviewer and model consumer. |
| XGBoost implementation | Sanskruti | Joint comparison layer; Vedant owns information parity, tuning governance, evaluation, and selection. |
| Prediction timestamps and subject split | Vedant | Accountable implementation owner beginning in later phases. |
| Tensor/loader/training framework, GRU/LSTM | Vedant | Accountable implementation owner. |
| Metrics, grouped bootstrap, selection, calibration, registry/test governance | Vedant | Accountable implementation owner. |
| Organ-support states and labels | Pulkit | Vedant reviews censoring/eligibility and consumes the frozen contract. |
| PredictionPipeline, explanations, API/dashboard, packaging | Pulkit | Vedant supplies manifest/model artifacts and reviews model metadata/equivalence. |
| Scientific contracts/configs | Domain owner plus mandatory reviewer(s) | Shared and versioned; no local compatibility patches. |
| Tests/evidence | Distributed by failure domain | Vedant owns split/model/statistics/governance evidence. |

Phase 0 paths introduced here:

- `src/vedant_infra/`: Vedant-owned infrastructure helpers only.
- `docs/vedant/`: Vedant baseline and blocker records.
- `configs/vedant/`: convention documentation; no fabricated scientific config.
- `experiments/registry.csv`: empty shared registry skeleton maintained by Vedant.
- `artifacts/governance/`: governance convention only; no freeze marker.
- `tests/test_phase0_infrastructure.py`: Vedant Phase 0 governance tests.
- `tests/fixtures/`: documentation for future non-sensitive synthetic fixtures.

## Frozen scientific invariants — recorded, not implemented

### Cohort and clock

- Prediction unit: adult subject's chronologically first ICU stay; no global exclusion of hospital deaths.
- `t_k = intime + 24h + 6h*k`, `k=0,...,11`.
- Retain only `t_k <= outtime - 6h`; at most 12 canonical cutoffs per stay.

### Canonical history

- History is `(t-48h,t]`, represented by eight consecutive disjoint six-hour bins: `(t-48,t-42], ..., (t-6,t]`.
- An observation at or before `t-48h` cannot enter the tensor or seed forward fill.
- The pre-ICU region is padding and must never be populated with hospital-floor history.
- Padding, within-ICU clinical missingness, genuine-observation masks, and TSLO remain distinct.

### Targets and eligibility

- `DeltaSOFA24 = SOFA(t+24h) - SOFA(t)`.
- `DeltaSOFA48 = SOFA(t+48h) - SOFA(t)`; it is independent and never chained through 24h.
- ICU-time target is `log1p((outtime-t) hours)`, meaning time until the current ICU stay terminates.
- Organ support is an eligible OFF-to-ON qualifying vasopressor or invasive-ventilation transition in `(t,t+24h]`.
- Incomplete follow-up without an observed positive event is censored/ineligible, never relabeled negative.
- A shared canonical grid has task-specific eligibility masks.

### Split and test isolation

- Train: anchor-year groups 2008–2010 plus 2011–2013.
- Validation: 2014–2016.
- Test: 2017–2019.
- Assignment is by `subject_id`; no subject may cross splits. This is a coarse temporal holdout.
- Test data cannot affect features, preprocessing, target statistics, class weights, hyperparameters, architecture, model-family selection, calibration, or thresholding.

### Candidates, tuning, and selection

- Candidates: naive, XGBoost, and GRU. LSTM is one fixed sensitivity run per task. MTL/TFT are optional and outside the core path.
- XGBoost and GRU each receive exactly 30 validation configurations per task, with search spaces frozen before search. Early stopping remains inside one configuration.
- Serving selection considers XGBoost versus GRU only and occurs independently per task.
- Recovery selects lowest validation 24h stay-balanced MAE.
- ICU time selects lowest validation stay-balanced weighted median absolute error in hours.
- Organ support selects highest validation stay-balanced AUPRC.
- Model superiority, baseline improvement, ablation improvement, and calibration improvement are empirical findings—not correctness gates.

### Statistics, calibration, and serving

- Eligible snapshot `j` in stay `i` receives weight `1/n_i` for its task.
- Grouped bootstrap samples `stay_id` clusters with replacement, carries every eligible row in each stay copy, recomputes the same stay-balanced estimator, and reports percentile 95% intervals.
- These intervals describe aggregate metric uncertainty, never individual-patient uncertainty.
- Only the selected organ-support classifier is calibrated, using isotonic regression on validation predictions.
- Its threshold is chosen by the predefined validation F1 criterion; calibration and threshold freeze before test.
- `selected_models_v1.json` is the serving manifest. Selected XGBoost routes to TreeSHAP; selected GRU routes to Integrated Gradients.

## Phase order and isolation

1. **Phase 0:** repository/environment/governance baseline only.
2. **Phase 1 and later:** implement only when specifically requested, respecting G0–G5 and upstream contracts.
3. G0 freezes science/contracts and proves the synthetic path.
4. G1 freezes real data/labels and structural QA.
5. G2 establishes candidate information parity and first real replay.
6. G3 freezes validation selection, LSTM sensitivity, support calibration/threshold, manifest, and explanation adapters before test.
7. G4 performs the one-time final test/evaluation freeze.
8. G5 completes independent reproduction, evidence, report, slides, runbook, and viva readiness.

Later-phase code must not be pulled forward merely because an interface is anticipated. Contract placeholders state provenance and expected ownership but contain no fabricated feature ordering, MIMIC fields, split rows, model results, or selected artifacts.

## Configuration convention

No repository convention existed. The Phase 0 engineering default is:

- human-authored scientific and training configuration should use versioned YAML where required by the authoritative documents;
- JSON should be used for machine manifests/contracts explicitly named JSON;
- CSV should be used for tabular registries/split mappings explicitly named CSV;
- paths stored inside portable metadata are repository-relative POSIX-style paths;
- configuration files are hashed as their exact stored bytes—no undocumented normalization;
- scientific contract names and serialization formats must be agreed at G0 before dependent implementation.

This is an **UNLOCKED PARAMETER / ENGINEERING DEFAULT**, not a scientific freeze. No YAML parser or configuration loader is added because there is no actual config or package-management convention yet.

## Experiment registry convention

`experiments/registry.csv` is header-only at Phase 0. It contains no fake runs. Its columns support:

- identity/time: `run_id`, `timestamp_utc`, `status`, `parent_run_id`;
- experiment: `task`, `model_family`, `seed`;
- lineage: `code_commit`, `config_ref`, `config_hash`, `search_space_hash`, `split_hash`, `feature_version`, `label_version`;
- outputs: `model_artifact_ref`, `model_sha256`, `metrics_ref`;
- notes: `notes`.

The validator accepts a header-only registry. A record marked `completed` must provide the mandatory lineage/output metadata; conditionally applicable fields such as `search_space_hash` and `parent_run_id` may remain empty. Portable reference fields must be repository-relative.

## Artifact naming, versioning, and SHA-256 policy

- Use semantic-purpose plus explicit version names, such as `<purpose>_v1.<ext>`; never use ambiguous `final_final` naming.
- Artifacts are immutable after registration. Prediction-affecting changes increment the relevant version and regenerate dependents.
- Hash configs, split artifacts, models, manifests, preprocessors, calibration artifacts, and threshold artifacts with SHA-256 over exact file bytes.
- Store lowercase 64-character hexadecimal digests.
- Registry/manifests store repository-relative artifact references plus their digest.
- Directory artifacts require an explicitly versioned manifest of member paths/hashes; do not rely on filesystem traversal order.
- Hash verification must fail closed on absence, malformed digest, changed bytes, or mismatched version linkage.

## Test-set governance convention

The future marker location is `artifacts/governance/g3_freeze.json`. Phase 0 deliberately does **not** create it. The proposed marker schema is an unlocked engineering convention documented in `artifacts/governance/README.md`.

Later final-test commands must refuse to execute unless the marker exists, is structurally complete, has gate/status `G3`/`frozen`, identifies the selected manifest and its SHA-256, confirms validation searches/LSTM sensitivity/support calibration/threshold are frozen, and records all-team review. Marker validity does not replace independent artifact-hash checks.

## Test entry point

No test framework existed, so Phase 0 uses Python's standard-library `unittest` without adding a package dependency:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_phase0_*.py' -v
```

The Phase 0 suite requires no MIMIC access. It covers registry structure/completed-record validation, path portability, stable/content-sensitive SHA-256 behavior, absent/incomplete G3 rejection, package import, and absence of fabricated later-phase artifacts.

## Environment and reproducibility status

- There is no package manifest, dependency lock, environment file, container definition, or CI workflow.
- The observed system Python/pip versions are development-machine facts only and are not asserted as supported or final.
- Exact Python/package/CUDA/driver versions must be frozen from the known-good environment before final training.
- The numeric project primary seed was not found and is recorded as unlocked rather than fabricated.
- A Git commit cannot yet be recorded because this directory is not a Git repository. Registered completed experiments will require a real `code_commit` once project Git conventions exist.

## Phase 0 non-actions

Phase 0 creates no `timestamp_spec_v1.yaml`, timestamp algorithm, `split_v1.csv`, feature schema/order/dimension, label implementation, model/training code, selected-model manifest, model checkpoint, calibrator, threshold, metric result, final package lock, numeric seed, G3 marker, MIMIC mapping, restricted data, or test-set access.
