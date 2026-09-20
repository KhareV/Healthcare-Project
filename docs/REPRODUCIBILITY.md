# Vedant ML/Evaluation Reproducibility Runbook

## Scope

This runbook covers Vedant's configuration, loaders, model/evaluation contracts,
metrics, grouped bootstrap, selection/calibration lineage, and registry audits.
It does not implement Sanskruti's source-data extraction or Pulkit's product
pipeline, API, dashboard, explainability, or packaging layer.

The current repository is framework/synthetic only. Real frozen-artifact and
cross-member claims remain blocked because Phase 19/20/G4 did not execute.

## Environment

An exact environment lock is mandatory for a final claim. None currently
exists. Do not infer a lock from the current machine or use wildcard versions.
Once Pulkit supplies the actual scientific environment lock, record its
repository-relative path and SHA-256 in `configs/reproducibility_v1.json`.

## Commands

From the repository root:

```bash
PYTHONPATH=src:tests python3 -m pytest -q
PYTHONPATH=src python3 -m reproducibility.reproduce audit --root .
PYTHONPATH=src python3 -m reproducibility.reproduce synthetic --root .
PYTHONPATH=src python3 -m reproducibility.reproduce frozen-artifacts --root .
PYTHONPATH=src python3 -m experiments.registry_cli audit --root .
```

`audit` is expected to exit 2 until every real prerequisite is present.
`synthetic` must pass without MIMIC access. `frozen-artifacts` must fail closed
until a valid G4 artifact exists.

## Reproduction modes

- Synthetic/CI: contract, metric, bootstrap, hash, path, and policy checks only.
- Frozen artifacts: inference and metrics from registered frozen artifacts; no
  fitting, selection, or test reopening.
- Source data: owned by Sanskruti and requires permitted MIMIC-IV access.
- Cross-member train/eval: one predesignated frozen validation configuration,
  executed by a non-owner without search or configuration changes.

## Cross-member procedure

Before the attempt, freeze the exact run ID, config hash, split hash, feature and
label versions, preprocessor hash, seed policy, code commit, environment-lock
hash, expected metrics, and predeclared training tolerance. The non-owner then
runs only:

```bash
PYTHONPATH=src python3 -m reproducibility.reproduce train-eval --root . --run-id <FROZEN_RUN_ID>
```

The command remains blocked until a real run has been designated. Never switch
to a different configuration after a failed attempt.

## Failure handling

Classify failures as environment, artifact, configuration, implementation,
scientific, or documentation failures. Never loosen tolerance after observing a
difference. Never refit calibration, reselect a threshold, tune a model, or
reopen final-test decisions. Result-affecting defects require formal
invalidation and version reset.

## Restricted data and secrets

Raw MIMIC data, identifying extracts, credentials, and tokens remain external.
Only synthetic fixtures and registered permitted artifacts may be committed.
Secrets are documented as external prerequisites, never copied into manifests.
