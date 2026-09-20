# Authoritative Experiment Registry and Artifact Lineage

## Authority

`experiments/registry.csv` is the one authoritative run registry. It is not to
be copied into suffixed or “final” alternatives. `experiments/artifacts.csv` is
its complementary many-to-many artifact index, not a second run registry.

The Implementation Master Plan controls scientific semantics. The Work
Division assigns the registry to Vedant and review to Pulkit. Phase 16 does not
authorize test access or create the G3 marker.

## Run records

Every run has a unique `run_id`. A `candidate_id` identifies one scientific
hyperparameter configuration and may have several attempt run IDs. A retry
retains the failed attempt and references it through `retry_of_run_id` and
`parent_run_id`.

The inherited status vocabulary is `planned`, `running`, `completed`, `failed`,
and `aborted`. `attempt_status_detail` preserves the Phase 11 detailed status.
Run types are `scientific`, `synthetic`, `smoke`, `development`, `test`,
`sensitivity_smoke`, and `legacy_incomplete`. Only finalized, completed
`scientific` rows are eligible for final-scientific queries.

Required completed-run fields include timestamp, task, family, seed, commit,
configuration reference/hash, split hash, feature and label versions, model
reference/hash, metrics reference, and status. Scientific records reject an
unavailable commit. ISO-8601 timestamps must include a timezone.

Finalized rows are immutable. An identical re-registration is idempotent; a
changed finalized row fails. Pre-final lifecycle updates may change status and
runtime/output fields but not task, family, seed, configuration, split,
feature/label version, candidate, or run type.

## Artifact records

Each row in `experiments/artifacts.csv` records an immutable ID and path,
artifact class/version, exact-byte SHA-256, producing run, parent artifacts,
task/family and data-contract compatibility, commit, generating script/date,
run type, and status.

Semicolon-separated artifact IDs represent multiple parents. They are IDs, not
inferred directory relationships. The graph must be acyclic. Output artifacts
require a registered producing run. Duplicate IDs or immutable paths fail.

Class checks include:

- predictions bind to exactly one checkpoint and its exact hash;
- calibrators bind to one support checkpoint and prediction;
- thresholds bind to their exact calibrator hash;
- result tables/plots bind to registered predictions;
- task, family, split, feature, and label metadata agree with the producing run.

`legacy_incomplete` is an explicit historical classification. It is never
treated as complete final provenance.

## Integrity and stale dependencies

All file verification hashes current exact bytes using the existing
`vedant_infra.hashing.sha256_file`. A mismatch fails closed and never updates a
record. `artifact_descendants()` reports every downstream artifact that becomes
suspect if an input changes. Compatibility mismatches are stale/invalid lineage;
Phase 16 does not regenerate them.

Configuration objects use the existing order-independent
`experiments.search_governance.canonical_sha256`. File references use exact-byte
SHA-256.

## Commands

```bash
PYTHONPATH=src python3 -m experiments.registry_cli audit
PYTHONPATH=src python3 -m experiments.registry_cli trace-run RUN_ID
PYTHONPATH=src python3 -m experiments.registry_cli trace-artifact ARTIFACT_ID
```

These commands display provenance metadata only. They do not inspect model
binary internals or test data.

## Restricted-data policy

Registry and artifact metadata reject explicit password/token/secret fields,
raw clinical-note fields, patient names, and subject/stay identifier fields.
Notes cannot substitute for structured provenance. Raw MIMIC rows, credentials,
and identifying extracts must never be stored here.

## Historical migration

The seven Phase 4–8 synthetic smoke rows were preserved. Their run types,
finalization state, and metric hashes were recovered from exact existing files.
Sixteen checkpoint/metric/preprocessor artifacts were indexed. Checkpoints and
two reusable preprocessors are verified; the seven metric outputs are
`legacy_incomplete` because no historical
prediction artifacts were present. Nothing was guessed.

## Unlocked parameters

- `UNLOCKED ENGINEERING PARAMETER — RUN ID FORMAT`
- `UNLOCKED ENGINEERING PARAMETER — DIRTY WORKTREE RUN POLICY`
- `UNLOCKED ENGINEERING PARAMETER — final environment-lock representation`

The repository already used UTC ISO-8601 timestamps and a status vocabulary;
those conventions were retained. The Phase 16 run-type and artifact-index
representations must be reviewed before real final experiments.
