# Phase 0 Blockers and Unlocked Parameters

Items here are not silently guessed. “Blocked” means the dependent later-phase work must wait; it does not prevent completion of the infrastructure-only Phase 0.

## Blocked — information required

- **BLOCKED — INFORMATION REQUIRED:** Exact MIMIC-IV v2.2 tables, fields, units, event-time columns, and available derived concepts. Owner: Sanskruti/data verification. Needed before real extraction/feature/label work.
- **BLOCKED — INFORMATION REQUIRED:** Pinned MIMIC-Code commits and verified feasibility for SOFA and invasive-ventilation concepts. Owners: Sanskruti/Pulkit with Vedant review. Needed before G0/G1 scientific freeze.
- **BLOCKED — INFORMATION REQUIRED:** Sanskruti's final feature order, aggregation behavior, mask/TSLO encoding, feature schema serialization, and tensor-compatible static-feature contract. Needed before the real tensor contract/loader.
- **BLOCKED — INFORMATION REQUIRED:** Pulkit's finalized `event_dict_v1.yaml`, qualifying support mappings, OFF/ON evidence fields, censor reason vocabulary, and organ-support label handoff schema. Needed before the real support classifier path.
- **BLOCKED — INFORMATION REQUIRED:** Actual first-stay cohort artifact/interface and deterministic tie/age derivations. Needed before real timestamp/split generation.
- **BLOCKED — INFORMATION REQUIRED:** Team decision on the repository-contract naming discrepancies already recorded in `docs/CODEX_PROJECT_CONTEXT_V1.md` (`feature_schema` YAML/JSON, timestamp module name, serving pipeline module name). Needed at G0 before dependent imports harden.

No conflicting scientific definition or infeasibility was found during Phase 0.

## Unlocked parameters / engineering defaults

- **UNLOCKED PARAMETER — PROJECT PRIMARY SEED:** No numeric project seed exists in the repository or authoritative documents. Do not invent it in code or completed run metadata.
- **UNLOCKED PARAMETER — PACKAGE/ENVIRONMENT MANAGER:** No `pyproject.toml`, requirements file, Conda file, lock, container, or existing convention exists.
- **UNLOCKED PARAMETER — FINAL ENVIRONMENT VERSIONS:** The known-good exact Python/package and optional CUDA/driver versions are not frozen. The locally observed interpreter is not the final lock.
- **UNLOCKED PARAMETER — CONFIG IMPLEMENTATION:** Versioned YAML for human configs and JSON for explicit machine contracts is the Phase 0 proposal; team review is required before adding a loader/dependency.
- **UNLOCKED PARAMETER — G3 MARKER SCHEMA:** The minimal proposed fields in `artifacts/governance/README.md` are an engineering convention requiring team review before G3.
- **UNLOCKED PARAMETER — REGISTRY STATUS VOCABULARY:** Phase 0 recognizes `planned`, `running`, `completed`, `failed`, and `aborted`; revise only through a compatible schema change.
- **UNLOCKED PARAMETER — GROUPED BOOTSTRAP REPLICATE COUNT:** The documents freeze cluster semantics and percentile bounds, not numeric `B`.
- **UNLOCKED PARAMETER — INTEGRATED GRADIENTS BASELINE:** Pulkit must freeze and document it for the selected GRU explanation path.
- **UNLOCKED PARAMETER — STOCHASTIC REPRODUCTION TOLERANCE:** Must be declared with the eventual deterministic training/environment policy.
- **UNLOCKED PARAMETER — CI PROVIDER/COMMAND:** No Git repository or CI system exists. The documented local `unittest` command is the current entry point.

## Phase 1 readiness dependencies

Synthetic contract work for timestamps can begin after Sanskruti reviews the baseline clock/boundary contract location and the team confirms package/config naming. Real-data timestamp generation additionally requires the cohort interface. Phase 1 must not require the feature order or support event schema unless its requested scope expands beyond timestamps.

## Phase 4 disposition

The synthetic Dataset/DataLoader and generic training infrastructure are now
implemented. Real integration remains blocked on the final feature schema,
TSLO sentinel, static feature contract, train-derived preprocessing artifacts,
real canonical feature/label artifacts, support event/label version, and
production serialization review. The Phase-4 test seed, batch size, epoch
count, optimizer values, CPU device, `num_workers=0`, and logging cadence are
explicit engineering test defaults and do not freeze later research settings.

## Phase 5 disposition

The synthetic recovery GRU, independent-horizon masked Huber loss, and
training-only target scaler are implemented. Real recovery training remains
blocked on the final feature/preprocessing contract, TSLO encoding, static
input contract, real canonical recovery labels, and Sanskruti's scientific
review. Equal weighting of available horizon means, population target standard
deviation, unit scaling for zero variance, and final-representation dropout
placement are explicit unlocked engineering parameters—not frozen scientific
choices. The Phase-5 smoke run is excluded from the later 30-configuration
validation search.

## Phase 6 disposition

The synthetic current-ICU remaining-time GRU, eligible-only log-space Huber
loss, and versioned `expm1(max(0, y_hat))` postprocessor are implemented. Real
training remains blocked on final canonical features/preprocessing, real
ICU-time labels and spot checks, TSLO/static contracts, and reviewer approval.
Smoke Huber delta, run length, seed, and other development settings are
engineering defaults. No additional ICU-time target normalization was added,
and the Phase-6 run is excluded from the later 30-configuration search.

## Phase 7 disposition

The synthetic uncalibrated organ-support GRU, eligible-only weighted
BCE-with-logits loss, and train-only class-weight artifact are implemented.
**BLOCKED — PULKIT FROZEN ORGAN-SUPPORT LABEL ARTIFACT REQUIRED FOR REAL
TRAINING:** no real label/event-dictionary handoff exists in the repository;
the current fixture is explicitly manual and synthetic. The proposed
`N_negative/N_positive` formula and weighted-BCE smoke monitor are unlocked
engineering parameters requiring review. No calibration, threshold, search,
or endpoint-state derivation was added.

## Phase 8 disposition

The reusable fixed-cell LSTM sensitivity implementation and three synthetic
task smoke paths are complete. Each smoke comparison inherits its Phase 5–7
GRU development config, seed, canonical fixture, split, model dimensions,
optimizer, early stopping, task adapter, and frozen task artifact; only the
recurrent cell changes from GRU to LSTM. These runs are explicitly synthetic,
non-scientific, excluded from search, and excluded from serving selection.

**BLOCKED — FINAL SELECTED GRU CONFIGURATIONS REQUIRED FOR SCIENTIFIC LSTM
RUNS:** the repository contains no completed 30-configuration validation
search, frozen selected GRU run per task, G3 marker, or selected-model manifest.
The three final one-run-per-task LSTM sensitivity executions must not occur
until those parent configurations and the project seed treatment are frozen.
No smoke/development GRU was promoted to selected status.

The descriptive parameter-count difference is expected and is not an unlocked
tuning parameter: recovery GRU/LSTM counts are 13,570/18,050; ICU-time and
support counts are 13,505/17,985. Hidden size remains 64; no parameter-count
matching was performed. The existing Phase 5–7 engineering/scientific blockers
for real features, TSLO/static inputs, labels, and support weighting remain in
force.

## Phase 9 disposition

The model-neutral stay-balanced point-estimate layer is implemented for
recovery, remaining ICU time, and organ support. One shared weight utility
assigns every eligible snapshot in stay `i` weight `1/n_i` using the
task/horizon-specific eligibility vector. Regression and classification
evaluators share this implementation; no model-family branches, training
class weights, model selection, calibration, threshold fitting, test access,
or bootstrap logic were added.

**BLOCKED — DIRECTIONAL AGREEMENT ZERO/TIE SEMANTICS REQUIRED:** neither
authoritative document defines the direction assigned when the true or
predicted SOFA delta is exactly zero. Directional agreement is calculated for
strictly nonzero pairs; if any eligible zero occurs the directional result is
`None` with this blocker while MAE/RMSE remain available.

**BLOCKED — AUPRC INTEGRATION CONVENTION REVIEW REQUIRED BEFORE REAL MODEL
SELECTION:** the documents say stay-balanced AUPRC but do not distinguish
step-integrated weighted average precision from trapezoidal PR area. The
implemented, versioned synthetic path uses weighted step-integrated average
precision with score ties processed as one threshold group. Sanskruti must
approve this convention before it can govern real model selection.

**UNLOCKED ENGINEERING DETAIL — WEIGHTED MEDIAN TIE CONVENTION:** weighted
quantiles use the lower/inverted empirical CDF: after ascending sort, return
the first observed value whose cumulative weight is greater than or equal to
`q * total_weight`. Thus an exact 0.5 boundary returns the lower value. This
must be frozen before real selection.

**UNLOCKED REPORTING PARAMETER — ICU ERROR PERCENTILE LEVELS:** the generic
weighted-percentile primitive is implemented, but no final percentile levels
were chosen because the source documents specify none.

**UNLOCKED ENGINEERING DETAIL — THRESHOLD EQUALITY:** supplied-threshold
metrics classify `probability >= threshold` as positive. Phase 9 never selects
or stores a project operating threshold.

## Phase 10 disposition

The grouped stay-cluster bootstrap framework is implemented against the
authoritative Phase-9 callbacks. Eligible ICU stays are sampled with
replacement; every sampled occurrence receives a replicate-local cluster ID,
carries all of its eligible task rows, and therefore receives total evaluation
weight one even when the same original stay is drawn repeatedly. The original
point estimate is the Phase-9 estimate on the original data, never a bootstrap
average. Invalid metric replicates are skipped per metric and counted.

**UNLOCKED PARAMETER — NUMBER OF BOOTSTRAP REPLICATES B:** neither
authoritative document specifies a final number. The versioned config leaves
`n_bootstrap=null`, and the API requires an explicit positive value. Small
values in tests are labelled test values only.

**UNLOCKED PARAMETER — BOOTSTRAP RNG SEED:** no scientific bootstrap seed is
frozen. The API requires an explicit integer; tests use a named test seed.

**UNLOCKED REPORTING / QA PARAMETER — MINIMUM VALID BOOTSTRAP REPLICATES:**
the documents require valid/invalid counts but specify no acceptance minimum.
Zero valid replicates return no CI bounds rather than fabricated numbers.

**UNLOCKED ENGINEERING DETAIL — BOOTSTRAP PERCENTILE INTERPOLATION:** the
2.5th and 97.5th percentiles use deterministic Type-7 linear interpolation
(`index=q*(n-1)`). This convention requires freeze before final test CIs.

Final scientific CI generation remains blocked on G3 authorization, frozen
`B`, frozen bootstrap seed, convention review, and final test predictions. No
test data was accessed and no final CI artifact was generated.

## Phase 11 disposition

The validation-only search-governance framework is implemented. The exact
Master Plan GRU and XGBoost ranges are serialized and canonically hashed. Six
synthetic dry-run manifests deterministically materialize exactly 30 unique
candidate configurations each, consume zero scientific candidate slots, and
hard-reject candidate 31. Candidate identity is distinct from execution
attempt identity; retries preserve the candidate and are traceable in the
extended existing registry schema. Train/validation access is allowlisted and
test access hard-fails.

**UNLOCKED ENGINEERING PARAMETER — SEARCH CONFIGURATION SAMPLER:** the source
documents do not freeze the scientific method that selects 30 configurations
from continuous ranges. The implemented deterministic sampler is explicitly
synthetic/test-only and cannot generate a scientific manifest.

**UNLOCKED PARAMETER — SCIENTIFIC SEARCH SEED POLICY:** no approved search
seed exists. Dry-run seeds are test values only.

**BLOCKED — EARLY-STOPPING MONITOR MUST BE FROZEN BEFORE SCIENTIFIC SEARCH:**
the Phase 5–7 smoke monitors are development settings, not approved scientific
search monitors.

**UNLOCKED ENGINEERING / MODEL PARAMETER — XGBOOST EARLY-STOPPING ROUNDS:**
the source documents require validation early stopping but do not provide the
round count.

**UNLOCKED PARAMETER — WITHIN-FAMILY VALIDATION TIE-BREAK POLICY:** exact ties
hard-fail ranking until a deterministic policy is approved.

**BLOCKED — REAL GRU SEARCH PREREQUISITES INCOMPLETE:** no real canonical
dataset, split_v1 artifact, final schemas/preprocessors/labels, approved
sampler/seed, or frozen early-stop monitors exist.

**BLOCKED — SANSKRUTI XGBOOST SEARCH ENTRYPOINT REQUIRED:** no owned XGBoost
model, canonical flattening entrypoint, artifact contract, or approved early
stopping implementation exists. No replacement XGBoost stack was created.

No scientific search, within-family real winner, LSTM scientific handoff,
cross-family selection, calibration, thresholding, or test access occurred.
