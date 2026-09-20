# Project Scope v2 Contract Impact Matrix

Status: **DRAFT COMPLETE — TEAM REVIEW REQUIRED**. This is an impact analysis only. It changes no runtime contract and authorizes no Phase-2+ implementation.

Action vocabulary: `UNCHANGED`, `WORDING_ONLY`, `NEW_ADAPTER`, `CONFIG_SUCCESSOR`, `VERSIONED_SCIENTIFIC_CHANGE`, `REGENERATE_ARTIFACT`, `RETIRE`.

## Vedant-owned systems

| Component | Current contract/module | v2 impact | Action | Code reuse | Artifact consequence | Future review |
|---|---|---|---|---|---|---|
| Prediction timestamps | `configs/timestamp_spec_v1.yaml`; `src/data/timestamps.py` | Clock and boundary semantics retained | UNCHANGED | Full | Regenerate rows from final episodes | Confirm exact equality |
| Subject split | `configs/split_spec_v1.yaml`; `src/data/split.py` | Era mapping superseded; isolation/integrity retained | CONFIG_SUCCESSOR | Utilities high; policy mapping replaced | New split artifact/hash; all fitted artifacts stale | Freeze policy before Phase 10 |
| Tensor schema | `configs/tensor_contract_v1.json`; `src/data/schema.py` | Information shape retained; `anchor_year_group` is MIMIC-bound | NEW_ADAPTER or VERSIONED_SCIENTIFIC_CHANGE | Validation/parity high | Canonical fixtures/data regenerate | Decide adapter versus v2 field |
| Dataset/loaders | `src/data/dataset.py`; `collate.py` | Data-source agnostic after canonical input | UNCHANGED | Full | Recreate datasets | Verify v2 canonical examples |
| Training framework | `src/training/` | Data-source agnostic | UNCHANGED | Full | New runs/checkpoints/logs | None beyond normal review |
| Recovery GRU | `src/models/gru_recovery.py`; task adapter | Output semantics retained | REGENERATE_ARTIFACT | Full | Retrain checkpoint/scaler | Verify synthetic SOFA labels |
| Remaining-time GRU | `src/models/gru_icu_time.py`; postprocess | Episode semantics retained | REGENERATE_ARTIFACT | Full | Retrain checkpoint/results | Verify explicit outtime |
| Support GRU | `src/models/gru_support.py` | Endpoint retained; input labels synthetic | NEW_ADAPTER | Full model code | Retrain checkpoint/class weight | Review synthetic support labels |
| LSTM sensitivity | `src/models/lstm.py`; comparison configs | Role retained | REGENERATE_ARTIFACT | Full | Rerun once/task after GRU selection | Confirm inherited configs |
| Metrics | `src/evaluation/metrics.py`; weights | Stay-balanced estimands retained | UNCHANGED | Full | Regenerate all results | Confirm stay identity |
| Bootstrap | `src/evaluation/bootstrap.py`; config | Stay clusters retained | UNCHANGED | Full | Regenerate confidence intervals | Freeze remaining parameters |
| Search | `src/experiments/search_governance.py` | Exact-30 validation governance retained | UNCHANGED | Full | New candidate manifests/runs | Freeze scientific seed/sampler |
| Selection | `src/evaluation/select.py` | Per-task criteria retained | UNCHANGED | Full | Regenerate comparisons/winners | Review new lineage |
| Calibration/threshold | `src/evaluation/calibrate.py`; `threshold.py` | Support-only validation workflow retained | UNCHANGED | Full | Refit artifacts | Freeze open implementation policies |
| Registry/lineage | `src/vedant_infra/registry.py`; `src/experiments/lineage.py` | MIMIC provenance fields need generator equivalents | NEW_ADAPTER | Core graph/hash logic full | New records; old smoke history remains nonfinal | Approve schema extension |
| G3/final-test guard | `src/vedant_infra/g3.py`; `src/evaluation/final_test.py` | Protection retained; scope vocabulary needs authorized-final-synthetic | CONFIG_SUCCESSOR | Guard logic high | No current authorization is promoted | Adversarial scope tests |
| Data acceptance | `configs/data_acceptance_v1.json`; `src/data/real_adapter.py` | MIMIC checks replaced by generator/schema checks | VERSIONED_SCIENTIFIC_CHANGE | Fail-closed patterns reusable | New acceptance report | Design in later phase |
| Reproducibility | `src/reproducibility/` | Upstream reproduction starts at generator | NEW_ADAPTER | Framework high | Regenerate manifest/report | Add generator command/hash |
| Evidence | `src/evidence/`; evidence configs | Synthetic claims/inputs replace MIMIC results | WORDING_ONLY plus REGENERATE_ARTIFACT | Framework high | Scientific evidence regenerated | Audit nonclaims |

## Pulkit-owned systems

| Component | Current contract/module | v2 impact | Action | Code reuse | Artifact consequence | Future review |
|---|---|---|---|---|---|---|
| Vasopressor state | `src/labels/support_state.py` | State semantics retained; source mapping synthetic | NEW_ADAPTER | State engine full | Regenerate states | Review interval/rate mapping |
| Ventilation state | `src/labels/ventilation_state.py` | State semantics retained; MIMIC concept replaced | NEW_ADAPTER | State engine full | Regenerate states | Review invasive categories |
| Composite support label | `src/labels/organ_support.py` | OFF→ON/censoring retained | UNCHANGED | Full | Regenerate labels | Validate synthetic events |
| Endpoint freeze | `src/labels/endpoint_freeze.py`; signoff config | MIMIC blockers become generator-contract blockers | CONFIG_SUCCESSOR | Governance high | New freeze/signoff artifact | Human approvals required |
| Prediction schema | `configs/prediction_schema_v1.json`; `src/serving/prediction_schema.py` | Tasks/output meanings retained | UNCHANGED | Full | New responses/model metadata | Confirm no schema bump |
| PredictionPipeline | `src/serving/pipeline.py` | Orchestration retained | UNCHANGED | Full | Compose new selected bundle | End-to-end equality |
| Artifact resolver | `src/serving/artifacts.py` | Hash/compatibility retained; authorized scope vocabulary added later | NEW_ADAPTER | High | New manifest/models | Scope isolation tests |
| History | `src/serving/history.py` | Synthetic fixture concept becomes authorized data through a separate contract | NEW_ADAPTER | Truncation logic high | New timeline manifest | Do not promote fixture class silently |
| Preprocessing seam | `src/serving/preprocessing.py` | Injected builder design retained | UNCHANGED | Full | Bind rebuilt canonical builder/preprocessor | Training-serving equality |
| Explanation router | `src/explainability/router.py` | Family routing retained | UNCHANGED | Full | New model/output lineage | Confirm task targets |
| Integrated Gradients | `src/explainability/ig.py` | Data source independent | REGENERATE_ARTIFACT | Full after dependency | New attributions/baseline evidence | Freeze final baseline/environment |
| TreeSHAP | `src/explainability/tree_shap.py` | Data source independent | REGENERATE_ARTIFACT | Full after dependency | New model/feature-map attributions | Verify exact flatten map |
| FastAPI | `api/main.py`; `api/schemas.py` | Transport and task schema retained | WORDING_ONLY | Full | New composed bundle/metadata | Synthetic nonclaim language |
| Current SOFA/recovery | `src/serving/recovery.py`; display | Provider/reconstruction retained; source provenance synthetic | NEW_ADAPTER | Full arithmetic/interface | New current-score records | Verify six flags/version |
| Dashboard | `dashboard/` | Architecture retained; final synthetic/cardio wording required | WORDING_ONLY | Full | Regenerate catalog/screens/evidence | Preserve replay banner |
| Replay | `dashboard/replay.py` | Historical cutoff semantics retained | UNCHANGED | Full | Regenerate replay evidence | Cutoff safety |
| Integration | Phase-14 integration suite | Same path with authorized final synthetic artifacts | REGENERATE_ARTIFACT | Tests/harness full | Regenerate equality/compatibility reports | Hostile mutations |
| Packaging | README/runbook/demo validator | Final synthetic data must remain distinct from fixtures | CONFIG_SUCCESSOR | Framework high | New package manifest/fixture docs | Environment lock |
| Release evidence | `src/evidence/system.py`; Phase-16 audit | Claims/source change | WORDING_ONLY plus REGENERATE_ARTIFACT | Generator high | New release evidence | Non-owner reproduction |

## Cross-cutting invalidation chains

```text
synthetic generator/schema/cohort identity
  -> accepted canonical dataset
  -> split artifact
  -> fitted preprocessors/scalers/class weights
  -> model checkpoints and predictions
  -> selection
  -> support calibration and threshold
  -> selected manifest
  -> explanations and serving composition
  -> validation/final metrics and evidence

synthetic SOFA specification
  -> sofa_at(T) and six flags
  -> recovery labels/current SOFA
  -> recovery models/selection
  -> API reconstruction/dashboard/evidence

synthetic support event mapping
  -> support states
  -> composite labels/eligibility
  -> support model/selection/calibration/threshold
  -> selected manifest/serving/evidence
```

Historical fixtures, reviews, and framework tests remain historical evidence. They are not final scientific artifacts and must not be relabeled.
