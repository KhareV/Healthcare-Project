# Sanskruti Final Scientific/Data Handoff

## Frozen authority

Phase 14 completes the Sanskruti scientific/data track for the synthetic adult cardiac retrospective-forecasting benchmark. The root authority is `artifacts/acceptance/g1_synthetic_data_freeze_v1.json`, SHA-256 `1fbcc1b0859c0052eb867aec15e89bd3f7df503ac318b30f92dd98ff9eeccf90`. The portable, hash-bound inventory is `artifacts/handoffs/sanskruti_phase14_handoff_v1.json`; every scientific parent is repository-relative and verified before either receiver is accepted.

The accepted model input is eight ordered six-hour bins with 21 temporal features, three raw static fields, eight transformed static fields, and a 520-column lossless XGBoost view. There are 12,222 canonical rows: 8,626 train, 1,754 validation, and 1,842 sealed test inventory rows. Test targets, predictions, errors, performance, and prevalence were not inspected.

## Vedant contract

Vedant consumes the existing Phase-10 rows through `data.xgb_canonical.Phase10XGBData`. The structured GRU view and flattened XGBoost view are exact alternate representations of the same values, observation masks, TSLO, padding, and statics. Recovery horizons retain independent eligibility masks and use the accepted recovery scaler; ICU time remains `log1p` remaining current-episode hours with no extra target scaler; support uses the accepted binary label, eligibility, and class weight. The preprocessor is transform-only.

The three Phase-12 references are XGBoost within-family validation winners, not final selected models. Phase 14 loads their native bundles, feature names, hashes, and lineage; it does not retrain or rerank them and does not run a GRU search.

## Pulkit contract

Pulkit consumes the accepted canonical timeline through `CanonicalTimelineHistoryProvider`, cutoff filtering through `SyntheticPointEventHistoryTruncator`, the same `SyntheticCanonicalFeatureBuilder`, and the frozen Phase-10 preprocessor. The thin history adapter also carries accepted support intervals whose onset is known by the cutoff, allowing the existing Pulkit state engines to reconstruct accepted support-state/rate channels. No future onset is exposed and the cessation bound is used only internally for state-as-of evaluation.

The receiver audit proves exact offline-versus-serving transformed equality and exact `CurrentSOFAProvider` equality against accepted `sofa_at(t)`. Serving makes zero `fit` or `fit_transform` calls. `PredictionPipeline` is interface-ready, but no final model bundle is fabricated.

## Verification quickstart

Run from the repository root:

```bash
shasum -a 256 artifacts/acceptance/g1_synthetic_data_freeze_v1.json
PYTHONPATH=src:. python3 scripts/audit_phase14_handoff.py
PYTHONPATH=src:tests:. python3 -m pytest -q tests/test_phase14_handoff.py
PYTHONPATH=src:tests:. python3 -m pytest -q
PYTHONPATH=src:. python3 -m experiments.registry_cli audit --root .
PYTHONPATH=src:. python3 scripts/audit_phase12_xgb_search.py
```

`scripts/build_phase14_handoff.py` deterministically regenerates the manifest and receiver evidence from frozen parents. It is not a scientific-generation or training command.

## Downstream boundary

Vedant must next run the governed GRU searches on this exact G1 lineage, compare best GRU and best XGBoost per task using validation only, and run the fixed LSTM sensitivity. Only afterward may the selected-model manifest, support calibration, validation threshold, and G3 be frozen. Pulkit may then bind selected models and family-specific explanations. The final test stays locked until G3.

Never recompute cohort, timestamps, bins, SOFA, labels, split, encoders, feature scaling, recovery scaling, support weight, or support state in a receiver. Any scientific-parent change requires a new artifact version, a new G1, a new Phase-14 handoff, and retraining of dependent models.
