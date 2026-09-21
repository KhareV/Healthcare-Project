# Sanskruti Rebuild Phase 10 — Synthetic Split and Preprocessing Review

## Outcome

Phase 10 is complete for the authorized final synthetic benchmark. The
obsolete MIMIC anchor-era rule was not reused and no synthetic years were
invented. The 2,000 retained subjects were assigned deterministically to
1,400 training, 300 validation, and 300 sealed final-test subjects. The
resulting model-ready train/validation row counts are 8,626 and 1,754. There
are 1,842 rows assigned to final test, but no transformed test file was
materialized before G3.

## Scientific and engineering decisions

The split is label-blind, outcome-blind, unstratified, and independent of
support prevalence or any model result. Exact raw trajectories are
fingerprinted after removing subject/stay/event identities and absolute
calendar time; clones are grouped before assignment. The final source data has
zero exact clones. SHA-256 ranking with seed `20260921` and a dedicated
namespace makes assignment deterministic and independent of input row order.

Training-only preprocessing learned 21 per-channel genuine-observation
medians and active-cell population z-score statistics. It preserves the
observation mask, structural padding mask, and raw-hour TSLO contract. Padding
is zero only in the dense post-transform tensor and is never reclassified as
clinical missingness. Raw statics became one standardized age feature plus
full one-hot sex and cardiac-condition vocabularies with unknown buckets,
giving eight encoded static fields.

Recovery target scaling was fit independently by horizon on 5,005 eligible
training rows at 24 hours and 1,868 at 48 hours. The support class weight used
only 5,559 eligible training rows: 2,445 positive and 3,114 negative, yielding
`pos_weight = 1.2736196319018405`. These are synthetic-benchmark training
statistics, not clinical estimates.

## Integration handoff

The feature preprocessor exposes a transform-only adapter compatible with
Pulkit's `FrozenPreprocessor` protocol. Direct Phase-10 transformation and
serving transformation call the same function and are equality-tested. The
recovery scaler and support class-weight files deserialize through Vedant's
existing `RecoveryTargetScaler` and `SupportClassWeight` implementations.

`synthetic_model_input_contract_v1` resolves the former placeholder tensor
shape: sequence `[B,8,21]`, observation mask `[B,8,21]`, TSLO `[B,8,21]`,
padding mask `[B,8]`, and encoded static vector `[B,8]`. XGBoost may only
flatten the same information, for a total flat dimension of 520; it may not
receive extra fields. Phase 11 model configurations must bind this contract
and must not reuse old smoke configs that disable TSLO/statics.

## Governance assertions

- Test subjects are assigned but test model-ready rows are not materialized.
- No test target statistic, metric, plot, calibration, or threshold exists.
- No model was trained and no search was run.
- No generator parameter was revised after split assignment.
- No prevalence or model performance selected any Phase-10 parameter.
- Existing v1 MIMIC split files were preserved as superseded history.
- Bulk train/validation JSONL products remain reproducible local build
  artifacts; contracts, hashes, split, and fitted transform artifacts are
  tracked.

## Verification commands

```bash
PYTHONPATH=src python3 scripts/audit_phase9_final.py
PYTHONPATH=src python3 scripts/audit_phase10_synthetic.py
PYTHONPATH=src:tests:. python3 -m pytest -q tests/test_synthetic_phase10.py
```

Phase 11+ remains outside this completion: no XGBoost/GRU training,
hyperparameter search, selection, calibration, thresholding, or final-test
evaluation was performed.
