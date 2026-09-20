# Vedant Methods Summary

- Split: subject-disjoint coarse temporal holdout; train eras 2008–2013, validation
  2014–2016, test 2017–2019.
- Prediction grid: t_k = intime + 24h + 6h*k for k=0..11, provided t_k <= outtime-6h.
- Tensor: eight 6-hour bins over (t-48h,t], with distinct padding, observation masks,
  and configured TSLO; no pre-lookback seeding.
- GRU: unidirectional recurrent encoder; recovery has two independent outputs, ICU time
  one log-space output, and support one logit.
- Search governance: planned exact-30 XGBoost and exact-30 GRU candidates per task;
  LSTM is one fixed sensitivity run and not serving-eligible.
- Selection: recovery minimizes validation MAE24; ICU time minimizes validation weighted
  median absolute error in hours; support maximizes validation uncalibrated AUPRC.
- Calibration: only the selected support classifier receives validation-fitted isotonic
  calibration and a validation-F1 threshold.
- Test governance: test access requires active valid G3 and is consumed once.
- Reproducibility: artifact hashes, registry lineage, exact environment, and non-owner
  reproduction are independently reported; absent layers remain blocked.
