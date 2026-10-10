# Defect D1 — `current_sofa` missingness leaked horizon availability into V2

Status: found 2026-10-10 during V3 development, before any V3 selection result or test access.
Governing record: `configs/performance_v3/protocol_amendment_v1_1.yaml`
(hash in `artifacts/performance_v3/governance/v3_protocol_amendment_v1_1_freeze.json`).
No V1/V2 artifact was modified; this document only describes them.

## Mechanism

The pre-split scientific package stores `baseline_sofa` only when the 24-hour recovery label can be
computed (`t + 24h <= outtime`). Performance-V2's `B__current_sofa` (present in every V2 feature variant)
was built from that field, so its *missingness* equals "the stay ends within 24 h of this cutoff".
SOFA(t) itself depends only on data at or before `t` and is available for every legal cutoff
(frozen `data.synthetic.sofa.sofa_at`; equal to `baseline_sofa` wherever the latter exists).

## Evidence (development data only)

| Check | Value |
|---|---|
| Development-pool rows with null `baseline_sofa` | 20,656 of 48,859 (42.3 %) |
| Null exactly when recovery24-ineligible | yes |
| Frozen `sofa_at(t)` available for those rows | 100 % |
| Support-eligible rows with null baseline: positive rate | 1.00 (vs 0.349 otherwise) |
| Other feature columns: elapsed-time-matched NaN-rate gap | <= 0.021 (393 columns scanned) |
| Frozen V2 on LCE, as evaluated vs as served: ICU weighted-median AE | 5.15 h vs 18.26 h |
| Frozen V2 on LCE, as evaluated vs as served: support raw AUPRC | 0.781 vs 0.578 |

On the sealed V3 test (one-time evaluation) the same contrast is ICU 5.29 h vs 18.25 h and calibrated
support AUPRC 0.693 vs 0.520 (`docs/performance_v3/FINAL_V3_EVALUATION.md`).

## Consequences

* V2's published ICU-time and support results (and the Phase-1 "Group B is decisive" attribution) were
  inflated by the null pattern; recovery tasks are unaffected because eligible rows always have a value.
* The V2 serving runtime (`src/serving/v2/runtime.py`) computes `current_sofa` for every cutoff, so the
  deployed V2 ICU-time and support forecasts behave like "V2 as served", far worse than the published
  numbers (ICU worse than the naive constant).
* Benchmark V1 is unaffected (its flat feature map excludes current SOFA).

## Correction in V3

`B__current_sofa` is the frozen `sofa_at(t)` total for every row; absence is a hard error; a
null-pattern audit (`src/performance_v3/null_audit.py`) must show every column's matched gap <= 0.05.
Artifacts built on the pre-correction cache (79-trial search ledger, first learning curve, first caches)
are retained under `INVALIDATED_*` / `invalidated_pre_D1/` and are not used.

## Not resolved

* A V2-recipe model retrained inside the pre-correction V3 pipeline scored differently (raw AUPRC 0.846 on
  LCE) from the frozen V2 file (0.781) despite 0.994 in-sample correlation; the cause was not isolated and
  is moot after the correction, but it is recorded here rather than hidden.
* The "clean subset" (>= 24 h follow-up) is itself selected on the outcome for ICU time and support, so it
  is not a leak-free comparison for those tasks.
