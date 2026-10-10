# V3.1 Latent-State Diagnostic (DEVELOPMENT_PRIVILEGED_DIAGNOSTIC_ONLY)

Privileged simulator-state predictors on a fresh development cohort. These are **not** clinical bounds, **not** serving inputs, and Oracle A/B are learned (finite-sample) models, not ceilings.
Protocol `configs/performance_v3_1/diagnostic_protocol_v1.yaml` (SHA-256 `b46863b7e4057d0f84060eb03ab14f71f6ce082eed908d060f0f41dd1819d78f`), frozen before results.

## Cohorts

| Cohort | Subjects | Eligible rows | Eligible stays | Stay-balanced prevalence |
|---|---:|---:|---:|---:|
| DIAG_EVAL (SYN-V31-DE) | 900 | 3353 | 697 | 0.4225 |
| DIAG_FIT (SYN-V31-DF) | 1600 | 6149 | 1240 | 0.4356 |

## Evaluation on DIAG_EVAL (identical rows and stays for every model)

| Model | AUPRC | AUROC | Brier | Log loss | Gap vs V3 AUPRC [95% CI] |
|---|---:|---:|---:|---:|---|
| REFERENCE_V3 | 0.6600 | 0.7422 | 0.2021 | 0.5886 | — |
| OBSERVED_REFIT_CONTROL_XGB | 0.6223 | 0.7207 | 0.2114 | 0.6111 | -0.0377 [-0.0568, -0.0184] |
| ORACLE_A_LOGISTIC | 0.6670 | 0.7394 | 0.2032 | 0.5909 | +0.0071 [-0.0134, +0.0258] |
| ORACLE_A_XGBOOST | 0.6594 | 0.7391 | 0.2030 | 0.5893 | -0.0005 [-0.0217, +0.0204] |
| ORACLE_B_LOGISTIC | 0.6857 | 0.7538 | 0.1984 | 0.5828 | +0.0257 [-0.0016, +0.0520] |
| ORACLE_B_XGBOOST | 0.6634 | 0.7505 | 0.2018 | 0.5914 | +0.0034 [-0.0260, +0.0320] |
| ORACLE_C_MC | 0.7189 | 0.7774 | 0.1887 | 0.5557 | +0.0589 [+0.0320, +0.0848] |

Oracle C (Monte Carlo conditional generator, M=200): mean binomial SE 0.037; reliability slope 1.047, ECE 0.028, valid=True.

## Bounded verification (declared before running)

B learners refit on 7900 privileged subjects (30908 rows), same presets, same DIAG_EVAL.

| Model | AUPRC | AUROC | Gap vs V3 [95% CI] |
|---|---:|---:|---|
| ORACLE_B_LARGE_LOGISTIC | 0.6950 | 0.7570 | +0.0350 [+0.0073, +0.0615] |
| ORACLE_B_LARGE_XGBOOST | 0.7002 | 0.7647 | +0.0403 [+0.0152, +0.0642] |

## Gate A

* Primary diagnostic: best B gap +0.0257 (lower bound -0.0016); Oracle C gap +0.0589 (lower bound +0.0320); audits pass = True -> passes = False.
* After the single declared verification: ORACLE_B_LARGE_XGBOOST gap +0.0403 (lower bound +0.0152) -> passes_gate_a = True.

## Interpretation limits

* A privileged gap says information exists in the simulator state (hidden subject effects, drift, equilibrium, shocks), not that vitals/labs can recover it.
* Even exact current latent coordinates (Oracle A) are about as good as the observed-data model: most of the privileged gap is hidden per-subject constants.
* AUPRC depends on prevalence and on the eligibility/censoring selection; compare only within this table.
