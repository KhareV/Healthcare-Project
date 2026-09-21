# Phase 13 Golden Cases

| Case | Subject | Stay | Cutoff | Expected | Actual | Status |
|---|---|---|---|---|---|---|
| early cutoff with structural padding | SYN-S-00000001 | SYN-E-00000001 | 2101-05-31T06:17:26.000000Z | padding prefix | 4 | PASS |
| genuine missingness in active bins | SYN-S-00000001 | SYN-E-00000001 | 2101-05-31T06:17:26.000000Z | null with false observation mask | 20 | PASS |
| 24h eligible / 48h ineligible | SYN-S-00000001 | SYN-E-00000001 | 2101-05-31T12:17:26.000000Z | True/False | True/False | PASS |
| both recovery horizons eligible | SYN-S-00000001 | SYN-E-00000001 | 2101-05-31T06:17:26.000000Z | True/True | True/True | PASS |
| current vasopressor ON | SYN-S-00000001 | SYN-E-00000001 | 2101-06-01T12:17:26.000000Z | vasopressor_on=1 | 1.0 | PASS |
| current invasive ventilation ON | SYN-S-00000005 | SYN-E-00000005 | 2107-10-04T05:45:52.000000Z | invasive_ventilation_on=1 | 1.0 | PASS |
| future OFF-to-ON positive | SYN-S-00000001 | SYN-E-00000001 | 2101-05-31T12:17:26.000000Z | eligible=True,label=1 | True/1 | PASS |
| early-exit support censoring | SYN-S-00000001 | SYN-E-00000001 | 2101-06-01T12:17:26.000000Z | ineligible/censored | False/None | PASS |
| ICU remaining-time formula | SYN-S-00000001 | SYN-E-00000001 | 2101-05-31T06:17:26.000000Z | exact log1p hours | 3.972695663737869 | PASS |
| future cessation hidden | SYN-S-00000001 | SYN-E-00000001 | 2101-06-01T12:17:26.000000Z | current state only; future end absent | [1.0, 0.0] | PASS |
