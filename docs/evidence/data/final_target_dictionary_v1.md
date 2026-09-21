# Final Target Dictionary

| Target | Formula | Eligibility | Input role |
|---|---|---|---|
| DeltaSOFA24 | `SOFA(t+24)-SOFA(t)` | independent 24h follow-up | LABEL_ONLY / prohibited from X |
| DeltaSOFA48 | `SOFA(t+48)-SOFA(t)` | independent 48h follow-up | LABEL_ONLY / prohibited from X |
| ICU time | `log1p((outtime-t) hours)` | legal structural cutoff | LABEL_ONLY / outtime prohibited from X |
| Organ support | qualifying vasopressor or invasive-ventilation OFF-to-ON in `(t,t+24h]` | at-risk and event/full follow-up rules | LABEL_ONLY / eligibility and future onset prohibited from X |
