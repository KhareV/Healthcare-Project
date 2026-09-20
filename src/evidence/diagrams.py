"""Report-ready architecture diagrams derived from implemented model contracts."""


def model_diagrams_markdown() -> str:
    return """# Vedant Model Architecture Evidence

Status: **FINAL** for implemented method structure; no selected real family is claimed.

## Recovery GRU

```mermaid
flowchart LR
  A[Canonical values B×8×F] --> D[Concatenate values + observation masks + configured TSLO]
  B[Prefix padding mask] --> D
  D --> E[Unidirectional GRU encoder]
  E --> F[Dropout + linear head: 2 outputs]
  F --> G[Standardized DeltaSOFA24 and DeltaSOFA48]
  G --> H[Frozen train-fitted inverse target transform]
  H --> I[Raw, unclipped delta evaluation by horizon]
```

The two recovery outputs are independent relative to the same SOFA(t) baseline.

## Remaining ICU stay time GRU

```mermaid
flowchart LR
  A[Canonical values/masks/configured TSLO] --> B[Unidirectional GRU]
  B --> C[One log1p remaining-current-ICU-time output]
  C --> D[Clip log prediction at 0]
  D --> E[expm1]
  E --> F[Remaining time until current ICU stay ends, hours]
```

## New Organ-Support Initiation Risk GRU

```mermaid
flowchart LR
  A[Canonical values/masks/configured TSLO] --> B[Unidirectional GRU]
  B --> C[One logit]
  C --> D[Sigmoid: raw probability]
  D --> E[Selected-model validation-fitted isotonic transform]
  E --> F[Calibrated probability]
  F --> G[Frozen validation-F1 threshold]
```

Calibration and threshold stages are method contracts only here; real artifacts are absent.

## XGBoost/GRU information parity

```mermaid
flowchart TD
  A[Same canonical 8-bin values, masks, TSLO and permitted statics] --> B[Flatten representation]
  A --> C[Preserve 8-step sequence representation]
  B --> D[XGBoost]
  C --> E[GRU]
```

The model families receive the same information; only representation differs. No attention,
transformer, CNN, bidirectional recurrence, or unfrozen static-input path is implied.
"""
