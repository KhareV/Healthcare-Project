# Phase 21 Review Package

- Real reproducibility: blocked because exact environment, real Phase-19/20/G4
  artifacts, frozen tolerances, a designated run, and non-owner evidence are
  absent.
- Synthetic CI reproducibility: implemented.
- Full product reproduction: deferred to Pulkit.
- Upstream source-data reproduction: pending Sanskruti/team reproduction.

Reviewer entrypoints are `configs/reproducibility_v1.json`,
`src/reproducibility/`, `docs/REPRODUCIBILITY.md`, and the three machine-readable
artifacts under `artifacts/reproducibility/`.

Registered artifact hashes:

- reproduction manifest:
  `8aae142a0a3d48cb3211c786d7d35e73c8e6e0c390a01a18be41c2e202843960`
- reproduction report:
  `714d84bce5abdd29c3132bddad5731f9a410f6e0cd9e4f75e398f0fcbb5d1110`
- blocked cross-member status:
  `fe4bb7f7700f2464b1ca9a754144e81d703fa4483de55e76956ccda1bbe14839`

The audit is byte-stable after registration: Phase-21 artifacts are excluded
from their own input snapshot to prevent self-referential hash drift.
