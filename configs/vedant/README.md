# Vedant Configuration Area

No executable configuration is created in Phase 0.

Proposed engineering convention, pending team review:

- versioned YAML for human-authored training/search/scientific configuration when the authoritative documents call for YAML;
- JSON for contracts/manifests explicitly named JSON;
- repository-relative POSIX paths in portable metadata;
- exact-file-byte SHA-256 recorded alongside registered artifacts;
- no implicit defaults for scientific definitions or numeric project seed;
- search spaces frozen and hashed before the first search run;
- test-only values live in synthetic fixtures and cannot masquerade as project defaults.

This directory is Vedant-owned for timestamp/split/tensor/training/evaluation configuration, subject to the mandatory reviewers in the Work Division. Sanskruti and Pulkit retain ownership of their domain contracts.

`timestamp_spec_v1.yaml` is intentionally absent: its declaration and implementation belong to Phase 1/G0 contract work, not this infrastructure baseline.

