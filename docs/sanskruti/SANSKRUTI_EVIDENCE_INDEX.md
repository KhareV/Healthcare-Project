# Sanskruti Evidence Index

- Governance: `docs/governance/project_scope_v2.md`, `configs/governance/project_scope_v2.json`
- Generator and provenance: `configs/synthetic/final_benchmark_v1.json`, `configs/synthetic/latent_process_v1.yaml`, `configs/synthetic/support_process_v1.yaml`, `artifacts/data/synthetic/final/phase9_final_v1/synthetic_dataset_manifest_v1.json`
- Cohort and structural grid: `artifacts/data/synthetic/cohorts/final/phase9_final_v1/synthetic_cohort_manifest_v2.json`, adjacent `structural_index.jsonl`
- Timeline: `artifacts/data/synthetic/timelines/final/phase9_final_v1/synthetic_processed_manifest_v1.json`
- SOFA: `configs/synthetic/sofa_spec_v1.json`
- Features: `configs/synthetic/feature_schema_v2.json`, `artifacts/data/synthetic/features/final/phase9_final_v2/canonical_feature_inputs.jsonl`
- Labels: `docs/evidence/data/final_target_dictionary_v1.md`, `artifacts/data/synthetic/pre_split/final/phase9_final_v1/pre_split_scientific_package.jsonl`
- Split and preprocessing: `artifacts/splits/synthetic_split_v2.csv`, `artifacts/preprocessors/synthetic_feature_preprocessor_v1.json`, accepted recovery-scaler and support-weight artifacts in the same directory
- XGBoost parity and search: `artifacts/manifests/synthetic_phase11_xgb_input_manifest_v1.json`, `artifacts/search/xgb/phase12/manifests/phase12_search_manifest_v1.json`, `artifacts/search/xgb/phase12/best_xgb_candidates_v1.json`
- QA and G1: `docs/evidence/data/data_qa_report_v1.md`, `docs/evidence/data/leakage_audit_v1.md`, `docs/evidence/data/reproducibility_report_v1.md`, `artifacts/acceptance/g1_synthetic_data_freeze_v1.json`
- Final handoff: `artifacts/handoffs/sanskruti_phase14_handoff_v1.json`, receiver acceptance artifacts, completion matrix, ownership audit, supersession map, downstream task list, and status in `artifacts/handoffs/`

The handoff manifest is the machine-readable source for exact hashes. This index is navigational and does not replace hash validation.

