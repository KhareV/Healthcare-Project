#!/usr/bin/env python3
"""Render V3.1 diagnostic/closure documents and the Phase-2 handoff from frozen result JSON (no hand-typed metrics)."""
import hashlib, json, subprocess
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
J = lambda p: json.loads((ROOT / p).read_text())
sha = lambda p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
d = "artifacts/performance_v3_1/diagnostic/"
met, boot, comp, ver = J(d + "oracle_metrics.json"), J(d + "oracle_bootstrap.json"), J(d + "oracle_comparison.json"), J(d + "oracle_verification_v1.json")
coh, rt = J(d + "cohort_manifest.json"), J(d + "oracle_runtime.json")
sel, hold = J("artifacts/performance_v3_1/phase2/candidate_selection_v1.json"), J("artifacts/performance_v3_1/phase2/hold_evaluation_v1.json")
freeze = J("artifacts/performance_v3_1/phase3/candidate/candidate_freeze_v1.json")
M = met["metrics"]
rows = ["REFERENCE_V3", "OBSERVED_REFIT_CONTROL_XGB", "ORACLE_A_LOGISTIC", "ORACLE_A_XGBOOST", "ORACLE_B_LOGISTIC", "ORACLE_B_XGBOOST", "ORACLE_C_MC"]
L = ["# V3.1 Latent-State Diagnostic (DEVELOPMENT_PRIVILEGED_DIAGNOSTIC_ONLY)", "",
 "Privileged simulator-state predictors on a fresh development cohort. These are **not** clinical bounds, **not** serving inputs, and Oracle A/B are learned (finite-sample) models, not ceilings.",
 f"Protocol `configs/performance_v3_1/diagnostic_protocol_v1.yaml` (SHA-256 `{sha('configs/performance_v3_1/diagnostic_protocol_v1.yaml')}`), frozen before results.", "",
 "## Cohorts", "", "| Cohort | Subjects | Eligible rows | Eligible stays | Stay-balanced prevalence |", "|---|---:|---:|---:|---:|"]
for k, v in coh.items():
    L.append(f"| {k} ({v['namespace']}) | {v['subjects']} | {v['eligible_support_rows']} | {v['eligible_stays']} | {v['prevalence_stay_balanced']:.4f} |")
L += ["", "## Evaluation on DIAG_EVAL (identical rows and stays for every model)", "", "| Model | AUPRC | AUROC | Brier | Log loss | Gap vs V3 AUPRC [95% CI] |", "|---|---:|---:|---:|---:|---|"]
for k in rows:
    m = M[k]; b = boot["comparisons"].get(k)
    gap = f"{b['diff_point']:+.4f} [{b['ci95'][0]:+.4f}, {b['ci95'][1]:+.4f}]" if b else "—"
    L.append(f"| {k} | {m['auprc']:.4f} | {m['auroc']:.4f} | {m['brier']:.4f} | {m['log_loss']:.4f} | {gap} |")
L += ["", f"Oracle C (Monte Carlo conditional generator, M={met['oracle_c']['M']}): mean binomial SE {met['oracle_c']['mean_binomial_se']:.3f}; reliability slope {comp['audits']['mc_reliability']['slope']:.3f}, ECE {comp['audits']['mc_reliability']['ece']:.3f}, valid={comp['audits']['mc_reliability']['valid']}.",
 "", "## Bounded verification (declared before running)", "", f"B learners refit on {ver['fit_subjects']} privileged subjects ({ver['fit_rows']} rows), same presets, same DIAG_EVAL.", "", "| Model | AUPRC | AUROC | Gap vs V3 [95% CI] |", "|---|---:|---:|---|"]
for k, v in ver["models"].items():
    L.append(f"| {k} | {v['metrics']['auprc']:.4f} | {v['metrics']['auroc']:.4f} | {v['diff_vs_reference']:+.4f} [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}] |")
g = comp["gate_a"]; v = ver["gate_a_after_verification"]
L += ["", "## Gate A", "", f"* Primary diagnostic: best B gap {g['b_gap']:+.4f} (lower bound {g['b_gap_lb95']:+.4f}); Oracle C gap {g['c_gap']:+.4f} (lower bound {g['c_gap_lb95']:+.4f}); audits pass = {g['audits_pass']} -> passes = {g['passes_gate_a']}.",
 f"* After the single declared verification: {v['best']} gap {v['gap']:+.4f} (lower bound {v['lb95']:+.4f}) -> passes_gate_a = {v['passes_gate_a']}.",
 "", "## Interpretation limits", "",
 "* A privileged gap says information exists in the simulator state (hidden subject effects, drift, equilibrium, shocks), not that vitals/labs can recover it.",
 "* Even exact current latent coordinates (Oracle A) are about as good as the observed-data model: most of the privileged gap is hidden per-subject constants.",
 "* AUPRC depends on prevalence and on the eligibility/censoring selection; compare only within this table.", ""]
(ROOT / "docs/performance_v3_1/LATENT_STATE_DIAGNOSTIC.md").write_text("\n".join(L), encoding="utf-8")

H = hold["metrics"]; P = hold["paired_vs_v3"]; G = hold["gate_c"]
D = ["# V3.1 Candidate Development and Terminal Decision", "",
 f"Protocol `configs/performance_v3_1/scientific_protocol_v1.yaml` (SHA-256 `{sha('configs/performance_v3_1/scientific_protocol_v1.yaml')}`).", "",
 "## Search (5-fold grouped CV, 32 pre-declared trials)", "", "| Feature family | best OOF AUPRC |", "|---|---:|"] + [f"| {k} | {v:.4f} |" for k, v in sel["by_family_best"].items()] + [
 "", f"All {sel['tie_group']} trials lie within the 0.003 tie tolerance of the best; by the frozen tie rule the selected set is **{sel['winner']['variant']}** ({sel['winner']['n_features']} features, AUPRC {sel['winner']['primary_mean']:.4f}).",
 "", "## Held-out development decision set V31_HOLD (single comparison, frozen candidate)", "", "| | AUPRC | AUROC | Brier | Log loss | ECE |", "|---|---:|---:|---:|---:|---:|"]
for k, m in H.items():
    D.append(f"| {k} | {m['auprc']:.4f} | {m['auroc']:.4f} | {m['brier']:.4f} | {m['log_loss']:.4f} | {m['ece']:.4f} |")
D += ["", f"Paired difference candidate − frozen V3: AUPRC {P['diff_auprc']:+.4f} [{P['ci95'][0]:+.4f}, {P['ci95'][1]:+.4f}], AUROC {P['diff_auroc']:+.4f}, Brier {P['diff_brier']:+.4f}, log loss {P['diff_log_loss']:+.4f}.",
 f"Per-seed AUPRC gaps {[round(x,4) for x in P['per_seed_auprc_gaps']]}; thirds {[round(x,4) for x in P['thirds_auprc_gaps']]}.", "",
 "## Gate C", "", "```", json.dumps(G, indent=1), "```", "",
 "**Decision: V3 retained for all four heads. No V3.1 sealed test was generated or opened.**", ""]
(ROOT / "docs/performance_v3_1/V3_1_DEVELOPMENT_AND_DECISION.md").write_text("\n".join(D), encoding="utf-8")

v3 = J("artifacts/performance_v3/product_handoff_v1.json")
out = {"artifact_version": "performance_v3_1_product_handoff_v1", "outcome": "V3_RETAINED_AFTER_NEGATIVE_V3_1_DEVELOPMENT",
       "terminal_status": "V3_1_STOPPED_AFTER_DEVELOPMENT — V3_RETAINED — NO_NEW_FINAL_TEST_CONSUMED",
       "effective_release": {"manifest_ref": "artifacts/performance_v3/phase3/selected_models_v3.json", "manifest_sha256": sha("artifacts/performance_v3/phase3/selected_models_v3.json"),
                             "heads": {t: "frozen V3" for t in ("recovery24", "recovery48", "icu_stay_time", "organ_support")}},
       "v3_handoff_ref": "artifacts/performance_v3/product_handoff_v1.json", "v3_handoff_sha256": sha("artifacts/performance_v3/product_handoff_v1.json"),
       "v3_1_support_candidate": {"status": "NOT_PROMOTED_GATE_C_FAILED", "freeze_ref": "artifacts/performance_v3_1/phase3/candidate/candidate_freeze_v1.json",
                                  "serve": False, "note": "development candidate only; never a serving model"},
       "privileged_diagnostics": {"status": "DEVELOPMENT_PRIVILEGED_DIAGNOSTIC_ONLY", "never_serving_inputs": True,
                                  "artifacts": ["artifacts/performance_v3_1/diagnostic/oracle_metrics.json", "artifacts/performance_v3_1/diagnostic/oracle_comparison.json", "artifacts/performance_v3_1/diagnostic/oracle_verification_v1.json"]},
       "new_final_tests": "NONE_CREATED", "v3_final_test_ledger": "artifacts/performance_v3/governance/v3_final_test_access_state.json (FINAL_RUN_COMPLETED, unchanged)",
       "phase_2_instructions": ["Select the effective release from this handoff (currently the frozen V3 four-head bundle); do not hard-code V3 support.",
                                "Keep the defect D1 disclosure (docs/performance_v3/DEFECT_D1.md).",
                                "Do not re-run or re-open any sealed final test; V3.1 produced none.",
                                "Surface the diagnostic as research evidence only: privileged oracle results are not serving metrics.",
                                "Product UI plan unchanged; no V3.1 serving adapter is required."],
       "evidence": {"diagnostic_doc": "docs/performance_v3_1/LATENT_STATE_DIAGNOSTIC.md", "decision_doc": "docs/performance_v3_1/V3_1_DEVELOPMENT_AND_DECISION.md"},
       "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()}
(ROOT / "artifacts/performance_v3_1/product_handoff.json").write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
print("\n".join(L[:40]))
