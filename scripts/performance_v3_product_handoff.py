#!/usr/bin/env python3
"""Write artifacts/performance_v3/product_handoff_v1.json for Master Phase 2 (values read from frozen artifacts)."""
import hashlib, json, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from performance_v3.serving import explain_route

sha = lambda p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
sel = json.loads((ROOT / "artifacts/performance_v3/phase3/selected_models_v3.json").read_text())
promo = json.loads((ROOT / "artifacts/performance_v3/phase4/paired_v2_v3_metrics.json").read_text())["promotion"]
registry = "artifacts/performance_v3/research/research_registry_v1.json"
tasks = {}
for task, e in sel["tasks"].items():
    tasks[task] = {
        "promotion_decision": promo[task]["decision"], "serve": "V3" if promo[task]["decision"] == "PROMOTE_V3" else "V2_FROZEN",
        "family": e["family"], "feature_variant": e["feature_variant"], "n_estimators": e["n_estimators"],
        "model_ref": e["model_ref"], "model_sha256": e["model_sha256"], "feature_names_ref": e["feature_names_ref"],
        "feature_names_sha256": e["feature_names_sha256"], "postprocess": e["postprocess"],
        "explanation": explain_route(task, e["family"]),
        "explanation_domain": "RAW_MARGIN_PRE_CALIBRATION (not a decomposition of the calibrated probability)",
    }
out = {
    "artifact_version": "performance_v3_product_handoff_v1",
    "status": "SCIENCE_FROZEN_PHASE_2_READY",
    "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip(),
    "recommended_bundle": {"manifest_ref": "artifacts/performance_v3/phase3/selected_models_v3.json",
                           "manifest_sha256": sha("artifacts/performance_v3/phase3/selected_models_v3.json"),
                           "bundle_id": sel["bundle_id"], "tasks": tasks,
                           "calibrator": {"method": sel["calibrator"]["method"], "params_sha256": sel["calibrator"]["sha256"]},
                           "threshold": sel["threshold"]},
    "v2_fallback": {"manifest_ref": "artifacts/performance_v2/phase3/selected_models_v2.json",
                    "manifest_sha256": sha("artifacts/performance_v2/phase3/selected_models_v2.json"),
                    "warning": "Defect D1: served V2 ICU-time/support forecasts are far worse than the published V2 evaluation (see docs/performance_v3/DEFECT_D1.md). Keep only for rollback/comparison."},
    "control_comparator": {"manifest_ref": "artifacts/performance_v3/phase3/control/bundle_manifest.json",
                           "manifest_sha256": sha("artifacts/performance_v3/phase3/control/bundle_manifest.json")},
    "serving_adapter": {"module": "src/performance_v3/serving.py", "entry_points": ["V3Bundle.load", "V3Bundle.predict_rows", "V3Bundle.predict_stay_at", "V2Comparator.predict_rows"],
                        "caveats": ["predict_stay_at obtains the cutoff-truncated canonical row through the existing V2 serving runtime (V1 demo stays only)",
                                    "current_sofa must be frozen sofa_at(t) for every request (defect D1)",
                                    "GRU serving / Integrated Gradients are not part of the V3 bundle (all selected families are tree models)"]},
    "response_schema": ["stay_id", "prediction_time", "grid_index", "current_sofa", "recovery24_delta", "recovery48_delta", "recovery24_sofa", "recovery48_sofa",
                        "icu_raw_log_prediction", "icu_remaining_hours", "support_raw_probability", "support_calibrated_probability", "support_threshold",
                        "support_alert", "model_hashes", "bundle_id", "serving_version"],
    "data_quality_schema_ref": "configs/prediction_schema_v1.json",
    "research_registry": {"ref": registry, "sha256": sha(registry) if (ROOT / registry).exists() else None},
    "evidence": {"final_evaluation_doc": "docs/performance_v3/FINAL_V3_EVALUATION.md",
                 "final_metrics": "artifacts/performance_v3/phase4/final_metrics_v3.json",
                 "paired": "artifacts/performance_v3/phase4/paired_v2_v3_metrics.json",
                 "evaluation_freeze": "artifacts/performance_v3/governance/v3_final_evaluation_freeze_v1.json",
                 "figure_index": "docs/performance_v3/FIGURE_AND_TABLE_INDEX.md"},
    "safe_demo_data": {"allowed": ["configs/performance_v2/v2_demo_manifest_v1.json (V1 demo stays)", "LCE development cohort (artifacts/performance_v3/cohorts/LCE)", "development pool cohorts D1-D3"],
                       "forbidden": ["V1 test (namespace SYN, split test)", "V2 fresh test (SYN-V2)", "V3 sealed test (SYN-V3-T)"]},
    "provenance_endpoints_required": ["bundle manifest + per-task model sha256", "protocol + amendment hashes", "calibrator method/params hash", "threshold value", "cohort lineage (generation_lineage.json)", "defect D1 notice", "research registry"],
    "known_limitations": [
        "Synthetic data from one generator; no clinical or external validation.",
        "Absolute skill is modest: ICU time improves ~6.7% over a constant baseline, consistent with the generator's irreducible duration noise.",
        "Recovery48 V3 vs V2-recipe refit paired CI includes zero; other V3 gains over that control are small.",
        "TreeSHAP explains the raw margin, not the calibrated probability.",
        "Learning-curve and robustness results are development evidence; only the sealed test is final evidence.",
        "V2's historical published ICU/support metrics are inflated by defect D1 and must not be presented as deployed performance.",
    ],
}
(ROOT / "artifacts/performance_v3/product_handoff_v1.json").write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
print("handoff written; registry sha:", out["research_registry"]["sha256"])
