"""Performance-v2 Phase 1, Parts 16-18: performance-ceiling table, the
automatic v2-path decision, and the exact frozen Phase-2 plan.

DEVELOPMENT_DIAGNOSTIC_ONLY. Synthesizes every prior Phase-1 artifact; makes
no new measurements.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.synthetic.config import canonical_json_bytes  # noqa: E402
from vedant_infra.hashing import sha256_file  # noqa: E402

OUT_DIR = ROOT / "artifacts/performance_v2/phase1"


def _load(name):
    return json.loads((OUT_DIR / name).read_text())


def main() -> None:
    v1_perf = _load("v1_validation_performance_summary.json")
    oracle = _load("observable_oracle/observable_oracle_summary_v1.json")
    latent = _load("latent_oracle/latent_oracle_summary_v1.json")
    baselines = _load("stronger_baselines_v1.json")
    ablation = _load("feature_ablation_v1.json")

    ceiling_rows = [
        {
            "task": "recovery_24h", "naive": baselines["recovery"]["recovery24"]["R0_naive_zero_delta"]["mae"],
            "current_selected_validation": v1_perf["recovery"]["best_xgb_validation"]["mae24"],
            "strong_simple_baseline": baselines["recovery"]["recovery24"]["R2_ridge_diagnostic_features"]["metrics"]["mae"],
            "observable_oracle": oracle["recovery"]["recovery24"]["metrics"]["mae"],
            "latent_oracle": latent["recovery24"]["metrics"]["mae"],
            "metric": "mae (lower is better)",
            "estimated_headroom_pct_of_current": None,
            "bottleneck": "REPRESENTATION_LIMITED",
        },
        {
            "task": "recovery_48h", "naive": baselines["recovery"]["recovery48"]["R0_naive_zero_delta"]["mae"],
            "current_selected_validation": v1_perf["recovery"]["best_xgb_validation"]["mae48"],
            "strong_simple_baseline": baselines["recovery"]["recovery48"]["R2_ridge_diagnostic_features"]["metrics"]["mae"],
            "observable_oracle": oracle["recovery"]["recovery48"]["metrics"]["mae"],
            "latent_oracle": latent["recovery48"]["metrics"]["mae"],
            "metric": "mae (lower is better)",
            "estimated_headroom_pct_of_current": None,
            "bottleneck": "REPRESENTATION_LIMITED",
        },
        {
            "task": "icu_stay_time", "naive": baselines["icu_stay_time"]["I0_naive_train_median"]["metrics"]["median_absolute_error"],
            "current_selected_validation": v1_perf["icu_stay_time"]["best_gru_validation"]["median_absolute_error_hours"],
            "strong_simple_baseline": baselines["icu_stay_time"]["I2_conditional_median_fine_grid"]["metrics"]["median_absolute_error"],
            "observable_oracle": oracle["icu_stay_time"]["metrics"]["median_absolute_error"],
            "latent_oracle": latent["icu_stay_time"]["metrics"]["median_absolute_error"],
            "metric": "median_absolute_error hours (lower is better)",
            "estimated_headroom_pct_of_current": None,
            "bottleneck": "REPRESENTATION_LIMITED",
        },
        {
            "task": "organ_support", "naive": baselines["organ_support"]["prevalence_baseline"]["metrics"]["auprc"],
            "current_selected_validation": v1_perf["organ_support"]["best_xgb_validation"]["auprc"],
            "strong_simple_baseline": baselines["organ_support"]["logistic_diagnostic_features"]["metrics"]["auprc"],
            "observable_oracle": oracle["organ_support"]["metrics"]["auprc"],
            "latent_oracle": latent["organ_support"]["metrics"]["auprc"],
            "metric": "auprc (higher is better)",
            "estimated_headroom_pct_of_current": None,
            "bottleneck": "REPRESENTATION_LIMITED",
        },
    ]
    for row in ceiling_rows:
        lower_is_better = "lower is better" in row["metric"]
        current, oracle_value = row["current_selected_validation"], row["observable_oracle"]
        if lower_is_better:
            row["estimated_headroom_pct_of_current"] = round(100.0 * (current - oracle_value) / current, 2)
        else:
            row["estimated_headroom_pct_of_current"] = round(100.0 * (oracle_value - current) / current, 2)

    ceiling = {
        "status": "DEVELOPMENT_DIAGNOSTIC_ONLY",
        "artifact_version": "performance_v2_phase1_performance_ceiling_summary_v1",
        "rows": ceiling_rows,
        "interpretation_note": (
            "estimated_headroom_pct_of_current is the relative improvement of "
            "the observable oracle over the CURRENT v1 selected model's own "
            "validation metric (never TEST). All four rows are labeled "
            "REPRESENTATION_LIMITED: the observable oracle materially beats "
            "the current model on every task (Part 7), the feature-group "
            "ablation (Part 15) shows this gain is overwhelmingly attributable "
            "to structural Group-B features (elapsed time, cutoff index, "
            "current SOFA) that are simply absent from the current canonical "
            "representation, and the latent oracle does NOT exceed the "
            "observable oracle on any task (Part 8) -- ruling out "
            "OBSERVABILITY_LIMITED as the primary bottleneck. A real, "
            "quantified GENERATOR_STOCHASTICITY_LIMITED component exists "
            "(Part 9-10) but is secondary: it does not prevent the observable "
            "oracle from already closing most of the current-vs-naive gap."
        ),
    }
    path = OUT_DIR / "performance_ceiling_summary_v1.json"
    path.write_bytes(canonical_json_bytes(ceiling))
    print("wrote", path, sha256_file(path))

    # Markdown table
    md_lines = [
        "| Task | Naive | Current selected (validation) | Strong simple baseline | Observable oracle | Latent oracle | Headroom vs current | Bottleneck |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in ceiling_rows:
        md_lines.append(
            f"| {row['task']} | {row['naive']:.4f} | {row['current_selected_validation']:.4f} | "
            f"{row['strong_simple_baseline']:.4f} | {row['observable_oracle']:.4f} | {row['latent_oracle']:.4f} | "
            f"{row['estimated_headroom_pct_of_current']:+.1f}% | {row['bottleneck']} |"
        )
    (OUT_DIR / "performance_ceiling_table_v1.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print("wrote", OUT_DIR / "performance_ceiling_table_v1.md")

    # --- Part 17/18: decision + exact plan ---
    decision = {
        "status": "DECIDED",
        "artifact_version": "performance_v2_phase1_selected_v2_strategy_v1",
        "decision_authority": "USER_DELEGATED_AI_PROJECT_DECISION",
        "selected_path": "PATH_A_FEATURE_AND_MODEL_V2",
        "decision_basis": [
            "Observable oracle materially beats the current v1 selected model on VALIDATION for all 4 tasks (Part 7): "
            "recovery24 {:.1f}%, recovery48 {:.1f}%, icu_stay_time {:.1f}%, organ_support {:+.1f}% (AUPRC).".format(
                ceiling_rows[0]["estimated_headroom_pct_of_current"], ceiling_rows[1]["estimated_headroom_pct_of_current"],
                ceiling_rows[2]["estimated_headroom_pct_of_current"], ceiling_rows[3]["estimated_headroom_pct_of_current"],
            ),
            "Feature-group ablation (Part 15) shows this gain is overwhelmingly attributable to Group B alone "
            "(elapsed_episode_hours_at_t, cutoff_index, hours_since_first_eligible_cutoff, current baseline SOFA) -- "
            "features that are lawful, already computable at cutoff t, and simply absent from the current canonical "
            "representation (Part 3 feature audit).",
            "Latent oracle (Part 8) does not exceed the observable oracle on any task, and underperforms the naive "
            "baseline on 3 of 4 -- no evidence that clinically meaningful state is hidden from observables that "
            "current features fail to expose. This rules out PATH C (GENERATOR_OBSERVABILITY_V2).",
            "Generator predictability/stochasticity analysis (Parts 9-10) finds a real, quantified irreducible noise "
            "component (ICU duration noise_scale=0.35 log-hours; latent forward dispersion ~0.40 std at 24-48h) but "
            "it does not prevent the observable oracle from already closing most of the naive-vs-current gap, so this "
            "is a secondary, not dominant, limitation. Generator mechanics (mean-reverting latent process, logistic "
            "hazard-style support initiation, log-linear duration) are physiologically plausible temporal dynamics, "
            "not a scientifically implausible or degenerate forecasting problem -- PATH D is not justified.",
            "No single dominant target/objective mismatch was identified as the primary bottleneck (Part 13); "
            "objective refinements are included as secondary, lower-priority Phase-2 work, not the primary driver.",
        ],
        "tasks": {
            "recovery": {
                "feature_groups_to_add": [
                    "B (highest priority, largest single gain): elapsed_episode_hours_at_t, cutoff_index, hours_since_first_eligible_cutoff, current baseline SOFA",
                    "F (secondary, small but real gain at 24h): support-history duration-in-state/transition features",
                    "C/D/E (optional, lower priority): ablation showed mixed/marginal/occasionally negative incremental value once B is present -- include only with regularization/feature-selection, not wholesale",
                ],
                "target_formulation": "unchanged: raw unclipped DeltaSOFA24/DeltaSOFA48, independent per-horizon models (Part 13 found no clear case to change this)",
                "models_to_test": ["xgboost with Group A+B (+F) features (primary candidate)", "gru with the same expanded feature set (to check if recovery still favors xgboost once B is added)"],
                "primary_validation_metric": "mae24 (primary), mae48, rmse24/48, median_ae24/48 (secondary)",
                "search_budget": "moderate: reuse the existing Phase-12-style validation search budget/process, not a new large exploration -- the ablation already identifies which features matter",
            },
            "icu_stay_time": {
                "feature_groups_to_add": [
                    "B (highest priority, by far the largest gain: ablation shows median AE dropping from ~12.9 to ~5.9 hours from B alone)",
                    "C/D/E/F: ablation showed no material incremental gain beyond B for this task -- deprioritize",
                ],
                "regression_vs_survival_formulation": (
                    "Keep the current log1p regression formulation for Phase 2's first pass, now WITH elapsed time as a "
                    "feature (the current formulation was never tested with elapsed time present, and Part 15 already "
                    "shows this alone closes most of the gap). A discrete-time hazard/AFT survival formulation is a "
                    "reasonable secondary experiment (Part 14) given the generator's own duration mechanics are a "
                    "clipped log-linear-plus-noise process, but is not required to capture the Part-15 gain, which a "
                    "regression model with the missing structural features already captures."
                ),
                "models_to_test": ["gru with Group A+B features (primary candidate, matches current family)", "xgboost with Group A+B features (comparison, given the oracle used xgboost)"],
                "primary_validation_metric": "median_absolute_error hours (primary), mae, rmse (secondary)",
                "search_budget": "moderate: reuse the existing final_v2-style validation search process with the expanded feature set",
            },
            "organ_support": {
                "decision": "TEST_HIGH_VALUE_FEATURE_ADDITIONS_ONLY",
                "rationale": (
                    "Support already has the strongest current absolute performance and the task explicitly warns "
                    "against destabilizing it without validation evidence. Despite that, Part 7/15 found a large "
                    "observable-oracle gain (+34.2% relative AUPRC) driven overwhelmingly by Group B, which is a "
                    "concrete, lawful, low-risk feature addition (not a new model family or objective change). "
                    "Recommendation: add Group B to the existing xgb-support-024-equivalent feature set and re-run "
                    "the existing validation search process; do NOT change the model family, calibration approach, "
                    "or threshold-selection procedure unless Phase-2 validation evidence supports it."
                ),
                "feature_groups_to_add": ["B (primary)", "C/D/E/F only if B's validation gain is confirmed and time permits -- ablation shows their marginal contribution beyond B is small (~0.001-0.001 AUPRC)"],
                "models_to_test": ["xgboost with Group A+B features (primary candidate, same family as today)"],
                "primary_validation_metric": "auprc (primary, matches v1 convention)",
            },
        },
        "generator_v2_required": False,
        "generator_v2_justification": (
            "Not justified. The observable oracle already closes most of the naive-vs-current-model gap using only "
            "lawful observable features; the latent oracle does not materially exceed it; and the identified "
            "stochasticity, while real, does not dominate. Per the task's explicit instruction, generator redesign "
            "is reserved for evidence that the CURRENT generator makes the forecasting problem scientifically "
            "unreasonable -- no such evidence was found."
        ),
        "fresh_dataset_generation_needed_before_phase2": False,
        "fresh_v2_test_strategy": {
            "development_scope": "Continue all Phase-2 feature/model/objective development on the existing frozen TRAIN/VALIDATION partitions only. Never inspect or evaluate against v1 TEST.",
            "when_to_generate_fresh_test": (
                "Only after the v2 feature set, model family, and objective are completely frozen (i.e., at the "
                "point Stage 3-equivalent finalization would occur for v2)."
            ),
            "procedure": [
                "Generate a fresh, independent synthetic cardiac cohort under the SAME frozen generator version/config used for v1 (generator_v2 is not required per the decision above).",
                "Freeze a new subject split and a new sealed test partition BEFORE any v2 model touches it, using the same governance pattern as v1 (G1-equivalent freeze, split isolation, clone-fingerprint checks).",
                "Never inspect the new test partition's outcomes before the final, one-time v2 evaluation.",
                "If a future phase's evidence later justifies generator_v2, freeze generator_v2 first, then generate train_v2/validation_v2/sealed_test_v2 under that frozen generator_v2 contract -- not before.",
            ],
        },
    }
    decision_path = OUT_DIR / "selected_v2_strategy_v1.json"
    decision_path.write_bytes(canonical_json_bytes(decision))
    print("wrote", decision_path, sha256_file(decision_path))


if __name__ == "__main__":
    main()
