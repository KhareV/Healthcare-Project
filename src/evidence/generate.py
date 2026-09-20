"""Generate the Phase-22 Vedant evidence package from registered sources."""

import argparse
import json
from pathlib import Path

from evidence.audit import audit_generated_package
from evidence.diagrams import model_diagrams_markdown
from evidence.report_values import blocked_report_values
from evidence.tables import (
    artifact_hash_rows,
    csv_text,
    load_json,
    reproducibility_rows,
    tuning_budget_rows,
)
from experiments.lineage import (
    ArtifactRecord,
    read_artifact_index,
    read_run_registry,
    register_artifact,
    trace_artifact,
    write_artifact_index,
)
from vedant_infra.hashing import sha256_file


GENERATOR = "src/evidence/generate.py"


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)
    return path


def _json(path: Path, value) -> Path:
    return _write(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def _item(root, config, identifier, title, kind, status, path, inputs=(), run_ids=(), notes=""):
    return {
        "evidence_id": identifier,
        "title": title,
        "type": kind,
        "status": status,
        "output_path": str(path.relative_to(root)),
        "output_sha256": sha256_file(path),
        "generator": GENERATOR,
        "input_artifacts": [str(value.relative_to(root)) for value in inputs],
        "input_hashes": [sha256_file(value) for value in inputs],
        "run_ids": list(run_ids),
        "split": None,
        "task": None,
        "model_family": None,
        "generated_at": config["generated_at"],
        "code_commit": config["code_commit"],
        "notes": notes,
    }


def generate(root: Path):
    root = root.resolve()
    config_path = root / "configs/evidence_v1.json"
    config = load_json(config_path)
    evidence = root / config["evidence_root"]
    xgb_path = root / "configs/search_space_xgb_v1.json"
    gru_path = root / "configs/search_space_gru_v1.json"
    repro_manifest_path = root / "artifacts/reproducibility/reproduction_manifest_v1.json"
    repro_report_path = root / "artifacts/reproducibility/reproduction_report_v1.json"
    artifact_index_path = root / "experiments/artifacts.csv"
    items = []

    source_records = tuple(
        record
        for record in read_artifact_index(artifact_index_path)
        if not record.artifact_id.startswith("phase22:")
    )
    registry_snapshot = evidence / "manifests/artifacts_input_snapshot.csv"
    write_artifact_index(registry_snapshot, source_records)
    items.append(_item(
        root, config, "registry_snapshot", "Phase-22 input artifact-index snapshot",
        "registry_snapshot", "FINAL", registry_snapshot,
        notes="phase22 outputs excluded to prevent self-referential evidence hashes",
    ))

    diagrams = _write(evidence / "diagrams/model_architectures.md", model_diagrams_markdown())
    model_sources = tuple(root / value for value in (
        "src/models/gru.py", "src/models/gru_recovery.py",
        "src/models/gru_icu_time.py", "src/models/gru_support.py",
    ))
    items.append(_item(root, config, "model_diagrams", "Implemented GRU and parity diagrams", "diagram", "FINAL", diagrams, model_sources))

    xgb = load_json(xgb_path)
    gru = load_json(gru_path)
    budget_rows = tuning_budget_rows(xgb, gru)
    budget = _write(
        evidence / "tuning/tuning_budget.csv",
        csv_text(tuple(budget_rows[0]), budget_rows),
    )
    items.append(_item(root, config, "tuning_budget", "Planned budgets and blocked execution evidence", "table", "BLOCKED", budget, (xgb_path, gru_path), notes="planned budgets are not completed-run counts"))
    spaces = _json(
        evidence / "tuning/search_spaces.json",
        {
            "status": "VALIDATION-ONLY",
            "xgboost": xgb,
            "xgboost_source_sha256": sha256_file(xgb_path),
            "gru": gru,
            "gru_source_sha256": sha256_file(gru_path),
            "lstm": {
                "role": "fixed_sensitivity_only",
                "independent_tuning": False,
                "real_execution_status": "BLOCKED",
            },
        },
    )
    items.append(_item(root, config, "search_spaces", "Frozen authoritative search ranges", "table", "VALIDATION-ONLY", spaces, (xgb_path, gru_path)))

    formulas = _write(evidence / "methods/metric_formulas.md", METRIC_TEXT)
    items.append(_item(root, config, "metric_formulas", "Stay-balanced metric formulas", "method", "FINAL", formulas, (root / "src/evaluation/metrics.py", root / "src/evaluation/weights.py")))
    bootstrap = _write(evidence / "methods/grouped_bootstrap.md", BOOTSTRAP_TEXT)
    items.append(_item(root, config, "grouped_bootstrap", "Stay-cluster bootstrap method", "method", "FINAL", bootstrap, (root / "src/evaluation/bootstrap.py", root / "configs/evaluation/bootstrap_v1.json")))
    methods = _write(evidence / "methods/vedant_methods_summary.md", METHODS_TEXT)
    items.append(_item(root, config, "methods_summary", "Vedant methodology summary", "method", "FINAL", methods, (root / "docs/CODEX_PROJECT_CONTEXT_V1.md",)))
    limitations = _write(evidence / "methods/limitations_and_nonclaims.md", LIMITATIONS_TEXT)
    items.append(_item(root, config, "limitations", "Limitations and non-claims", "method", "FINAL", limitations, (root / "docs/CODEX_PROJECT_CONTEXT_V1.md",)))

    blocked_definitions = (
        ("validation_comparison", "comparison/validation_comparison_status.json", "Validation comparison", "BLOCKED — EVIDENCE SOURCE ARTIFACT REQUIRED"),
        ("final_test", "comparison/final_test_status.json", "Final test results", config["final_test_status"]),
        ("selected_models", "manifests/selected_models_status.json", "Selected-model manifest table", "BLOCKED — EVIDENCE SOURCE ARTIFACT REQUIRED"),
        ("calibration", "calibration/calibration_status.json", "Calibration plot/table", "BLOCKED — EVIDENCE SOURCE ARTIFACT REQUIRED"),
        ("ablations", "ablations/ablation_status.json", "Recovery ablation evidence", "BLOCKED — EVIDENCE SOURCE ARTIFACT REQUIRED"),
        ("complete_component", "sensitivity/complete_component_status.json", "Complete-component SOFA evidence", "BLOCKED — EVIDENCE SOURCE ARTIFACT REQUIRED"),
        ("error_analysis", "error_analysis/error_analysis_status.json", "Prespecified error analysis", "BLOCKED — EVIDENCE SOURCE ARTIFACT REQUIRED"),
    )
    for identifier, relative, title, detail in blocked_definitions:
        path = _json(evidence / relative, {"status": "BLOCKED", "detail": detail, "scientific_values": [], "source_split": None})
        items.append(_item(root, config, identifier, title, "result_slot", "BLOCKED", path, notes=detail))

    report_values = _json(
        evidence / "report_values_v1.json",
        blocked_report_values(config["display_precision"]),
    )
    items.append(_item(root, config, "report_values", "Machine-readable report values", "report_values", "BLOCKED", report_values, (config_path,)))

    repro_report = load_json(repro_report_path)
    repro_rows = reproducibility_rows(repro_report)
    repro_table = _write(evidence / "reproducibility/reproducibility.csv", csv_text(tuple(repro_rows[0]), repro_rows))
    items.append(_item(root, config, "reproducibility", "Layered reproducibility status", "table", "BLOCKED", repro_table, (repro_manifest_path, repro_report_path)))

    artifacts = read_artifact_index(registry_snapshot)
    trace_target = next(record for record in artifacts if record.artifact_id.startswith("phase21:report:"))
    trace = trace_artifact(trace_target.artifact_id, artifacts, read_run_registry(root / "experiments/registry.csv"))
    trace["status"] = "BLOCKED_NO_FINAL_RESULT_ROW_DEVELOPMENT_EXAMPLE_ONLY"
    trace_path = _json(evidence / "manifests/registry_trace_example.json", trace)
    items.append(_item(root, config, "registry_trace", "Registry trace example", "lineage", "BLOCKED", trace_path, (registry_snapshot, root / trace_target.artifact_path), notes="development evidence only; final metric lineage unavailable"))

    hash_rows = artifact_hash_rows(root, index_path=registry_snapshot)
    hashes = _write(evidence / "manifests/artifact_hashes.csv", csv_text(tuple(hash_rows[0]), hash_rows))
    items.append(_item(root, config, "artifact_hashes", "Registered artifact hash table", "table", "FINAL", hashes, (registry_snapshot,)))

    invariant_rows = INVARIANT_ROWS
    invariants = _write(evidence / "testing/invariant_matrix.csv", csv_text(tuple(invariant_rows[0]), invariant_rows))
    items.append(_item(root, config, "invariant_matrix", "Test and invariant matrix", "table", "FINAL", invariants, tuple(root / row["test_source"] for row in invariant_rows if row["test_source"])))
    contributions = _write(evidence / "contributions/contribution_table.csv", csv_text(tuple(CONTRIBUTION_ROWS[0]), CONTRIBUTION_ROWS))
    items.append(_item(root, config, "contributions", "Ownership and evidence status", "table", "FINAL", contributions, (root / "docs/CODEX_PROJECT_CONTEXT_V1.md",)))
    sections = _write(evidence / "report/report_section_status_v1.csv", csv_text(tuple(SECTION_ROWS[0]), SECTION_ROWS))
    items.append(_item(root, config, "report_sections", "Report-section evidence status", "table", "FINAL", sections, (root / "docs/CODEX_PROJECT_CONTEXT_V1.md",)))
    slides = _write(evidence / "report/slide_evidence_mapping.csv", csv_text(tuple(SLIDE_ROWS[0]), SLIDE_ROWS))
    items.append(_item(root, config, "slide_mapping", "Slide evidence mapping", "table", "FINAL", slides, (root / "docs/CODEX_PROJECT_CONTEXT_V1.md",)))
    viva = _write(evidence / "viva/viva_vedant.md", VIVA_TEXT)
    items.append(_item(root, config, "viva", "Vedant viva evidence", "viva", "FINAL", viva, (root / "docs/CODEX_PROJECT_CONTEXT_V1.md",)))
    gaps = _write(evidence / "deferred/pulkit_evidence_checklist.md", PULKIT_TEXT)
    items.append(_item(root, config, "pulkit_gaps", "Pulkit evidence gap checklist", "checklist", "DEFERRED", gaps))
    bibliography = _write(evidence / "deferred/bibliography_verification.md", BIBLIOGRAPHY_TEXT)
    items.append(_item(root, config, "bibliography", "Bibliography verification checklist", "checklist", "DEFERRED", bibliography))
    handoff = _json(evidence / "manifests/pulkit_handoff_status.json", {
        "status": "BLOCKED",
        "available": {
            "evidence_manifest": "docs/evidence/models/evidence_manifest_v1.json",
            "reproduction_report": "artifacts/reproducibility/reproduction_report_v1.json",
        },
        "missing": ["selected_models_v1", "real selected checkpoints", "frozen preprocessors", "support calibrator", "support threshold"],
        "pulkit_track": config["pulkit_status"],
    })
    items.append(_item(root, config, "pulkit_handoff", "Pulkit handoff status", "manifest", "BLOCKED", handoff, (repro_report_path,)))

    manifest = {
        "manifest_version": "vedant_evidence_manifest_v1",
        "status": "BLOCKED_REAL_RESULT_EVIDENCE_SOURCES",
        "evidence_root": config["evidence_root"],
        "generated_at": config["generated_at"],
        "code_commit": config["code_commit"],
        "latest_authorized_scientific_state": config["latest_authorized_scientific_state"],
        "display_precision": config["display_precision"],
        "items": items,
        "scientific_results_generated": False,
        "final_test_evidence_generated": False,
        "pulkit_outputs_fabricated": False,
    }
    manifest_path = _json(evidence / "evidence_manifest_v1.json", manifest)
    audit_generated_package(root)
    register_package(root, config_path, manifest_path, items)
    return manifest_path


def register_package(root, config_path, manifest_path, items):
    index = root / "experiments/artifacts.csv"
    runs = read_run_registry(root / "experiments/registry.csv")
    definitions = [("manifest", manifest_path)] + [(item["evidence_id"], root / item["output_path"]) for item in items]
    for identifier, path in definitions:
        digest = sha256_file(path)
        artifact_id = "phase22:" + identifier + ":" + digest[:16]
        existing = read_artifact_index(index)
        match = [record for record in existing if record.artifact_id == artifact_id]
        if match:
            continue
        register_artifact(index, ArtifactRecord(
            artifact_id=artifact_id,
            artifact_path=str(path.relative_to(root)),
            artifact_type="evidence_" + identifier,
            artifact_version="vedant_evidence_v1",
            artifact_sha256=digest,
            producing_run_id="",
            config_hash=sha256_file(config_path),
            generating_script=GENERATOR,
            run_type="development",
            status="registered",
        ), runs, root)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("generate", "audit"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.root.resolve()
    if args.command == "generate":
        print(generate(root))
    else:
        print(json.dumps(audit_generated_package(root), indent=2, sort_keys=True))


METRIC_TEXT = """# Stay-Balanced Metric Evidence

Status: **FINAL** method evidence. Implementation: `stay_balanced_metrics_v1`.

For task or horizon T, stay i has n_i eligible snapshots and each eligible row receives
`w_ij = 1 / n_i`; consequently each represented stay contributes total weight 1.

- Recovery: weighted MAE, weighted RMSE, and same-horizon directional agreement;
  24h and 48h remain separate.
- Remaining ICU stay time: weighted median absolute error in hours, weighted MAE,
  weighted RMSE, and frozen weighted error percentiles.
- New Organ-Support Initiation Risk within 24h: stay-balanced AUPRC, AUROC, Brier,
  sensitivity, specificity, precision, and F1. Selection uses validation AUPRC on raw
  probabilities, not F1.
"""

BOOTSTRAP_TEXT = """# Grouped Bootstrap Evidence

Status: **FINAL** method evidence. Implementation: `stay_cluster_percentile_bootstrap_v1`.

```text
sample ICU stay IDs with replacement
  -> carry every eligible snapshot for each sampled stay copy
  -> assign a distinct identity to duplicate sampled copies
  -> recompute weights so each sampled copy contributes total weight 1
  -> recompute the same point estimator
  -> repeat frozen B times with the frozen seed
  -> use the 2.5th and 97.5th percentiles
```

The resampling unit is the ICU stay, never an individual prediction row. A classification
replicate lacking required class diversity is skipped only for that metric, and valid and
invalid replicate counts are reported. The interval quantifies uncertainty in aggregate
performance. It is not an individual prediction interval or a confidence percentage.

Real B and seed remain unresolved, so no real CI evidence is generated.
"""

METHODS_TEXT = """# Vedant Methods Summary

- Split: subject-disjoint coarse temporal holdout; train eras 2008–2013, validation
  2014–2016, test 2017–2019.
- Prediction grid: t_k = intime + 24h + 6h*k for k=0..11, provided t_k <= outtime-6h.
- Tensor: eight 6-hour bins over (t-48h,t], with distinct padding, observation masks,
  and configured TSLO; no pre-lookback seeding.
- GRU: unidirectional recurrent encoder; recovery has two independent outputs, ICU time
  one log-space output, and support one logit.
- Search governance: planned exact-30 XGBoost and exact-30 GRU candidates per task;
  LSTM is one fixed sensitivity run and not serving-eligible.
- Selection: recovery minimizes validation MAE24; ICU time minimizes validation weighted
  median absolute error in hours; support maximizes validation uncalibrated AUPRC.
- Calibration: only the selected support classifier receives validation-fitted isotonic
  calibration and a validation-F1 threshold.
- Test governance: test access requires active valid G3 and is consumed once.
- Reproducibility: artifact hashes, registry lineage, exact environment, and non-owner
  reproduction are independently reported; absent layers remain blocked.
"""

LIMITATIONS_TEXT = """# Limitations and Non-Claims

This is a retrospective event-time research prototype and retrospective sequential replay.
Prospective documentation/result latency is not modeled. Horizon eligibility may be
informative because patients remaining in ICU differ from those exiting earlier. Evidence
comes from a single-center MIMIC-IV setting without external or bedside validation.
Organ-support initiation partly reflects clinical practice patterns. Regression heads are
point forecasts; no individual prediction intervals are supplied. Outputs are not treatment
advice, and explanation routing—when implemented—is non-causal.
"""

INVARIANT_ROWS = [
    {"invariant": "strict 48h lookback and future perturbation", "status": "IMPLEMENTED", "test_source": "tests/test_tensor_contract.py", "owner": "Vedant/Sanskruti"},
    {"invariant": "padding versus clinical missingness", "status": "IMPLEMENTED", "test_source": "tests/test_dataset.py", "owner": "Vedant"},
    {"invariant": "timestamp start/grid/cap", "status": "IMPLEMENTED", "test_source": "tests/test_timestamps.py", "owner": "Vedant"},
    {"invariant": "recovery independent horizons", "status": "IMPLEMENTED", "test_source": "tests/test_recovery_training.py", "owner": "Vedant"},
    {"invariant": "subject split isolation/hash", "status": "IMPLEMENTED", "test_source": "tests/test_split.py", "owner": "Vedant"},
    {"invariant": "XGBoost/GRU information parity", "status": "IMPLEMENTED", "test_source": "tests/test_search_information_parity.py", "owner": "Vedant"},
    {"invariant": "stay-balanced toy metrics", "status": "IMPLEMENTED", "test_source": "tests/test_stay_weights.py", "owner": "Vedant"},
    {"invariant": "whole-stay grouped bootstrap", "status": "IMPLEMENTED", "test_source": "tests/test_grouped_bootstrap.py", "owner": "Vedant"},
    {"invariant": "validation-only calibration", "status": "IMPLEMENTED", "test_source": "tests/test_calibration_isolation.py", "owner": "Vedant"},
    {"invariant": "mixed-family manifest", "status": "IMPLEMENTED", "test_source": "tests/test_selected_manifest.py", "owner": "Vedant"},
    {"invariant": "support OFF-to-ON/censoring", "status": "CONTRACT_TESTED", "test_source": "tests/test_support_eligibility.py", "owner": "Pulkit"},
    {"invariant": "training-serving equality", "status": "DEFERRED — PULKIT TRACK", "test_source": "", "owner": "Pulkit"},
    {"invariant": "API/replay regression fixture", "status": "DEFERRED — PULKIT TRACK", "test_source": "", "owner": "Pulkit"},
]

CONTRIBUTION_ROWS = [
    {"member": "Vedant", "area": "timestamps, split, tensors/loaders, GRU/LSTM, training, search governance, metrics/bootstrap, selection, calibration, ablations, error analysis, registry, test governance, evidence", "status": "IMPLEMENTED_FRAMEWORK_REAL_RESULTS_BLOCKED", "evidence": "src/data; src/models; src/training; src/evaluation; src/experiments; src/evidence"},
    {"member": "Sanskruti", "area": "extraction, cohort, features/provenance, SOFA, recovery/ICU labels, XGBoost", "status": "UPSTREAM_DEPENDENCY_NOT_PRESENT", "evidence": "DEFERRED_TO_OWNER_HANDOFF"},
    {"member": "Pulkit", "area": "support endpoint, PredictionPipeline, explanations, API, dashboard, packaging/integration", "status": "DEFERRED — PULKIT TRACK", "evidence": "docs/evidence/models/deferred/pulkit_evidence_checklist.md"},
]

SECTION_ROWS = [
    {"section": index, "title": title, "status": status, "vedant_evidence": evidence}
    for index, title, status, evidence in (
        (1, "Introduction", "VEDANT COMPLETE / TEAM PENDING", "limitations/non-claims"),
        (2, "Literature", "BLOCKED", "bibliography verification pending"),
        (3, "Objectives and scope", "VEDANT COMPLETE / TEAM PENDING", "methods/limitations"),
        (4, "Dataset/cohort/split", "SANSKRUTI DEPENDENCY", "split method only"),
        (5, "Temporal representation/features", "SANSKRUTI DEPENDENCY", "tensor method only"),
        (6, "SOFA", "SANSKRUTI DEPENDENCY", "complete-component slot blocked"),
        (7, "Targets/masks", "VEDANT COMPLETE / TEAM PENDING", "method contracts"),
        (8, "Models/tuning", "VEDANT COMPLETE / TEAM PENDING", "diagrams and planned budgets"),
        (9, "Metrics/bootstrap/test isolation", "COMPLETE", "method evidence"),
        (10, "Selection/calibration", "BLOCKED", "real sources absent"),
        (11, "Results", "BLOCKED", "real sources absent"),
        (12, "Error/sensitivity", "BLOCKED", "real sources absent"),
        (13, "Explainability", "PULKIT DEFERRED", "routing only after selection"),
        (14, "Pipeline/API/dashboard", "PULKIT DEFERRED", "gap checklist"),
        (15, "Reproducibility/registry/tests", "VEDANT COMPLETE / TEAM PENDING", "Phase-21 and registry evidence"),
        (16, "Limitations/ethics", "VEDANT COMPLETE / TEAM PENDING", "limitations evidence"),
        (17, "Conclusion", "BLOCKED", "requires real findings"),
        (18, "Appendices", "VEDANT COMPLETE / TEAM PENDING", "hashes/tests/configs"),
    )
]

SLIDE_ROWS = [
    {"slide": i, "topic": topic, "status": status, "evidence": evidence}
    for i, topic, status, evidence in (
        (1, "Problem/research question", "TEAM_PENDING", "methods"),
        (2, "Dataset/cohort/split", "SANSKRUTI_DEPENDENCY", "split methods"),
        (3, "Temporal design", "VEDANT_AVAILABLE", "methods"),
        (4, "Targets", "TEAM_PENDING", "methods"),
        (5, "Leakage/reproducibility", "VEDANT_AVAILABLE", "reproducibility/testing"),
        (6, "Models", "VEDANT_AVAILABLE", "diagrams/tuning"),
        (7, "Evaluation", "VEDANT_AVAILABLE", "metrics/bootstrap"),
        (8, "Calibration/explanation", "BLOCKED_AND_PULKIT_DEFERRED", "blocked calibration/routing"),
        (9, "Replay demo", "DEFERRED — PULKIT TRACK", "gap checklist"),
        (10, "Error analysis/limitations", "BLOCKED_RESULTS_METHODS_AVAILABLE", "error slot/limitations"),
        (11, "Contributions", "AVAILABLE", "contribution table"),
        (12, "Conclusion", "BLOCKED", "real findings required"),
    )
]

VIVA_TEXT = """# Vedant Viva Evidence

1. **Why ICU stay?** It is the episode, prediction clock, remaining-time endpoint, and bootstrap cluster.
2. **First cutoff?** ICU hour 24, then every 6 hours, at most 12 cutoffs.
3. **Why an eight-bin tensor at hour 24?** Pre-ICU positions are padding, never floor-history values.
4. **Lookback?** Only `(t-48h,t]`; older observations cannot seed forward fill.
5. **DeltaSOFA48?** `SOFA(t+48)-SOFA(t)`, independent of DeltaSOFA24.
6. **Baseline SOFA leakage?** SOFA(t) uses only information through t and defines the target baseline.
7. **Split?** Subject-disjoint coarse temporal eras: train 2008–13, validation 2014–16, test 2017–19.
8. **XGB/GRU parity?** Same canonical values, masks, TSLO and permitted statics; representation differs.
9. **Stay balancing?** Each eligible row has weight `1/n_i`, giving each represented stay total weight 1.
10. **Bootstrap?** Resample whole ICU stays with replacement and preserve duplicate stay-copy identity.
11. **Support censoring?** OFF at t is required; incomplete 24h follow-up without an event is censored.
12. **Calibration versus threshold?** Isotonic maps probabilities; the separate validation-F1 threshold creates alerts.
13. **Why G3?** It proves all development decisions are frozen before one-time test access.
14. **Mixed families?** XGBoost versus GRU is selected independently per task.
15. **Bootstrap CI meaning?** Aggregate test-set performance uncertainty, not patient uncertainty.
16. **Selected manifest?** It binds each task to family, run, model, preprocessing, split, label and routing hashes.
17. **Registry lineage?** Results trace through predictions and checkpoints to run/config/data versions and commit.
18. **Main limitations?** Retrospective event time, informative horizon eligibility, single-center data, point forecasts,
no external/bedside validation, and non-causal explanations.
"""

PULKIT_TEXT = """# Pulkit Evidence Gap Checklist

- [ ] PredictionPipeline evidence — **DEFERRED — PULKIT TRACK**
- [ ] API request/response example — **DEFERRED — PULKIT TRACK**
- [ ] Retrospective replay screenshot — **DEFERRED — PULKIT TRACK**
- [ ] TreeSHAP/Integrated Gradients output — **DEFERRED — PULKIT TRACK**
- [ ] Training-serving equality — **DEFERRED — PULKIT TRACK**
- [ ] Serving artifact compatibility — **DEFERRED — PULKIT TRACK**
- [ ] Frozen replay regression fixture — **DEFERRED — PULKIT TRACK**
- [ ] Clean-machine product demonstration — **DEFERRED — PULKIT TRACK**

No screenshot, API output, explanation, or demo result is synthesized by Phase 22.
"""

BIBLIOGRAPHY_TEXT = """# Bibliography Verification Checklist

- [ ] Verify each citation against its original source.
- [ ] Confirm authors, title, venue, year, DOI/URL, and access date where applicable.
- [ ] Verify MIMIC-IV and MIMIC-Code version references.
- [ ] Ensure no citation is inferred from memory or generated without source review.

No bibliography entry is created by this phase.
"""


if __name__ == "__main__":
    main()
