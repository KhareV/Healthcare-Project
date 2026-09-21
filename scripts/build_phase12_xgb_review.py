#!/usr/bin/env python3
"""Generate the governed human-readable Phase-12 search review."""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SEARCH_ROOT = ROOT / "artifacts/search/xgb/phase12"
OUTPUT = ROOT / "docs/sanskruti/PHASE12_XGBOOST_VALIDATION_SEARCH_REVIEW.md"


def _load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _value(value):
    if value is None: return "UNAVAILABLE_BY_FROZEN_METRIC_SEMANTICS"
    if isinstance(value, float): return f"{value:.10g}"
    return str(value)


def _table(task, rows):
    if task == "recovery":
        columns = ("candidate_id", "status", "mae24", "mae48", "rmse24", "rmse48", "median_absolute_error24", "median_absolute_error48")
    elif task == "icu_time":
        columns = ("candidate_id", "status", "median_absolute_error_hours", "mae_hours", "rmse_hours")
    else:
        columns = ("candidate_id", "status", "auprc", "auroc", "brier")
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    lines.extend("| " + " | ".join(_value(row.get(column)) for column in columns) + " |" for row in rows)
    return "\n".join(lines)


def main() -> None:
    master = _load(SEARCH_ROOT / "manifests/phase12_search_manifest_v1.json")
    best = _load(ROOT / master["best_xgb_manifest_path"])
    reproducibility = _load(SEARCH_ROOT / "winner_reproducibility_audit_v1.json")
    summaries = {task: _load(SEARCH_ROOT / f"{task}_validation_summary.json")["candidates"] for task in ("recovery", "icu_time", "organ_support")}
    failed = {task: [row["candidate_id"] for row in rows if row["status"] != "COMPLETE"] for task, rows in summaries.items()}
    text = f"""# Sanskruti Rebuild Phase 12 — Governed XGBoost Validation Search Review

## Scope and outcome

The first scientific XGBoost search on the final synthetic adult cardiac benchmark completed under `{master['search_version']}`. It used training rows for fitting and validation rows for early stopping and within-family ranking. The sealed test partition was not loaded, predicted, summarized, or evaluated. These results characterize a designed synthetic benchmark and are not evidence of bedside accuracy, clinical effectiveness, treatment effects, or real cardiac-population generalization.

## Frozen prerequisite and execution identity

- Phase-11 manifest: `{master['phase11_manifest_path']}` (`{master['phase11_manifest_sha256']}`)
- implementation commit: `{master['implementation_commit']}`
- scientific execution commit: `{master['execution_commit']}`
- search-space canonical SHA-256: `{master['search_space_sha256']}`
- scientific environment SHA-256: `{master['environment_sha256']}`
- split SHA-256: `{master['split_sha256']}`
- feature schema SHA-256: `{master['feature_schema_sha256']}`
- flat feature-map SHA-256: `{master['flat_feature_map_sha256']}`
- preprocessor SHA-256: `{master['preprocessor_sha256']}`

## Sampler, budget, retry, and early stopping

The sampler was `{master['sampler']}` with master seed `{master['master_search_seed']}` and task seeds `{json.dumps(master['task_sampler_seeds'], sort_keys=True)}`. All three independent candidate lists were materialized and hashed before candidate 1. Each contains exactly 30 unique configurations; recovery therefore produced 30 scientific candidate bundles and 60 horizon estimators, not a 60-configuration search. Maximum attempts per candidate were `{master['max_attempts']}`. Early stopping used 50 rounds and ordinary row-weighted internal monitors; authoritative ranking used the existing stay-balanced external evaluators.

Candidate-list hashes:

{json.dumps({task: value['candidate_list_hash'] for task, value in master['candidate_manifests'].items()}, indent=2, sort_keys=True)}

Failed candidates: `{json.dumps(failed, sort_keys=True)}`. Retry details remain in the registry; retries do not consume new candidate slots.

## Frozen ranking and winners

- recovery: minimize validation stay-balanced MAE24 in original delta-SOFA units; tie-break MAE48, RMSE24, candidate ID. Winner: `{best['tasks']['recovery']['candidate_id']}`.
- ICU time: minimize validation stay-balanced weighted median absolute error in remaining-current-episode hours; tie-break MAE, RMSE, candidate ID. Winner: `{best['tasks']['icu_time']['candidate_id']}`.
- organ support: maximize validation stay-balanced AUPRC from raw uncalibrated probabilities; tie-break Brier, AUROC, candidate ID. Winner: `{best['tasks']['organ_support']['candidate_id']}`.

This is within-XGBoost-family validation selection only. It is not XGBoost-versus-GRU selection, serving selection, calibration, threshold selection, or final-test evaluation. `selected_models_v1.json` was not created or changed.

## Recovery candidates

{_table('recovery', summaries['recovery'])}

Directional agreement remains unavailable when exact-zero target/prediction semantics trigger the existing frozen metric blocker; no new direction rule was invented.

## ICU-time candidates

{_table('icu_time', summaries['icu_time'])}

## Organ-support candidates

{_table('organ_support', summaries['organ_support'])}

All support values are raw and uncalibrated. No threshold-dependent metric participated in ranking.

## Reproducibility, lineage, and isolation

Winner refits reused the same candidate identities without consuming scientific slots. Reproducibility status: `{reproducibility['status']}`. Prediction and metric differences are recorded in `{str((SEARCH_ROOT / 'winner_reproducibility_audit_v1.json').relative_to(ROOT))}`. Every completed candidate has registry, native-model, validation-prediction, metric, split, feature, preprocessing, environment, and candidate-list lineage. The Phase-12 audit verifies exact budget, artifact hashes, known producers, no unauthorized candidates, no scientific orphans, and `test_accessed=false`.

## Phase-13 boundary

Phase 12 does not claim G1 acceptance. Phase 13 owns consolidated data QA, leakage attacks, reproducibility acceptance, and G1/data-freeze readiness. No Phase-13 acceptance decision is implied here.
"""
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(text, encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
